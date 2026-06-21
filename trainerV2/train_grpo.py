# -*- coding: utf-8 -*-
"""
MiniMind MLX GRPO（Group Relative Policy Optimization）训练脚本

基于 MLX 框架实现的 GRPO 强化学习训练。GRPO 通过组内相对优势估计
优化语言模型的生成策略，无需训练 Critic 模型。

核心思想：
- 每个 prompt 生成多个候选回答
- 组内计算相对优势（reward 标准化）
- 使用 PPO clip 目标函数更新策略

使用方式：
    python trainerV2/train_grpo.py --data_path ../.dataset/rlaif.jsonl
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from trainerV2.model_mlx import MiniMindConfig, MiniMindForCausalLM
from trainerV2.trainer_utils import (
    format_duration, get_lr, logger, save_checkpoint,
    setup_seed, setup_wandb, clip_grad_norm, log_model_params,
)
from trainerV2.reward_utils import compute_repetition_penalty


# =====================================================================================
# Rollout 引擎（MLX 原生）
# =====================================================================================

def rollout_generate(model, input_ids: mx.array, max_new_tokens: int = 256,
                     temperature: float = 0.8, num_generations: int = 4) -> dict:
    """使用当前策略模型生成候选回答

    Args:
        model: 策略模型
        input_ids: prompt token IDs [B, P]
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度
        num_generations: 每个 prompt 的生成数量

    Returns:
        dict 包含 output_ids, completion_ids, completions, per_token_logps
    """
    B, P = input_ids.shape
    # 复制 prompt（每个生成 num_generations 次）
    expanded_ids = mx.repeat(input_ids, num_generations, axis=0)  # [B*G, P]

    # 自回归生成
    generated = expanded_ids
    all_logps = []

    for _ in range(max_new_tokens):
        output = model(generated)
        next_logits = output.logits[:, -1, :] / temperature

        # 采样
        probs = mx.softmax(next_logits, axis=-1)
        next_token = mx.random.categorical(mx.log(probs + 1e-10))[:, None]

        # 计算该 token 的 log prob
        log_probs = next_logits - mx.logsumexp(next_logits, axis=-1, keepdims=True)
        token_logp = mx.take_along_axis(log_probs, next_token, axis=-1)
        all_logps.append(token_logp)

        generated = mx.concatenate([generated, next_token], axis=-1)
        mx.eval(generated)

        # 简单 EOS 检查（token_id == 2）
        if mx.all(next_token.reshape(-1) == 2).item():
            break

    completion_ids = generated[:, P:]  # [B*G, R]
    per_token_logps = mx.concatenate(all_logps, axis=-1) if all_logps else mx.zeros((B * num_generations, 0))

    return {
        "output_ids": generated,
        "completion_ids": completion_ids,
        "per_token_logps": per_token_logps,
    }


def compute_per_token_logps(model, input_ids: mx.array, n_keep: int) -> mx.array:
    """计算序列尾部 n_keep 个 token 的对数概率"""
    output = model(input_ids)
    logits = output.logits[:, -(n_keep + 1):-1, :]
    labels = input_ids[:, -n_keep:]

    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    B, L, V = log_probs.shape
    batch_idx = mx.arange(B)[:, None]
    seq_idx = mx.arange(L)[None, :]
    per_token_logps = log_probs[batch_idx, seq_idx, labels]
    return per_token_logps


# =====================================================================================
# GRPO 奖励计算
# =====================================================================================

def calculate_rewards(responses: list, device=None) -> mx.array:
    """计算生成样本的综合奖励

    奖励维度：
    1. 长度奖励：20~800 字符 +0.5
    2. 思考奖励：含合理思考内容 +1.0
    3. 重复惩罚：n-gram 重复扣分
    """
    rewards = []
    for response in responses:
        score = 0.0
        # 长度奖励
        score += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
        # 思考奖励
        if '</think>' in response:
            thinking, answer = response.split('</think>', 1)
            score += 1.0 if 20 <= len(thinking.strip()) <= 300 else -0.5
            score += 0.25 if response.count('</think>') == 1 else -0.25
            response = answer.strip()
        # 重复惩罚
        score -= compute_repetition_penalty(response)
        rewards.append(score)

    return mx.array(rewards)


# =====================================================================================
# GRPO 训练循环
# =====================================================================================

def grpo_train_step(model, ref_model, optimizer, prompt_ids: mx.array,
                    tokenizer, args) -> dict:
    """GRPO 单步训练

    流程：
    1. Rollout: 生成 num_generations 个候选
    2. 计算奖励
    3. 组内标准化优势
    4. PPO clip 目标函数更新策略
    """
    B = prompt_ids.shape[0]
    G = args.num_generations

    # 1. Rollout
    rollout = rollout_generate(model, prompt_ids, args.max_gen_len,
                               temperature=0.8, num_generations=G)
    completion_ids = rollout["completion_ids"]
    output_ids = rollout["output_ids"]
    n_keep = completion_ids.shape[1]

    # 解码文本用于奖励计算
    completions = []
    for i in range(B * G):
        ids = completion_ids[i].tolist()
        text = tokenizer.decode(ids, skip_special_tokens=True)
        completions.append(text)

    # 2. 计算奖励
    rewards = calculate_rewards(completions)  # [B*G]

    # 3. 组内标准化优势
    rewards_grouped = rewards.reshape(B, G)
    mean_r = mx.mean(rewards_grouped, axis=1, keepdims=True)
    std_r = mx.maximum(mx.std(rewards_grouped, axis=1, keepdims=True), mx.array(1e-6))
    advantages = ((rewards_grouped - mean_r) / std_r).reshape(-1)  # [B*G]

    # 4. 计算当前策略和参考策略的 log probs
    old_logps = mx.stop_gradient(rollout["per_token_logps"])

    # 定义 GRPO 损失函数
    def grpo_loss_fn(model):
        # 当前策略的 log probs
        cur_logps = compute_per_token_logps(model, output_ids, n_keep)

        # PPO clip 目标
        log_ratio = cur_logps - old_logps
        ratio = mx.exp(mx.sum(log_ratio, axis=-1))  # [B*G]

        # Clipped surrogate
        clip_ratio = mx.clip(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon)
        obj1 = ratio * advantages
        obj2 = clip_ratio * advantages
        policy_loss = -mx.mean(mx.minimum(obj1, obj2))

        # KL 惩罚（相对于参考模型）
        ref_logps = mx.stop_gradient(compute_per_token_logps(ref_model, output_ids, n_keep))
        kl = mx.mean(mx.sum(cur_logps - ref_logps, axis=-1))

        return policy_loss + args.kl_coef * kl

    loss, grads = nn.value_and_grad(model, grpo_loss_fn)(model)
    grads = clip_grad_norm(grads, args.grad_clip)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    return {
        "loss": loss.item(),
        "reward_mean": mx.mean(rewards).item(),
        "reward_std": mx.std(rewards).item(),
    }


# =====================================================================================
# 主程序
# =====================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind MLX GRPO")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", type=str, default="grpo_mlx")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="../.dataset/rlaif.jsonl")
    parser.add_argument("--from_weight", type=str, default="full_sft_mlx")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-MLX-GRPO")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for attr in ('save_dir', 'data_path'):
        path = getattr(args, attr)
        if not os.path.isabs(path):
            setattr(args, attr, os.path.normpath(os.path.join(script_dir, path)))

    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        num_key_value_heads=2,
    )

    model = MiniMindForCausalLM(config)
    ref_model = MiniMindForCausalLM(config)
    mx.eval(model.parameters())
    mx.eval(ref_model.parameters())

    weight_path = os.path.join(args.save_dir, f"{args.from_weight}.safetensors")
    if os.path.exists(weight_path):
        weights = nn.utils.load_safetensors(weight_path)
        model.load_weights(list(weights.items()))
        ref_model.load_weights(list(weights.items()))
        logger(f"Loaded weights: {weight_path}")

    ref_model.freeze()
    log_model_params(model)

    tokenizer_path = os.path.normpath(os.path.join(script_dir, '..', 'model'))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 加载 RLAIF 数据
    import json
    prompts = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'prompt' in data:
                prompts.append(data['prompt'])
            elif 'conversations' in data:
                prompts.append(tokenizer.apply_chat_template(
                    data['conversations'][:1], tokenize=False, add_generation_prompt=True
                ))
    logger(f"Loaded {len(prompts)} prompts")

    optimizer = optim.AdamW(learning_rate=args.learning_rate)
    wandb = setup_wandb(args, run_name_prefix="MiniMind-MLX-GRPO")

    logger(f"\n🚀 GRPO Training started (G={args.num_generations}, clip={args.clip_epsilon})")

    step = 0
    for epoch in range(args.epochs):
        setup_seed(42 + epoch)
        import random
        random.shuffle(prompts)

        for i in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[i:i + args.batch_size]
            encoded = tokenizer(batch_prompts, return_tensors="np", padding=True,
                                truncation=True, max_length=args.max_seq_len)
            prompt_ids = mx.array(encoded["input_ids"])

            metrics = grpo_train_step(model, ref_model, optimizer, prompt_ids, tokenizer, args)
            step += 1

            if step % args.log_interval == 0:
                logger(
                    f"Step {step} | loss: {metrics['loss']:.4f} | "
                    f"reward: {metrics['reward_mean']:.3f}±{metrics['reward_std']:.3f}"
                )
                if wandb:
                    wandb.log(metrics)

            if step % args.save_interval == 0:
                save_checkpoint(model, optimizer, args.save_dir, args.save_weight,
                                epoch=epoch, step=step)

    logger(f"🎉 GRPO Training complete!")
