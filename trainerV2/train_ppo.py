# -*- coding: utf-8 -*-
"""
MiniMind MLX PPO (Proximal Policy Optimization) 近端策略优化训练脚本

基于 MLX 框架实现的 PPO 强化学习训练：
1. Actor-Critic 架构：Actor 生成回答，Critic 估计价值函数
2. GAE (Generalized Advantage Estimation)：广义优势估计
3. KL 散度约束：通过参考模型防止策略偏离
4. 早停机制：当 KL 散度超过阈值时停止更新

与 PyTorch 版本的差异：
- 无需 DDP 分布式（MLX 单机 Apple Silicon）
- 函数式梯度（nn.value_and_grad）取代 backward()
- 统一内存无需 device 管理
- 无需 autocast / GradScaler

使用方式：
    python trainerV2/train_ppo.py --data_path ../.dataset/rlaif.jsonl
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from trainerV2.model_mlx import MiniMindConfig, MiniMindForCausalLM
from trainerV2.trainer_utils import (
    format_duration, get_lr, logger, save_checkpoint, load_checkpoint,
    setup_seed, setup_wandb, clip_grad_norm, log_model_params,
)
from trainerV2.reward_utils import compute_repetition_penalty, compute_batch_rewards
from trainerV2.rollout_engine import (
    MLXRolloutEngine, RolloutResult, compute_per_token_logps, create_rollout_engine,
)


# =====================================================================================
# Critic 模型
# =====================================================================================

class CriticModel(nn.Module):
    """价值函数估计器（Value Function Estimator）

    在 Actor-Critic 架构中，Critic 负责估计状态价值 V(s)。
    复用 MiniMind 的 Transformer 主干，将 lm_head 替换为 value_head。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 复用完整的 CausalLM 模型（获取 Transformer 主干）
        self.backbone = MiniMindForCausalLM(config)
        # 替换输出头：hidden_size -> 1（标量价值）
        self.value_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.config = config

    def __call__(self, input_ids: mx.array) -> mx.array:
        """前向传播：计算每个 token 位置的状态价值

        Args:
            input_ids: [B, L]

        Returns:
            values: [B, L]，每个位置的价值估计
        """
        # 获取 backbone 的隐藏状态（跳过 lm_head）
        output = self.backbone(input_ids)
        # 用最后一层的隐藏状态（通过重新前向 backbone.model）
        # 更优雅的做法：直接访问内部 transformer block
        hidden = self.backbone.model(input_ids)
        hidden = self.backbone.norm(hidden)
        values = self.value_head(hidden).squeeze(-1)  # [B, L]
        return values


# =====================================================================================
# 奖励计算
# =====================================================================================

def calculate_rewards(prompts: list, responses: list, reward_model=None) -> mx.array:
    """计算生成样本的综合奖励

    奖励维度：
    1. 长度奖励：20~800 字符 +0.5
    2. 思考奖励：含合理思考内容 +1.0
    3. 重复惩罚：n-gram 重复扣分
    4. Reward Model 评分（可选）
    """
    return compute_batch_rewards(
        prompts, responses,
        reward_model=reward_model,
        num_generations=1,
    )


# =====================================================================================
# GAE (Generalized Advantage Estimation)
# =====================================================================================

def compute_gae(rewards_at_last: mx.array, values: mx.array,
                resp_mask: mx.array, resp_lengths: mx.array,
                gamma: float = 1.0, lam: float = 0.95) -> tuple:
    """计算 GAE 优势估计和目标回报值

    GAE 公式: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)

    Args:
        rewards_at_last: [B] 外部奖励（仅在最后一个 token 处）
        values: [B, R] 每个 response token 的价值估计
        resp_mask: [B, R] 有效 token 掩码
        resp_lengths: [B] 每个样本的有效 response 长度
        gamma: 折扣因子
        lam: GAE lambda

    Returns:
        (advantages, returns): 均为 [B, R]
    """
    B, R = values.shape

    # 构建 token 级奖励：仅在最后一个有效 token 加上外部奖励
    token_rewards = mx.zeros_like(values)
    last_idx = (resp_lengths - 1).astype(mx.int32)  # [B]
    for i in range(B):
        idx = last_idx[i].item()
        if 0 <= idx < R:
            token_rewards = token_rewards.at[i, idx].add(rewards_at_last[i])

    # 反向递推 GAE
    advantages = mx.zeros_like(values)
    lastgaelam = mx.zeros((B,))

    for t in range(R - 1, -1, -1):
        next_value = values[:, t + 1] if t < R - 1 else mx.zeros((B,))
        delta = token_rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages = advantages.at[:, t].add(lastgaelam)

    returns = advantages + values

    # 标准化优势
    masked_adv = advantages * resp_mask
    adv_sum = mx.sum(masked_adv)
    mask_sum = mx.maximum(mx.sum(resp_mask), mx.array(1.0))
    adv_mean = adv_sum / mask_sum
    adv_var = mx.sum((advantages - adv_mean) ** 2 * resp_mask) / mask_sum
    advantages = (advantages - adv_mean) * mx.rsqrt(adv_var + 1e-8) * resp_mask

    return advantages, returns


# =====================================================================================
# PPO 训练循环
# =====================================================================================

def ppo_train_step(actor_model, critic_model, ref_model,
                   actor_optimizer, critic_optimizer,
                   rollout_engine: MLXRolloutEngine,
                   prompt_ids: mx.array, prompts: list,
                   tokenizer, args, reward_model=None) -> dict:
    """PPO 单步训练

    完整流程：
    1. Rollout：使用当前 Actor 生成回答
    2. 计算奖励
    3. GAE 优势估计
    4. Mini-batch PPO 多轮更新（Actor + Critic）
    """
    B, P = prompt_ids.shape

    # ==================== Phase 1: Rollout ====================
    rollout = rollout_engine.rollout(
        prompt_ids, num_generations=1,
        max_new_tokens=args.max_gen_len, temperature=0.8,
    )
    output_ids = rollout.output_ids  # [B, P+R]
    completions = rollout.completions
    R = output_ids.shape[1] - P

    if R <= 0:
        return {"reward": 0.0, "kl": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

    # ==================== Phase 2: 计算奖励 ====================
    rewards = calculate_rewards(prompts, completions, reward_model)

    # ==================== Phase 3: 计算旧策略 logp 和 values ====================
    old_logps = compute_per_token_logps(actor_model, output_ids, R)  # [B, R]
    old_logps = mx.stop_gradient(old_logps)

    old_values = critic_model(output_ids)[:, P - 1:-1]  # [B, R]
    old_values = mx.stop_gradient(old_values)

    # 参考模型 logp
    ref_logps = compute_per_token_logps(ref_model, output_ids, R)
    ref_logps = mx.stop_gradient(ref_logps)

    # 构建 response mask（EOS 之后不参与）
    resp_ids = output_ids[:, P:]  # [B, R]
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 2
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else 0
    resp_mask = (resp_ids != pad_id).astype(mx.float32)

    # 计算 response 长度（到第一个 EOS）
    is_eos = (resp_ids == eos_id) & (resp_mask > 0)
    resp_lengths = mx.sum(resp_mask, axis=1).astype(mx.int32)
    for i in range(B):
        eos_positions = mx.where(is_eos[i])[0]
        if len(eos_positions) > 0:
            resp_lengths = resp_lengths.at[i].add(
                mx.array(eos_positions[0].item() + 1) - resp_lengths[i]
            )

    # ==================== Phase 4: GAE ====================
    advantages, returns = compute_gae(
        rewards, old_values, resp_mask, resp_lengths,
        gamma=args.gamma, lam=args.lam,
    )
    advantages = mx.stop_gradient(advantages)
    returns = mx.stop_gradient(returns)

    # ==================== Phase 5: Mini-batch PPO 更新 ====================
    policy_loss_sum, value_loss_sum, kl_sum = 0.0, 0.0, 0.0
    update_count = 0
    stop_ppo = False

    for ppo_epoch in range(args.ppo_update_iters):
        if stop_ppo:
            break

        # Actor 损失函数
        def actor_loss_fn(actor):
            cur_logps = compute_per_token_logps(actor, output_ids, R)
            log_ratio = cur_logps - old_logps
            ratio = mx.exp(log_ratio)

            # PPO clip
            clipped_ratio = mx.clip(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon)
            obj1 = -advantages * ratio
            obj2 = -advantages * clipped_ratio
            policy_loss = mx.sum(mx.maximum(obj1, obj2) * resp_mask) / mx.maximum(mx.sum(resp_mask), mx.array(1.0))

            # KL 惩罚
            kl_penalty = mx.sum(
                (mx.exp(ref_logps - cur_logps) - (ref_logps - cur_logps) - 1) * resp_mask
            ) / mx.maximum(mx.sum(resp_mask), mx.array(1.0))

            return policy_loss + args.kl_coef * kl_penalty

        # Critic 损失函数
        def critic_loss_fn(critic):
            cur_values = critic(output_ids)[:, P - 1:-1]
            # Clipped value loss
            v_clipped = mx.clip(
                cur_values,
                old_values - args.cliprange_value,
                old_values + args.cliprange_value,
            )
            loss1 = (cur_values - returns) ** 2
            loss2 = (v_clipped - returns) ** 2
            value_loss = 0.5 * mx.sum(mx.maximum(loss1, loss2) * resp_mask) / mx.maximum(mx.sum(resp_mask), mx.array(1.0))
            return value_loss

        # Actor 梯度计算与更新
        actor_loss_and_grad = nn.value_and_grad(actor_model, actor_loss_fn)
        actor_loss_val, actor_grads = actor_loss_and_grad(actor_model)
        actor_grads = clip_grad_norm(actor_grads, args.grad_clip)
        actor_optimizer.update(actor_model, actor_grads)
        mx.eval(actor_model.parameters(), actor_optimizer.state)

        # Critic 梯度计算与更新
        critic_loss_and_grad = nn.value_and_grad(critic_model, critic_loss_fn)
        critic_loss_val, critic_grads = critic_loss_and_grad(critic_model)
        critic_grads = clip_grad_norm(critic_grads, args.grad_clip)
        critic_optimizer.update(critic_model, critic_grads)
        mx.eval(critic_model.parameters(), critic_optimizer.state)

        # 统计
        policy_loss_sum += actor_loss_val.item()
        value_loss_sum += critic_loss_val.item()
        update_count += 1

        # KL 早停检查
        with mx.no_grad():
            cur_logps_check = compute_per_token_logps(actor_model, output_ids, R)
            approx_kl = mx.mean(0.5 * (cur_logps_check - old_logps) ** 2).item()
            kl_sum += approx_kl
            if approx_kl > args.early_stop_kl:
                stop_ppo = True

    return {
        "reward": rewards.mean().item() if hasattr(rewards, 'mean') else float(np.mean(rewards)),
        "policy_loss": policy_loss_sum / max(update_count, 1),
        "value_loss": value_loss_sum / max(update_count, 1),
        "kl": kl_sum / max(update_count, 1),
        "avg_resp_len": float(mx.sum(resp_mask).item() / B),
    }


# =====================================================================================
# 主训练流程
# =====================================================================================

def train(args):
    """PPO 完整训练流程"""
    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    # 模型配置
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        num_key_value_heads=2,
    )

    # 加载 tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("../model", trust_remote_code=True)

    # ========== Actor 模型 ==========
    actor_model = MiniMindForCausalLM(config)
    weight_path = os.path.join(args.save_dir, f"{args.from_weight}.safetensors")
    if os.path.exists(weight_path):
        weights = mx.load(weight_path)
        actor_model.load_weights(list(weights.items()))
        logger(f"Actor loaded from {weight_path}")

    # ========== 参考模型（冻结） ==========
    ref_model = MiniMindForCausalLM(config)
    if os.path.exists(weight_path):
        ref_model.load_weights(list(weights.items()))
    ref_model.freeze()
    logger("Reference model frozen")

    # ========== Critic 模型 ==========
    critic_model = CriticModel(config)
    if os.path.exists(weight_path):
        # 用 Actor 权重初始化 Critic backbone
        critic_model.backbone.load_weights(list(weights.items()))
    logger("Critic model initialized")

    # ========== 优化器 ==========
    actor_optimizer = optim.AdamW(learning_rate=args.learning_rate)
    critic_optimizer = optim.AdamW(learning_rate=args.critic_learning_rate)

    # ========== Rollout 引擎 ==========
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 2
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else 0
    rollout_engine = create_rollout_engine(
        actor_model, tokenizer, eos_token_id=eos_id, pad_token_id=pad_id,
    )

    # ========== 数据加载 ==========
    from dataset.lm_dataset import RLAIFDataset
    train_ds = RLAIFDataset(
        args.data_path, tokenizer,
        max_length=(args.max_seq_len + args.max_gen_len),
    )
    logger(f"Dataset loaded: {len(train_ds)} samples")
    log_model_params(actor_model)

    # ========== wandb ==========
    wandb = setup_wandb(args) if args.use_wandb else None

    # ========== 训练循环 ==========
    total_steps = math.ceil(len(train_ds) / args.batch_size) * args.epochs
    global_step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        setup_seed(42 + epoch)
        indices = np.random.permutation(len(train_ds)).tolist()

        for batch_start in range(0, len(train_ds), args.batch_size):
            global_step += 1
            batch_indices = indices[batch_start:batch_start + args.batch_size]
            batch = [train_ds[i] for i in batch_indices]

            # 获取 prompt
            prompts = [item["prompt"] for item in batch]
            enc = tokenizer(
                prompts, return_tensors="np", padding=True,
                truncation=True, max_length=args.max_seq_len,
            )
            prompt_ids = mx.array(enc["input_ids"])

            # 动态学习率
            lr = get_lr(global_step, total_steps, args.learning_rate)
            actor_optimizer.learning_rate = lr
            critic_optimizer.learning_rate = lr * (args.critic_learning_rate / args.learning_rate)

            # PPO 训练步
            metrics = ppo_train_step(
                actor_model, critic_model, ref_model,
                actor_optimizer, critic_optimizer,
                rollout_engine, prompt_ids, prompts,
                tokenizer, args,
            )

            # 同步策略到 rollout 引擎
            rollout_engine.update_policy(actor_model)

            # 日志
            if global_step % args.log_interval == 0:
                elapsed = time.time() - start_time
                logger(
                    f"Epoch:[{epoch + 1}/{args.epochs}]({global_step}/{total_steps}), "
                    f"Reward: {metrics['reward']:.4f}, "
                    f"PolicyLoss: {metrics['policy_loss']:.4f}, "
                    f"ValueLoss: {metrics['value_loss']:.4f}, "
                    f"KL: {metrics['kl']:.6f}, "
                    f"AvgLen: {metrics.get('avg_resp_len', 0):.1f}, "
                    f"LR: {lr:.2e}, "
                    f"Time: {format_duration(elapsed)}"
                )

                if wandb:
                    wandb.log({
                        "reward": metrics["reward"],
                        "policy_loss": metrics["policy_loss"],
                        "value_loss": metrics["value_loss"],
                        "kl": metrics["kl"],
                        "learning_rate": lr,
                    })

            # 保存 checkpoint
            if global_step % args.save_interval == 0:
                save_checkpoint(
                    actor_model, actor_optimizer,
                    args.save_dir, args.save_weight,
                    epoch=epoch, step=global_step, config=config,
                )

    # 最终保存
    save_checkpoint(
        actor_model, actor_optimizer,
        args.save_dir, args.save_weight,
        epoch=args.epochs, step=global_step, config=config,
    )
    total_time = time.time() - start_time
    logger(f"Training complete! Total time: {format_duration(total_time)}")


# =====================================================================================
# 命令行入口
# =====================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind MLX PPO Training")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", type=str, default="ppo_actor")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument("--critic_learning_rate", type=float, default=5e-7)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--max_gen_len", type=int, default=1024)
    parser.add_argument("--data_path", type=str, default="../.dataset/rlaif.jsonl")
    # PPO 特有参数
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--cliprange_value", type=float, default=0.2)
    parser.add_argument("--ppo_update_iters", type=int, default=2)
    parser.add_argument("--early_stop_kl", type=float, default=0.25)
    parser.add_argument("--mini_batch_size", type=int, default=2)
    parser.add_argument("--from_weight", type=str, default="full_sft")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO-MLX")
    parser.add_argument("--thinking_ratio", type=float, default=0.9)
    args = parser.parse_args()

    train(args)
