# -*- coding: utf-8 -*-
"""
MiniMind MLX DPO（Direct Preference Optimization）训练脚本

基于 MLX 框架实现的 DPO 偏好对齐训练。DPO 通过直接优化策略模型使其更偏好
chosen 响应而非 rejected 响应，无需显式训练奖励模型。

算法核心：
L_DPO = -E[log σ(β * (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x))))]

使用方式：
    python trainerV2/train_dpo.py --data_path ../.dataset/dpo.jsonl --beta 0.15
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np

from trainerV2.model_mlx import MiniMindConfig, MiniMindForCausalLM
from trainerV2.trainer_utils import (
    format_duration, get_lr, logger, save_checkpoint,
    load_checkpoint, setup_seed, setup_wandb, clip_grad_norm, log_model_params,
)


# =====================================================================================
# DPO 数据集
# =====================================================================================

class DPODataset:
    """DPO 偏好对数据集

    每条样本包含 (prompt, chosen, rejected) 三元组。
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.samples = []

        logger(f"Loading DPO data from: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if 'chosen' in data and 'rejected' in data:
                    self.samples.append(data)
        logger(f"Loaded {len(self.samples):,} preference pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # 编码 chosen 和 rejected
        chosen_text = self.tokenizer.apply_chat_template(
            item['chosen'], tokenize=False, add_generation_prompt=False
        )
        rejected_text = self.tokenizer.apply_chat_template(
            item['rejected'], tokenize=False, add_generation_prompt=False
        )

        chosen_ids = self.tokenizer.encode(chosen_text)[:self.max_length]
        rejected_ids = self.tokenizer.encode(rejected_text)[:self.max_length]

        # Padding 到相同长度
        max_len = max(len(chosen_ids), len(rejected_ids))
        pad_id = self.tokenizer.pad_token_id or 0

        chosen_ids = chosen_ids + [pad_id] * (max_len - len(chosen_ids))
        rejected_ids = rejected_ids + [pad_id] * (max_len - len(rejected_ids))

        return {
            'chosen_ids': np.array(chosen_ids, dtype=np.int32),
            'rejected_ids': np.array(rejected_ids, dtype=np.int32),
        }


class DPODataIterator:
    """DPO 专用数据迭代器"""

    def __init__(self, dataset: DPODataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))
        if shuffle:
            import random
            random.shuffle(self.indices)

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batch = []
        for idx in self.indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []
        if batch:
            yield self._collate(batch)

    def _collate(self, items):
        # 找到 batch 内最大长度并统一 padding
        max_len = max(max(len(item['chosen_ids']), len(item['rejected_ids'])) for item in items)
        pad_id = 0

        chosen_batch = []
        rejected_batch = []
        for item in items:
            c = list(item['chosen_ids']) + [pad_id] * (max_len - len(item['chosen_ids']))
            r = list(item['rejected_ids']) + [pad_id] * (max_len - len(item['rejected_ids']))
            chosen_batch.append(c)
            rejected_batch.append(r)

        return {
            'chosen_ids': mx.array(np.array(chosen_batch, dtype=np.int32)),
            'rejected_ids': mx.array(np.array(rejected_batch, dtype=np.int32)),
        }


# =====================================================================================
# DPO 算法核心
# =====================================================================================

def compute_log_probs(model, input_ids: mx.array) -> mx.array:
    """计算序列的 token 级别对数概率

    Args:
        model: 语言模型
        input_ids: [B, L] token IDs

    Returns:
        per_token_logps: [B, L-1] 每个 token 的 log prob
    """
    output = model(input_ids)
    logits = output.logits[:, :-1, :]  # [B, L-1, V]
    labels = input_ids[:, 1:]  # [B, L-1]

    # log_softmax + gather
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    # 手动 gather
    B, L, V = log_probs.shape
    batch_idx = mx.arange(B)[:, None]
    seq_idx = mx.arange(L)[None, :]
    per_token_logps = log_probs[batch_idx, seq_idx, labels]

    return per_token_logps


def dpo_loss(policy_model, ref_model, chosen_ids: mx.array,
             rejected_ids: mx.array, beta: float) -> mx.array:
    """计算 DPO 损失

    L_DPO = -E[log σ(β * (π_logratio - ref_logratio))]
    """
    # 策略模型的对数概率
    policy_chosen_logps = compute_log_probs(policy_model, chosen_ids)
    policy_rejected_logps = compute_log_probs(policy_model, rejected_ids)

    # 参考模型的对数概率（不计算梯度）
    ref_chosen_logps = mx.stop_gradient(compute_log_probs(ref_model, chosen_ids))
    ref_rejected_logps = mx.stop_gradient(compute_log_probs(ref_model, rejected_ids))

    # 创建 mask（忽略 padding）
    chosen_mask = (chosen_ids[:, 1:] != 0).astype(mx.float32)
    rejected_mask = (rejected_ids[:, 1:] != 0).astype(mx.float32)

    # 序列级对数概率（对有效 token 求和）
    policy_chosen_sum = mx.sum(policy_chosen_logps * chosen_mask, axis=-1)
    policy_rejected_sum = mx.sum(policy_rejected_logps * rejected_mask, axis=-1)
    ref_chosen_sum = mx.sum(ref_chosen_logps * chosen_mask, axis=-1)
    ref_rejected_sum = mx.sum(ref_rejected_logps * rejected_mask, axis=-1)

    # DPO logits
    pi_logratios = policy_chosen_sum - policy_rejected_sum
    ref_logratios = ref_chosen_sum - ref_rejected_sum
    logits = pi_logratios - ref_logratios

    # 损失：-log_sigmoid(beta * logits)
    loss = -nn.log_sigmoid(beta * logits)
    return mx.mean(loss)


# =====================================================================================
# 训练循环
# =====================================================================================

def train_epoch(policy_model, ref_model, optimizer, dataset, args, epoch: int,
                start_step: int = 0, wandb=None):
    """DPO 训练一个 epoch"""

    def loss_fn(model, chosen_ids, rejected_ids):
        return dpo_loss(model, ref_model, chosen_ids, rejected_ids, args.beta)

    loss_and_grad_fn = nn.value_and_grad(policy_model, loss_fn)

    data_iter = DPODataIterator(dataset, args.batch_size, shuffle=True)
    iters = len(data_iter)
    epoch_start = time.time()

    accumulated_grads = None
    accum_count = 0

    for step, batch in enumerate(data_iter, start=1):
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        optimizer.learning_rate = lr

        loss, grads = loss_and_grad_fn(
            policy_model, batch['chosen_ids'], batch['rejected_ids']
        )

        # 梯度累积
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = mlx.utils.tree_map(lambda a, g: a + g, accumulated_grads, grads)
        accum_count += 1

        if accum_count == args.accumulation_steps:
            accumulated_grads = mlx.utils.tree_map(lambda g: g / args.accumulation_steps, accumulated_grads)
            accumulated_grads = clip_grad_norm(accumulated_grads, args.grad_clip)
            optimizer.update(policy_model, accumulated_grads)
            mx.eval(policy_model.parameters(), optimizer.state)
            accumulated_grads = None
            accum_count = 0

        # 日志
        if step % args.log_interval == 0 or step == iters:
            elapsed = time.time() - epoch_start
            eta = elapsed / max(step, 1) * (iters - step)
            logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) | "
                f"dpo_loss: {loss.item():.4f} | lr: {lr:.2e} | ETA: {format_duration(eta)}"
            )
            if wandb:
                wandb.log({"dpo_loss": loss.item(), "learning_rate": lr})

        # 保存
        if step % args.save_interval == 0 or step == iters:
            save_checkpoint(policy_model, optimizer, args.save_dir, args.save_weight,
                            epoch=epoch, step=step)

    # 刷新残余
    if accumulated_grads is not None and accum_count > 0:
        accumulated_grads = mlx.utils.tree_map(lambda g: g / accum_count, accumulated_grads)
        accumulated_grads = clip_grad_norm(accumulated_grads, args.grad_clip)
        optimizer.update(policy_model, accumulated_grads)
        mx.eval(policy_model.parameters(), optimizer.state)


# =====================================================================================
# 主程序
# =====================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind MLX DPO")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", type=str, default="dpo_mlx")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=4e-8)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--data_path", type=str, default="../.dataset/dpo.jsonl")
    parser.add_argument("--from_weight", type=str, default="full_sft_mlx")
    parser.add_argument("--beta", type=float, default=0.15, help="DPO 温度参数")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-MLX-DPO")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for attr in ('save_dir', 'data_path'):
        path = getattr(args, attr)
        if not os.path.isabs(path):
            setattr(args, attr, os.path.normpath(os.path.join(script_dir, path)))

    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    # 模型
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        num_key_value_heads=2,
    )

    # 策略模型（可训练）
    policy_model = MiniMindForCausalLM(config)
    mx.eval(policy_model.parameters())

    # 参考模型（冻结）
    ref_model = MiniMindForCausalLM(config)
    mx.eval(ref_model.parameters())

    # 加载权重
    weight_path = os.path.join(args.save_dir, f"{args.from_weight}.safetensors")
    if os.path.exists(weight_path):
        weights = nn.utils.load_safetensors(weight_path)
        policy_model.load_weights(list(weights.items()))
        ref_model.load_weights(list(weights.items()))
        logger(f"Loaded weights: {weight_path}")

    ref_model.freeze()  # 冻结参考模型
    log_model_params(policy_model)

    # Tokenizer + 数据
    tokenizer_path = os.path.normpath(os.path.join(script_dir, '..', 'model'))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    optimizer = optim.AdamW(learning_rate=args.learning_rate)
    wandb = setup_wandb(args, run_name_prefix="MiniMind-MLX-DPO")

    logger(f"\n🚀 DPO Training started (beta={args.beta})")
    for epoch in range(args.epochs):
        setup_seed(42 + epoch)
        train_epoch(policy_model, ref_model, optimizer, train_ds, args, epoch, 0, wandb)

    logger(f"🎉 DPO Training complete!")
