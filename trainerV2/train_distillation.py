# -*- coding: utf-8 -*-
"""
MiniMind MLX 知识蒸馏训练脚本

将大模型（教师）的知识迁移到小模型（学生），通过温度缩放的 KL 散度损失
实现 soft label 蒸馏。

算法：
- 总损失 = α * CE_loss + (1-α) * T² * KL(teacher_soft || student_soft)
- 温度 T 软化概率分布，暴露 dark knowledge

使用方式：
    python trainerV2/train_distillation.py --student_hidden_size 512 --teacher_hidden_size 768
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np

from trainerV2.model_mlx import MiniMindConfig, MiniMindForCausalLM
from trainerV2.trainer_utils import (
    DataIterator, format_duration, get_lr, logger, save_checkpoint,
    setup_seed, setup_wandb, clip_grad_norm, log_model_params,
)
from trainerV2.train_full_sft import SFTDataset


# =====================================================================================
# 蒸馏损失
# =====================================================================================

def distillation_loss(student_logits: mx.array, teacher_logits: mx.array,
                      temperature: float = 1.5) -> mx.array:
    """KL 散度蒸馏损失

    步骤：
    1. 温度缩放：logits / T
    2. 教师 softmax，学生 log_softmax
    3. KL(teacher || student)
    4. 乘以 T² 保持梯度尺度

    Args:
        student_logits: [N, V] 学生模型 logits
        teacher_logits: [N, V] 教师模型 logits
        temperature: 蒸馏温度

    Returns:
        蒸馏损失标量
    """
    # 教师 soft labels
    teacher_probs = mx.softmax(teacher_logits / temperature, axis=-1)
    teacher_probs = mx.stop_gradient(teacher_probs)

    # 学生 log probs
    student_log_probs = student_logits / temperature - mx.logsumexp(
        student_logits / temperature, axis=-1, keepdims=True
    )

    # KL 散度：sum(p * (log p - log q))
    kl = mx.sum(teacher_probs * (mx.log(teacher_probs + 1e-10) - student_log_probs), axis=-1)
    return temperature ** 2 * mx.mean(kl)


# =====================================================================================
# 训练循环
# =====================================================================================

def train_epoch(student_model, teacher_model, optimizer, dataset, args,
                epoch: int, start_step: int = 0, wandb=None):
    """蒸馏训练一个 epoch"""

    def loss_fn(model, input_ids, labels):
        # 学生前向
        student_output = model(input_ids)
        student_logits = student_output.logits[:, :-1, :].reshape(-1, student_output.logits.shape[-1])

        # 教师前向（无梯度）
        teacher_output = teacher_model(input_ids)
        teacher_logits = mx.stop_gradient(
            teacher_output.logits[:, :-1, :].reshape(-1, teacher_output.logits.shape[-1])
        )

        # 截断教师词表维度（若不匹配）
        vocab_size = student_logits.shape[-1]
        teacher_logits = teacher_logits[:, :vocab_size]

        # CE 损失
        shift_labels = labels[:, 1:].reshape(-1)
        valid_mask = (shift_labels != -100).astype(mx.float32)
        safe_labels = mx.where(shift_labels != -100, shift_labels, mx.zeros_like(shift_labels))
        ce = nn.losses.cross_entropy(student_logits, safe_labels, reduction="none")
        ce_loss = mx.sum(ce * valid_mask) / mx.maximum(mx.sum(valid_mask), mx.array(1.0))

        # 蒸馏损失（仅在有效位置）
        valid_indices = mx.where(valid_mask.reshape(-1) > 0)
        if valid_indices.size > 0:
            distill = distillation_loss(
                student_logits, teacher_logits, args.temperature
            )
        else:
            distill = mx.array(0.0)

        # 混合损失
        total = args.alpha * ce_loss + (1 - args.alpha) * distill
        return total

    loss_and_grad_fn = nn.value_and_grad(student_model, loss_fn)

    data_iter = DataIterator(dataset, args.batch_size, shuffle=True, skip_steps=start_step)
    iters = len(data_iter) + start_step
    epoch_start = time.time()

    accumulated_grads = None
    accum_count = 0

    for step, batch in enumerate(data_iter, start=start_step + 1):
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        optimizer.learning_rate = lr

        loss, grads = loss_and_grad_fn(student_model, batch.input_ids, batch.labels)

        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = mlx.utils.tree_map(lambda a, g: a + g, accumulated_grads, grads)
        accum_count += 1

        if accum_count == args.accumulation_steps:
            accumulated_grads = mlx.utils.tree_map(lambda g: g / args.accumulation_steps, accumulated_grads)
            accumulated_grads = clip_grad_norm(accumulated_grads, args.grad_clip)
            optimizer.update(student_model, accumulated_grads)
            mx.eval(student_model.parameters(), optimizer.state)
            accumulated_grads = None
            accum_count = 0

        if step % args.log_interval == 0 or step == iters:
            elapsed = time.time() - epoch_start
            eta = elapsed / max(step - start_step, 1) * (iters - step)
            logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) | "
                f"loss: {loss.item():.4f} | lr: {lr:.2e} | ETA: {format_duration(eta)}"
            )
            if wandb:
                wandb.log({"loss": loss.item(), "learning_rate": lr})

        if step % args.save_interval == 0 or step == iters:
            save_checkpoint(student_model, optimizer, args.save_dir, args.save_weight,
                            epoch=epoch, step=step)

    if accumulated_grads is not None and accum_count > 0:
        accumulated_grads = mlx.utils.tree_map(lambda g: g / accum_count, accumulated_grads)
        accumulated_grads = clip_grad_norm(accumulated_grads, args.grad_clip)
        optimizer.update(student_model, accumulated_grads)
        mx.eval(student_model.parameters(), optimizer.state)


# =====================================================================================
# 主程序
# =====================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind MLX Knowledge Distillation")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", type=str, default="distill_mlx")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--accumulation_steps", type=int, default=16)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=340)
    parser.add_argument("--student_hidden_size", type=int, default=512)
    parser.add_argument("--student_num_layers", type=int, default=8)
    parser.add_argument("--teacher_hidden_size", type=int, default=768)
    parser.add_argument("--teacher_num_layers", type=int, default=8)
    parser.add_argument("--student_use_moe", type=int, default=0)
    parser.add_argument("--teacher_use_moe", type=int, default=0)
    parser.add_argument("--from_student_weight", type=str, default="full_sft_mlx")
    parser.add_argument("--from_teacher_weight", type=str, default="full_sft_mlx")
    parser.add_argument("--alpha", type=float, default=0.5, help="CE 权重")
    parser.add_argument("--temperature", type=float, default=1.5, help="蒸馏温度")
    parser.add_argument("--data_path", type=str, default="../.dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-MLX-Distill")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for attr in ('save_dir', 'data_path'):
        path = getattr(args, attr)
        if not os.path.isabs(path):
            setattr(args, attr, os.path.normpath(os.path.join(script_dir, path)))

    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    # 学生模型
    student_config = MiniMindConfig(
        hidden_size=args.student_hidden_size,
        num_hidden_layers=args.student_num_layers,
        use_moe=bool(args.student_use_moe),
        num_key_value_heads=2,
    )
    student_model = MiniMindForCausalLM(student_config)
    mx.eval(student_model.parameters())

    # 教师模型
    teacher_config = MiniMindConfig(
        hidden_size=args.teacher_hidden_size,
        num_hidden_layers=args.teacher_num_layers,
        use_moe=bool(args.teacher_use_moe),
        num_key_value_heads=2,
    )
    teacher_model = MiniMindForCausalLM(teacher_config)
    mx.eval(teacher_model.parameters())
    teacher_model.freeze()  # 冻结教师模型

    logger(f"Student params: {get_model_params(student_model)[0]:.2f}M")
    logger(f"Teacher params: {get_model_params(teacher_model)[0]:.2f}M")

    # Tokenizer + 数据
    tokenizer_path = os.path.normpath(os.path.join(script_dir, '..', 'model'))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    optimizer = optim.AdamW(learning_rate=args.learning_rate)
    wandb = setup_wandb(args, run_name_prefix="MiniMind-MLX-Distill")

    logger(f"\n🚀 Distillation started (alpha={args.alpha}, T={args.temperature})")
    for epoch in range(args.epochs):
        setup_seed(42 + epoch)
        train_epoch(student_model, teacher_model, optimizer, train_ds, args, epoch, 0, wandb)

    logger(f"🎉 Distillation complete!")


def get_model_params(model):
    """辅助函数：获取参数量"""
    total = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    return total / 1e6, 0
