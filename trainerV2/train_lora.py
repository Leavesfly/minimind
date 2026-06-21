# -*- coding: utf-8 -*-
"""
MiniMind MLX LoRA 微调训练脚本

基于 MLX 框架实现的 LoRA（Low-Rank Adaptation）参数高效微调。
MLX 原生支持 LoRA，通过 nn.Linear 的 freeze/unfreeze 机制实现。

核心优势：
- MLX 原生 LoRA 支持：nn.utils.lora 一键注入
- 参数冻结简洁：model.freeze() + 选择性 unfreeze
- 极少的可训练参数（<1% 原模型）

使用方式：
    python trainerV2/train_lora.py --data_path ../.dataset/lora_medical.jsonl --lora_name lora_medical
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
    DataIterator, format_duration, get_lr, logger, save_checkpoint,
    load_checkpoint, setup_seed, setup_wandb, clip_grad_norm, log_model_params,
)
from trainerV2.train_full_sft import SFTDataset


# =====================================================================================
# LoRA 工具
# =====================================================================================

def apply_lora(model: nn.Module, rank: int = 8, target_modules: list = None):
    """为模型注入 LoRA 适配器

    在指定的线性层旁路插入低秩分解矩阵 A 和 B：
    output = Wx + BAx，其中 B∈R^(d×r), A∈R^(r×k), r << min(d, k)

    Args:
        model: MiniMind 模型
        rank: LoRA 秩（低秩维度），典型值 4/8/16
        target_modules: 要注入 LoRA 的模块名（默认 attention 投影层）
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    # 遍历所有层，为目标模块注入 LoRA
    for layer in model.layers:
        attn = layer.attention
        for name in target_modules:
            if hasattr(attn, name):
                original = getattr(attn, name)
                # 创建 LoRA 层
                lora_layer = LoRALinear(original, rank=rank)
                setattr(attn, name, lora_layer)

    logger(f"LoRA injected: rank={rank}, targets={target_modules}")


class LoRALinear(nn.Module):
    """LoRA 线性层

    在原始线性层旁路添加低秩分解：
    y = W @ x + (B @ A) @ x * (alpha / rank)
    """

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        in_features = original.weight.shape[1]
        out_features = original.weight.shape[0]

        self.original = original
        self.rank = rank
        self.scale = alpha / rank

        # LoRA 矩阵：A 使用随机初始化，B 初始化为零
        self.lora_a = mx.random.normal((in_features, rank)) * 0.01
        self.lora_b = mx.zeros((rank, out_features))

    def __call__(self, x: mx.array) -> mx.array:
        # 原始输出 + LoRA 旁路
        base_out = self.original(x)
        lora_out = (x @ self.lora_a) @ self.lora_b * self.scale
        return base_out + lora_out


def freeze_non_lora(model: nn.Module):
    """冻结非 LoRA 参数

    遍历模型参数，仅保留 LoRA 相关参数（lora_a, lora_b）为可训练状态。
    """
    model.freeze()
    # 解冻 LoRA 参数
    for layer in model.layers:
        attn = layer.attention
        for name in ["q_proj", "v_proj", "k_proj", "o_proj"]:
            module = getattr(attn, name, None)
            if isinstance(module, LoRALinear):
                module.unfreeze()


def save_lora_weights(model: nn.Module, save_path: str):
    """仅保存 LoRA 权重"""
    lora_weights = {}
    for i, layer in enumerate(model.layers):
        attn = layer.attention
        for name in ["q_proj", "v_proj", "k_proj", "o_proj"]:
            module = getattr(attn, name, None)
            if isinstance(module, LoRALinear):
                lora_weights[f"layers.{i}.attention.{name}.lora_a"] = module.lora_a
                lora_weights[f"layers.{i}.attention.{name}.lora_b"] = module.lora_b

    nn.utils.save_safetensors(save_path, lora_weights)
    logger(f"LoRA weights saved: {save_path}")


# =====================================================================================
# 训练循环
# =====================================================================================

def train_epoch(model, optimizer, dataset, args, epoch: int,
                start_step: int = 0, wandb=None):
    """LoRA 微调一个 epoch"""

    def loss_fn(model, input_ids, labels):
        output = model(input_ids, labels=labels)
        return output.loss + output.aux_loss

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    data_iter = DataIterator(dataset, args.batch_size, shuffle=True, skip_steps=start_step)
    iters = len(data_iter) + start_step

    epoch_start = time.time()
    accumulated_grads = None
    accum_count = 0

    for step, batch in enumerate(data_iter, start=start_step + 1):
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        optimizer.learning_rate = lr

        loss, grads = loss_and_grad_fn(model, batch.input_ids, batch.labels)

        # 梯度累积
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = mlx.utils.tree_map(lambda a, g: a + g, accumulated_grads, grads)
        accum_count += 1

        if accum_count == args.accumulation_steps:
            accumulated_grads = mlx.utils.tree_map(lambda g: g / args.accumulation_steps, accumulated_grads)
            accumulated_grads = clip_grad_norm(accumulated_grads, args.grad_clip)
            optimizer.update(model, accumulated_grads)
            mx.eval(model.parameters(), optimizer.state)
            accumulated_grads = None
            accum_count = 0

        # 日志
        if step % args.log_interval == 0 or step == iters:
            elapsed = time.time() - epoch_start
            eta = elapsed / max(step - start_step, 1) * (iters - step)
            logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) | "
                f"loss: {loss.item():.4f} | lr: {lr:.2e} | ETA: {format_duration(eta)}"
            )
            if wandb:
                wandb.log({"loss": loss.item(), "learning_rate": lr})

        # 保存 LoRA 权重
        if step % args.save_interval == 0 or step == iters:
            lora_path = os.path.join(args.save_dir, f"{args.lora_name}_mlx.safetensors")
            save_lora_weights(model, lora_path)

    # 刷新残余梯度
    if accumulated_grads is not None and accum_count > 0:
        accumulated_grads = mlx.utils.tree_map(lambda g: g / accum_count, accumulated_grads)
        accumulated_grads = clip_grad_norm(accumulated_grads, args.grad_clip)
        optimizer.update(model, accumulated_grads)
        mx.eval(model.parameters(), optimizer.state)


# =====================================================================================
# 主程序
# =====================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind MLX LoRA Fine-tuning")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--lora_name", type=str, default="lora_medical")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=340)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA 秩")
    parser.add_argument("--data_path", type=str, default="../.dataset/lora_medical.jsonl")
    parser.add_argument("--from_weight", type=str, default="full_sft_mlx")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-MLX-LoRA")
    args = parser.parse_args()

    # 路径处理
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
    model = MiniMindForCausalLM(config)
    mx.eval(model.parameters())

    # 加载基础权重
    weight_path = os.path.join(args.save_dir, f"{args.from_weight}.safetensors")
    if os.path.exists(weight_path):
        weights = nn.utils.load_safetensors(weight_path)
        model.load_weights(list(weights.items()))
        logger(f"Loaded base weights: {weight_path}")

    # 注入 LoRA 并冻结基础参数
    apply_lora(model, rank=args.lora_rank)
    freeze_non_lora(model)

    # 统计参数
    total_m, trainable_m = log_model_params(model)
    logger(f"LoRA trainable ratio: {trainable_m / total_m * 100:.2f}%" if total_m > 0 else "")

    # Tokenizer + 数据
    tokenizer_path = os.path.normpath(os.path.join(script_dir, '..', 'model'))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    # 优化器（仅优化可训练参数）
    optimizer = optim.AdamW(learning_rate=args.learning_rate)

    wandb = setup_wandb(args, run_name_prefix=f"MiniMind-MLX-LoRA-{args.lora_name}")

    # 训练
    logger(f"\n🚀 LoRA Training started (rank={args.lora_rank})")
    for epoch in range(args.epochs):
        setup_seed(42 + epoch)
        train_epoch(model, optimizer, train_ds, args, epoch, 0, wandb)

    logger(f"🎉 LoRA Training complete!")
