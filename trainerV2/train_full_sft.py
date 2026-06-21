# -*- coding: utf-8 -*-
"""
MiniMind MLX 全量监督微调（Full SFT）训练脚本

基于 MLX 框架实现的 SFT 训练，使用指令跟随数据微调模型。
与预训练的区别在于使用 instruction-response 格式的数据，
且学习率通常更小（1e-5 ~ 5e-5）。

使用方式：
    python trainerV2/train_full_sft.py --epochs 2 --batch_size 8 --learning_rate 1e-5
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
    DataIterator, format_duration, get_lr, get_memory_usage,
    log_model_params, logger, make_progress_bar, save_checkpoint,
    load_checkpoint, setup_seed, setup_wandb, clip_grad_norm,
)


# =====================================================================================
# SFT 数据集
# =====================================================================================

class SFTDataset:
    """SFT 微调数据集

    加载 JSONL 格式的指令跟随数据，使用 chat template 编码为 token 序列。
    支持 conversation 格式 [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 768):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.samples = []

        logger(f"Loading SFT data from: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                conversations = data.get('conversations', data.get('messages', []))
                if conversations:
                    self.samples.append(conversations)
        logger(f"Loaded {len(self.samples):,} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        conversations = self.samples[idx]

        # 使用 tokenizer 的 chat template 编码
        text = self.tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=False
        )
        encoded = self.tokenizer.encode(text, add_special_tokens=False)

        # 截断
        encoded = encoded[:self.max_length + 1]

        # Padding
        if len(encoded) < self.max_length + 1:
            pad_len = self.max_length + 1 - len(encoded)
            encoded = encoded + [self.tokenizer.pad_token_id or 0] * pad_len

        encoded = np.array(encoded, dtype=np.int32)
        input_ids = encoded[:-1]
        labels = encoded[1:]

        return input_ids, labels


# =====================================================================================
# 训练循环
# =====================================================================================

def train_epoch(model, optimizer, dataset, args, epoch: int,
                start_step: int = 0, wandb=None):
    """执行一个 epoch 的 SFT 训练"""

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
        # 学习率调度
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        optimizer.learning_rate = lr

        # 前向 + 梯度
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
                f"loss: {loss.item():.4f} | lr: {lr:.2e} | "
                f"ETA: {format_duration(eta)} | {get_memory_usage()}"
            )
            if wandb:
                wandb.log({"loss": loss.item(), "learning_rate": lr})

        # 保存
        if step % args.save_interval == 0 or step == iters:
            save_checkpoint(model, optimizer, args.save_dir, args.save_weight,
                            epoch=epoch, step=step)

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
    parser = argparse.ArgumentParser(description="MiniMind MLX Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", type=str, default="full_sft_mlx")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--data_path", type=str, default="../.dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--from_weight", type=str, default="pretrain_mlx", help="加载预训练权重")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-MLX-SFT")
    args = parser.parse_args()

    # 路径处理
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for attr in ('save_dir', 'data_path'):
        path = getattr(args, attr)
        if not os.path.isabs(path):
            setattr(args, attr, os.path.normpath(os.path.join(script_dir, path)))

    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    # 模型初始化
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        num_key_value_heads=2,
    )
    model = MiniMindForCausalLM(config)
    mx.eval(model.parameters())

    # 加载预训练权重
    if args.from_weight != "none":
        weight_path = os.path.join(args.save_dir, f"{args.from_weight}.safetensors")
        if os.path.exists(weight_path):
            weights = nn.utils.load_safetensors(weight_path)
            model.load_weights(list(weights.items()))
            logger(f"Loaded pretrain weights: {weight_path}")

    log_model_params(model)

    # Tokenizer
    tokenizer_path = os.path.normpath(os.path.join(script_dir, '..', 'model'))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 数据
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    # 优化器
    optimizer = optim.AdamW(learning_rate=args.learning_rate)

    # 续训
    start_epoch, start_step = 0, 0
    if args.from_resume:
        start_epoch, start_step = load_checkpoint(model, args.save_dir, args.save_weight)

    wandb = setup_wandb(args, run_name_prefix="MiniMind-MLX-SFT")

    # 训练
    logger(f"\n🚀 SFT Training started")
    for epoch in range(start_epoch, args.epochs):
        setup_seed(42 + epoch)
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        train_epoch(model, optimizer, train_ds, args, epoch, skip, wandb)

    logger(f"🎉 SFT Training complete!")
