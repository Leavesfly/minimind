# -*- coding: utf-8 -*-
"""
MiniMind MLX 预训练脚本

基于 Apple MLX 框架实现的因果语言建模预训练，充分利用 Apple Silicon
统一内存架构实现高效训练。

核心特性：
- MLX 惰性求值：计算图自动融合优化
- 函数式梯度计算：nn.value_and_grad 替代命令式 backward
- 统一内存：零拷贝数据访问，无需 pin_memory
- 原生低精度：直接以 float16/bfloat16 计算，无需 GradScaler

使用方式：
    python trainerV2/train_pretrain.py --epochs 2 --batch_size 8 --learning_rate 5e-4

算法说明：
- 优化器：AdamW，余弦退火学习率调度
- 损失：CrossEntropyLoss + MoE Auxiliary Loss
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
    DataIterator, format_duration, get_lr, get_memory_usage,
    log_model_params, logger, make_progress_bar, save_checkpoint,
    load_checkpoint, setup_seed, setup_wandb, clip_grad_norm,
)


# =====================================================================================
# 数据集
# =====================================================================================

class PretrainDataset:
    """预训练数据集（MLX 版本）

    从 JSONL 文件加载预训练数据，每行是一个 JSON 对象，
    包含 'text' 字段。使用 tokenizer 编码后截断到 max_length。
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        import json
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.samples = []

        logger(f"Loading pretrain data from: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                text = data.get('text', data.get('content', ''))
                if text:
                    self.samples.append(text)
        logger(f"Loaded {len(self.samples):,} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encoded = self.tokenizer.encode(text, add_special_tokens=True)

        # 截断到 max_length + 1（需要 shift 一位做 label）
        encoded = encoded[:self.max_length + 1]

        # Padding
        if len(encoded) < self.max_length + 1:
            encoded = encoded + [self.tokenizer.pad_token_id or 0] * (self.max_length + 1 - len(encoded))

        encoded = np.array(encoded, dtype=np.int32)
        input_ids = encoded[:-1]
        labels = encoded[1:]

        return input_ids, labels


# =====================================================================================
# 训练核心
# =====================================================================================

def create_loss_fn(model: MiniMindForCausalLM):
    """创建损失函数闭包，用于 MLX 的 value_and_grad

    MLX 的梯度计算采用函数式风格：
    loss_fn(model, input_ids, labels) -> scalar loss
    通过 nn.value_and_grad(model, loss_fn) 同时获得损失值和梯度。
    """

    def loss_fn(model, input_ids, labels):
        output = model(input_ids, labels=labels)
        total_loss = output.loss + output.aux_loss
        return total_loss

    return loss_fn


def train_epoch(model, optimizer, dataset, args, epoch: int,
                start_step: int = 0, wandb=None):
    """执行一个 epoch 的预训练

    MLX 训练范式：
    1. 构造 loss_fn(model, batch) -> loss
    2. 使用 nn.value_and_grad 获取 (loss, grads)
    3. 梯度裁剪
    4. optimizer.update(model, grads)
    5. mx.eval() 触发惰性求值

    Args:
        model: MiniMind MLX 模型
        optimizer: MLX 优化器
        dataset: 预训练数据集
        args: 命令行参数
        epoch: 当前 epoch
        start_step: 起始 step（续训）
        wandb: 日志记录器
    """
    loss_fn = create_loss_fn(model)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # 创建数据迭代器
    data_iter = DataIterator(dataset, args.batch_size, shuffle=True, skip_steps=start_step)
    iters = len(data_iter) + start_step

    # 训练状态
    epoch_start = time.time()
    running_loss = mx.array(0.0)  # 使用 mx.array 避免每步 .item()
    log_step_count = 0
    log_start_time = time.time()
    tokens_per_batch = args.batch_size * args.max_seq_len

    # 梯度累积缓冲
    accumulated_grads = None
    accum_count = 0

    for step, batch in enumerate(data_iter, start=start_step + 1):
        # 学习率调度
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        optimizer.learning_rate = lr

        # 前向传播 + 梯度计算（MLX 函数式风格）
        loss, grads = loss_and_grad_fn(model, batch.input_ids, batch.labels)

        # 梯度累积
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = mlx.utils.tree_map(lambda a, g: a + g, accumulated_grads, grads)
        accum_count += 1

        # 关键：每步强制求值，防止计算图在累积期间无限膨胀导致 OOM
        mx.eval(loss, *mlx.utils.tree_flatten(accumulated_grads))

        # 达到累积步数时执行更新
        if accum_count == args.accumulation_steps:
            # 平均梯度
            accumulated_grads = mlx.utils.tree_map(
                lambda g: g / args.accumulation_steps, accumulated_grads
            )
            # 梯度裁剪
            accumulated_grads = clip_grad_norm(accumulated_grads, args.grad_clip)
            # 参数更新
            optimizer.update(model, accumulated_grads)
            # 触发惰性求值
            mx.eval(model.parameters(), optimizer.state)
            # 清理缓存释放显存
            mx.clear_cache()
            # 重置累积
            accumulated_grads = None
            accum_count = 0

        # 累积 loss（保持为 mx.array，避免 .item() 同步）
        running_loss = running_loss + loss
        log_step_count += 1

        # 日志记录
        is_log_step = (step % args.log_interval == 0 or step == iters or step == start_step + 1)
        if is_log_step:
            # 只在日志时才调用 .item() 触发求值
            avg_loss = running_loss.item() / max(log_step_count, 1)
            cur_loss = loss.item()

            now = time.time()
            elapsed = now - log_start_time
            steps_done = step - start_step

            tokens_per_sec = log_step_count * tokens_per_batch / max(elapsed, 1e-6)
            avg_step_time = (now - epoch_start) / max(steps_done, 1)
            eta_seconds = avg_step_time * (iters - step)

            global_step = epoch * iters + step
            global_total = args.epochs * iters
            progress = make_progress_bar(global_step, global_total)
            mem = get_memory_usage()

            logger(
                f"{progress} Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) | "
                f"loss: {cur_loss:.4f} (avg: {avg_loss:.4f}) | "
                f"lr: {lr:.2e} | {tokens_per_sec:.0f} tok/s | "
                f"{avg_step_time * 1000:.0f}ms/step | ETA: {format_duration(eta_seconds)} | {mem}"
            )

            if wandb:
                wandb.log({
                    "loss": loss.item(),
                    "avg_loss": avg_loss,
                    "learning_rate": lr,
                    "tokens_per_sec": tokens_per_sec,
                })

            # 重置日志计数
            running_loss = mx.array(0.0)
            log_step_count = 0
            log_start_time = now

        # 模型保存
        if step % args.save_interval == 0 or step == iters:
            save_checkpoint(model, optimizer, args.save_dir, args.save_weight,
                            epoch=epoch, step=step, config=None)

    # 处理未完成的梯度累积
    if accumulated_grads is not None and accum_count > 0:
        accumulated_grads = mlx.utils.tree_map(lambda g: g / accum_count, accumulated_grads)
        accumulated_grads = clip_grad_norm(accumulated_grads, args.grad_clip)
        optimizer.update(model, accumulated_grads)
        mx.eval(model.parameters(), optimizer.state)

    # Epoch 汇总
    elapsed = time.time() - epoch_start
    total_tokens = (iters - start_step) * tokens_per_batch
    logger(
        f"✅ Epoch {epoch + 1}/{args.epochs} complete | "
        f"steps: {iters - start_step} | time: {format_duration(elapsed)} | "
        f"avg: {total_tokens / max(elapsed, 1):.0f} tok/s"
    )


# =====================================================================================
# 主程序
# =====================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind MLX Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--save_weight", type=str, default="pretrain_mlx", help="保存权重前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--accumulation_steps", type=int, default=32, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=50, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--hidden_size", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="隐藏层数量")
    parser.add_argument("--max_seq_len", type=int, default=340, help="最大序列长度")
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1], help="是否使用MoE")
    parser.add_argument("--data_path", type=str, default="../.dataset/pretrain_t2t_mini.jsonl", help="数据路径")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1], help="是否续训")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-MLX-Pretrain", help="wandb项目名")
    args = parser.parse_args()

    # 路径处理
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for attr in ('save_dir', 'data_path'):
        path = getattr(args, attr)
        if not os.path.isabs(path):
            setattr(args, attr, os.path.normpath(os.path.join(script_dir, path)))

    # 初始化
    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    # 打印训练配置
    logger("=" * 60)
    logger(f"  MiniMind MLX Pretraining (Apple Silicon Native)")
    logger(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | Accum: {args.accumulation_steps}")
    logger(f"  Effective batch: {args.batch_size * args.accumulation_steps}")
    logger(f"  LR: {args.learning_rate} | Grad clip: {args.grad_clip}")
    logger(f"  Max seq len: {args.max_seq_len}")
    logger(f"  Model: hidden={args.hidden_size}, layers={args.num_hidden_layers}, MoE={bool(args.use_moe)}")
    logger("=" * 60)

    # 初始化模型
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        num_key_value_heads=2,
    )
    model = MiniMindForCausalLM(config)
    mx.eval(model.parameters())  # 初始化参数
    log_model_params(model)

    # 加载 tokenizer
    tokenizer_path = os.path.normpath(os.path.join(script_dir, '..', 'model'))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 数据集
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    # 优化器
    optimizer = optim.AdamW(learning_rate=args.learning_rate)

    # 续训
    start_epoch, start_step = 0, 0
    if args.from_resume:
        start_epoch, start_step = load_checkpoint(model, args.save_dir, args.save_weight)

    # WandB
    wandb = setup_wandb(args, run_name_prefix="MiniMind-MLX-Pretrain")

    # 开始训练
    logger(f"\n🚀 Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    training_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        setup_seed(42 + epoch)
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        train_epoch(model, optimizer, train_ds, args, epoch, skip, wandb)

    logger(f"\n🎉 Training complete! Total time: {format_duration(time.time() - training_start)}")
