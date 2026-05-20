# -*- coding: utf-8 -*-
"""
MiniMind 全量监督微调（Full SFT）训练脚本

本脚本实现 MiniMind 模型的全量参数监督微调（Supervised Fine-Tuning）流程。
与预训练不同，SFT 阶段使用指令跟随数据（instruction-response 对）进行训练，
使模型学会遵循人类指令、完成特定任务。

核心特性：
- 分布式训练（DDP）：支持多卡并行训练，自动处理进程组初始化和同步
- 混合精度训练：支持 bfloat16（推荐）和 float16，通过 GradScaler 加速训练
- 梯度累积：通过 accumulation_steps 模拟更大的 batch size，节省显存
- 梯度裁剪：防止梯度爆炸，保证训练稳定性
- 断点续训：自动检测并恢复之前的训练状态（epoch、step、optimizer、scaler）
- WandB 日志记录：实时上传训练指标到云端，支持可视化监控
- MPS/CUDA 设备优化：针对 Apple Silicon 和 NVIDIA GPU 进行设备适配

训练算法：
- 因果语言建模（Causal LM）：预测下一个 token，使用交叉熵损失
- 学习率调度：线性衰减策略，从初始学习率逐步降低到接近 0
- 梯度累积 + 裁剪：每 accumulation_steps 步执行一次参数更新，更新前裁剪梯度范数

使用方式：
    # 单卡训练
    python trainer/train_full_sft.py --epochs 2 --batch_size 8 --learning_rate 1e-5
    
    # 多卡分布式训练
    torchrun --nproc_per_node=4 trainer/train_full_sft.py --epochs 2 --batch_size 8
    
    # 启用 WandB 日志
    python trainer/train_full_sft.py --use_wandb --wandb_project My-SFT-Project
    
    # 断点续训
    python trainer/train_full_sft.py --from_resume 1 --save_weight full_sft

注意事项：
- 数据格式：使用 JSONL 格式的 SFT 数据集，每条样本包含 instruction 和 response
- 序列长度：max_seq_len 建议设置为 512~1024，过长会增加显存占用
- 学习率：SFT 阶段学习率通常比预训练小一个数量级（1e-5 ~ 5e-5）
- 混合精度：推荐使用 bfloat16（数值稳定性更好），float16 需配合 GradScaler
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings

import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler

from dataset.lm_dataset import SFTDataset
from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import (
    Logger, SkipBatchSampler, build_train_dataloader, get_default_device, get_device_type, get_lr,
    init_distributed_mode, init_model, is_main_process, lm_checkpoint,
    restore_training_state, save_checkpoint, setup_precision_context, setup_seed,
    setup_wandb, wrap_model_for_training,
)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    执行一个 epoch 的全量监督微调训练
    
    本函数实现完整的 SFT 训练循环，包含前向传播、反向传播、梯度累积、优化器更新等核心步骤。
    支持断点续训（通过 start_step 跳过已完成的 step），并定期记录日志和保存模型检查点。
    
    训练流程：
    1. 数据加载：从 loader 获取 input_ids 和 labels
    2. 学习率调度：根据全局 step 计算当前学习率（线性衰减）
    3. 前向传播：在 autocast 上下文中计算 loss（logits loss + aux loss）
    4. 梯度累积：将 loss 除以 accumulation_steps，累加梯度
    5. 优化器更新：每 accumulation_steps 步执行一次梯度裁剪和参数更新
    6. 日志记录：定期计算平均 loss、tokens/sec、ETA 等指标
    7. 模型保存：定期保存模型权重和训练状态
    
    Args:
        epoch: 当前 epoch 索引（从 0 开始）
        loader: DataLoader 实例，提供 (input_ids, labels) 批次
        iters: 本 epoch 的总步数（用于进度计算）
        start_step: 起始 step，用于断点续训时跳过已完成的步骤
        wandb: WandB 日志记录器，用于上传训练指标（可选）
        
    Returns:
        None。训练状态通过全局变量 optimizer、scaler、model 等隐式更新。
        
    Note:
        - 使用 loss.item() * accumulation_steps 还原真实 loss 值用于日志展示
        - epoch 结束时若最后一步未完成梯度累积，会强制执行一次优化器更新
        - 每次迭代后删除中间变量（input_ids, labels, res, loss）以释放显存
    """
    # ===== 阶段 1：初始化训练状态 =====
    start_time = time.time()
    last_step = start_step
    
    # ===== 阶段 2：训练主循环 =====
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # --- 步骤 2.1：数据准备 ---
        # 将输入数据移动到指定设备（CUDA/MPS）
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        
        # --- 步骤 2.2：学习率调度 ---
        # 根据全局 step 计算当前学习率（线性衰减策略）
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # --- 步骤 2.3：前向传播（混合精度） ---
        # 在 autocast 上下文中执行前向传播，自动使用低精度加速计算
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            # 总损失 = logits 交叉熵损失 + 辅助损失（如 MoE 的负载均衡损失）
            loss = res.loss + res.aux_loss
            # 梯度累积：将损失除以累积步数，使梯度的量级与单步更新一致
            loss = loss / args.accumulation_steps

        # --- 步骤 2.4：反向传播 ---
        # 使用 GradScaler 缩放 loss 后进行反向传播，防止 float16 下梯度下溢
        scaler.scale(loss).backward()

        # --- 步骤 2.5：优化器更新（梯度累积完成时） ---
        if step % args.accumulation_steps == 0:
            # 取消梯度缩放，恢复原始梯度值
            scaler.unscale_(optimizer)
            # 梯度裁剪：限制梯度范数不超过 grad_clip，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 执行参数更新
            scaler.step(optimizer)
            scaler.update()

            # 清空梯度，set_to_none=True 比 zero_() 更高效（减少内存分配）
            optimizer.zero_grad(set_to_none=True)

        # --- 步骤 2.6：日志记录 ---
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # 还原真实 loss 值（乘以 accumulation_steps 抵消之前的除法）
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            # 估算剩余训练时间（分钟）
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss,
                                 "learning_rate": current_lr, "epoch_time": eta_min})

        # --- 步骤 2.7：模型保存 ---
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            # 统一保存：推理权重 + 训练状态（自动处理 DDP / torch.compile 包装）
            save_checkpoint(model, lm_config, args.save_dir, args.save_weight,
                            optimizer=optimizer, scaler=scaler,
                            epoch=epoch, step=step, wandb=wandb)
            model.train()

        # 释放显存：删除中间变量引用，触发 Python GC 回收 GPU 内存
        del input_ids, labels, res, loss

    # ===== 阶段 3：epoch 结束时的清理 =====
    # 如果最后一步未完成梯度累积，强制执行一次优化器更新，确保所有梯度都被应用
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # ===== 主程序入口：SFT 训练流程编排 =====
    
    # --- 步骤 1：参数解析 ---
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default=get_default_device(), help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=768, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../.dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str,
                        help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1],
                        help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ===== 阶段 1：初始化分布式环境和随机种子 =====
    # 初始化 DDP 进程组，获取当前进程的 local_rank
    local_rank = init_distributed_mode()
    # 如果是分布式训练，将设备设置为对应的 CUDA 卡
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    # 设置随机种子，确保不同进程的数据打乱顺序不同，但可复现
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    Logger(f'Training device: {args.device}')

    # ===== 阶段 2：配置模型参数和检查点 =====
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    # 初始化模型配置对象
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               use_moe=bool(args.use_moe), num_key_value_heads=2)
    # 如果启用断点续训，尝试加载之前的检查点数据
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

    # ===== 阶段 3：设置混合精度训练上下文 =====
    # 检测设备类型（cuda/mps/cpu），自动选择合适的 autocast 和 GradScaler 配置
    device_type = get_device_type(args.device)
    autocast_ctx, scaler, use_scaler = setup_precision_context(device_type, args.dtype, lm_config)

    # ===== 阶段 4：初始化 WandB 日志记录器 =====
    # 如果启用了 WandB，创建或恢复运行实例，支持断点续训时继续记录
    wandb = setup_wandb(args, ckp_data, run_name_prefix="MiniMind-Full-SFT")

    # ===== 阶段 5：初始化模型、数据集和优化器 =====
    # 加载预训练权重（如果有）并初始化模型和 tokenizer
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 加载 SFT 数据集
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 如果是分布式训练，创建分布式采样器，确保每个进程看到不同的数据子集
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 重新配置 GradScaler（仅在 CUDA + float16 时启用）
    use_scaler = (device_type == "cuda" and args.dtype == "float16")
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_scaler) if device_type == "cuda" else torch.amp.GradScaler(
        enabled=False)
    # 创建 AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ===== 阶段 6：从检查点恢复训练状态 =====
    # 如果存在检查点数据，恢复 epoch、step、optimizer 状态和 scaler 状态
    start_epoch, start_step = restore_training_state(ckp_data, model, optimizer=optimizer, scaler=scaler)

    # ===== 阶段 7：模型编译和分布式包装 =====
    # 如果启用了 torch.compile，对模型进行编译加速；如果是分布式训练，包装为 DDP 模型
    model = wrap_model_for_training(model, use_compile=bool(args.use_compile), local_rank=local_rank)

    # ===== 阶段 8：执行训练循环 =====
    for epoch in range(start_epoch, args.epochs):
        # 设置分布式采样器的 epoch，确保每个 epoch 数据打乱顺序不同
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        # 为每个 epoch 设置不同的随机种子，保证数据打乱的可复现性
        setup_seed(42 + epoch)
        # 生成随机索引，用于构建批次采样器
        indices = torch.randperm(len(train_ds)).tolist()
        # 计算需要跳过的步数（仅在第一个 epoch 且是断点续训时）
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 创建支持跳过步数的批次采样器
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # 创建 DataLoader，启用 pin_memory 加速 CUDA 数据传输
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers,
                            pin_memory=(device_type == "cuda"))
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # 传入 skip 后的总步数，确保进度计算正确
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ===== 阶段 9：清理分布式资源 =====
    # 如果是分布式训练，销毁进程组，释放资源
    if dist.is_initialized():
        dist.destroy_process_group()
