# -*- coding: utf-8 -*-
"""
MiniMind 知识蒸馏训练脚本

本脚本实现了知识蒸馏训练，将大模型（教师）的知识迁移到小模型（学生）。
主要特点：
1. 温度缩放：通过温度参数软化教师模型的概率分布
2. 混合损失：结合 CE 损失（监督学习）和 KL 散度损失（蒸馏）
3. 灵活配置：支持不同大小的学生和教师模型
4. MoE 支持：支持 MoE 模型蒸馏到 Dense 模型
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, \
    init_model, SkipBatchSampler, get_default_device, get_device_type

warnings.filterwarnings('ignore')


def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    """
    计算 KL 散度蒸馏损失
    
    蒸馏损失的计算步骤：
    1. 温度缩放：将教师和学生的 logits 除以温度 T，软化概率分布
    2. 概率计算：教师使用 softmax，学生使用 log_softmax
    3. KL 散度：计算学生分布与教师分布之间的 KL 散度
    4. 温度平方缩放：将 KL 散度乘以 T²，保持梯度尺度
    
    温度的作用：
    - T > 1：软化概率分布，暴露更多暗知识
    - T = 1：标准 softmax，无软化效果
    - T < 1：锐化概率分布，强调高概率类别
    
    参数:
        student_logits: 学生模型的 logits，形状 [B, L, V]
        teacher_logits: 教师模型的 logits，形状 [B, L, V]
        temperature: 蒸馏温度，推荐范围 1.0-2.0
        reduction: 损失归约方式
    
    返回:
        torch.Tensor: 蒸馏损失，已乘以 T²
    """
    with torch.no_grad():
        # 教师模型：softmax 并温度缩放
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # 学生模型：log_softmax 并温度缩放
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # 计算 KL 散度
    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )

    # 温度平方缩放，保持梯度尺度
    return (temperature ** 2) * kl


def train_epoch(epoch, loader, iters, teacher_model, lm_config_student, start_step=0, wandb=None, alpha=0.0,
                temperature=1.0):
    """
    蒸馏训练一个 epoch
    
    蒸馏训练流程：
    1. 学生模型前向：计算学生模型的 logits
    2. 教师模型前向：计算教师模型的 logits（无梯度）
    3. CE 损失计算：使用真实标签计算交叉熵损失
    4. 蒸馏损失计算：使用教师 logits 计算 KL 散度损失
    5. 混合损失：总损失 = alpha * CE + (1-alpha) * KL
    
    参数:
        epoch (int): 当前训练轮数
        loader (DataLoader): 数据加载器
        iters (int): 总迭代次数
        teacher_model: 教师模型（冻结参数）
        lm_config_student: 学生模型配置
        start_step (int): 起始 step
        wandb: wandb 日志记录器
        alpha (float): CE 损失权重，总损失 = alpha*CE + (1-alpha)*KL
        temperature (float): 蒸馏温度
    """
    start_time = time.time()
    last_step = start_step

    # 教师模型设置为评估模式并冻结参数
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        last_step = step
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        loss_mask = (labels[..., 1:] != -100).float()
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 1. 学生模型前向传播
        with autocast_ctx:
            res = model(input_ids)
            student_logits = res.logits[..., :-1, :].contiguous()

        # 2. 教师模型前向传播（只在 eval & no_grad）
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
                # 如果词表大小不同，截断教师 logits
                vocab_size_student = student_logits.size(-1)
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # ========== 计算损失 ==========
        # 3) Ground-Truth CE Loss
        shift_labels = labels[..., 1:].contiguous()
        loss_mask_flat = loss_mask.view(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='none'
        )
        ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
        if lm_config_student.use_moe:
            ce_loss = ce_loss_raw + res.aux_loss
        else:
            ce_loss = ce_loss_raw

        # 4) Distillation Loss
        if teacher_model is not None:
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # 5) 总损失 = alpha * CE + (1-alpha) * Distill
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_ce_loss = ce_loss_raw.item()
            current_aux_loss = res.aux_loss.item() if lm_config_student.use_moe else 0.0
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60

            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, ce: {current_ce_loss:.4f}, aux_loss: {current_aux_loss:.4f}, distill: {distill_loss.item():.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')

            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "ce_loss": current_ce_loss,
                    "aux_loss": current_aux_loss,
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config_student.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config_student, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del input_ids, labels, loss_mask, res, student_logits, ce_loss, distill_loss, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # 模拟用moe模型蒸馏dense模型，也可以用更大teacher_hidden_size模型蒸馏更小student_hidden_size的
    parser = argparse.ArgumentParser(description="MiniMind Knowledge Distillation")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_dist', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default=get_default_device(), help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--max_seq_len", type=int, default=340, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--data_path", type=str, default="../.dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument('--student_hidden_size', default=768, type=int, help="学生模型隐藏层维度")
    parser.add_argument('--student_num_layers', default=8, type=int, help="学生模型隐藏层数量")
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help="教师模型隐藏层维度")
    parser.add_argument('--teacher_num_layers', default=8, type=int, help="教师模型隐藏层数量")
    parser.add_argument('--student_use_moe', default=0, type=int, choices=[0, 1], help="学生模型是否使用MoE（0=否，1=是）")
    parser.add_argument('--teacher_use_moe', default=1, type=int, choices=[0, 1], help="教师模型是否使用MoE（0=否，1=是）")
    parser.add_argument('--from_student_weight', default='full_sft', type=str, help="学生模型基于哪个权重")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help="教师模型基于哪个权重")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--alpha', default=0.5, type=float, help="CE损失权重，总损失=alpha*CE+(1-alpha)*KL")
    parser.add_argument('--temperature', default=1.5, type=float, help="蒸馏温度（推荐范围1.0-2.0）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Distillation", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1],
                        help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    Logger(f'Training device: {args.device}')

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config_student = MiniMindConfig(hidden_size=args.student_hidden_size, num_hidden_layers=args.student_num_layers,
                                       use_moe=bool(args.student_use_moe))
    lm_config_teacher = MiniMindConfig(hidden_size=args.teacher_hidden_size, num_hidden_layers=args.teacher_num_layers,
                                       use_moe=bool(args.teacher_use_moe))
    ckp_data = lm_checkpoint(lm_config_student, weight=args.save_weight,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = get_device_type(args.device)

    # MPS 上 F.scaled_dot_product_attention 性能极差（forward 慢 15x，backward 慢 100x+），
    # 强制关闭 flash_attn，使用手动 attention 实现
    if device_type == "mps":
        for cfg in [lm_config_student, lm_config_teacher]:
            if cfg.flash_attn:
                cfg.flash_attn = False
        Logger('⚡ MPS: flash_attn disabled (SDPA is extremely slow on MPS, using manual attention)')

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if device_type == "cuda":
        autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
    elif device_type == "mps":
        autocast_ctx = torch.autocast(device_type="mps", dtype=dtype)
    else:
        autocast_ctx = nullcontext()

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义学生和教师模型 ==========
    model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
    Logger(f'学生模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    Logger(f'教师模型总参数量：{sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    use_scaler = (device_type == "cuda" and args.dtype == "float16")
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_scaler) if device_type == "cuda" else torch.amp.GradScaler(
        enabled=False)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        if use_scaler and 'scaler' in ckp_data:
            scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch);
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers,
                            pin_memory=(device_type == "cuda"))
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, teacher_model, lm_config_student, start_step, wandb,
                        args.alpha, args.temperature)
        else:
            train_epoch(epoch, loader, len(loader), teacher_model, lm_config_student, 0, wandb, args.alpha,
                        args.temperature)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
