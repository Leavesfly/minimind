# -*- coding: utf-8 -*-
"""
MiniMind 预训练脚本（Pretraining）

本脚本用于 MiniMind 语言模型的预训练阶段，采用因果语言建模（Causal Language Modeling）任务，
通过大规模无标注文本数据训练模型学习语言的统计规律和语义表示。

核心特性：
- 分布式训练：支持 DDP 多卡并行训练
- 混合精度：支持 bfloat16/float16 加速训练，自动适配 CUDA/MPS/CPU 设备
- 梯度累积：通过 accumulation_steps 模拟更大的 batch size，节省显存
- 梯度裁剪：防止梯度爆炸，稳定训练过程
- 断点续训：自动保存和恢复训练状态（epoch、step、optimizer、scaler）
- 日志监控：支持 WandB 和 TensorBoard 可视化训练指标
- 设备优化：针对 MPS（Apple Silicon）和 CUDA 进行性能调优

使用方式：
    python trainer/train_pretrain.py \
        --epochs 2 \
        --batch_size 8 \
        --learning_rate 5e-4 \
        --accumulation_steps 32 \
        --data_path .dataset/pretrain_t2t_mini.jsonl \
        --save_dir out \
        --use_wandb

算法说明：
- 优化器：AdamW，带线性学习率衰减调度
- Loss 计算：CrossEntropyLoss + Auxiliary Loss（MoE 架构下的负载均衡损失）
- 序列长度：max_seq_len 控制输入序列的最大 token 数
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings

from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import (
    Logger, SkipBatchSampler, build_train_dataloader, get_default_device,
    get_device_type, get_lr, init_distributed_mode, init_model, is_main_process,
    lm_checkpoint, restore_training_state, save_checkpoint,
    setup_precision_context, setup_seed, setup_wandb, wrap_model_for_training,
)

warnings.filterwarnings('ignore')


def format_duration(seconds):
    """
    将秒数格式化为可读的时间字符串
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串，如 "30s", "5.5min", "2h30m"
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h{minutes:02d}m"


def get_memory_usage(device_type):
    """
    获取当前设备的内存使用信息
    
    Args:
        device_type: 设备类型，支持 "cuda" 或 "mps"
        
    Returns:
        内存使用信息字符串，如 "mem: 12.3/15.0GB"
    """
    if device_type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        return f"mem: {allocated:.1f}/{reserved:.1f}GB"
    elif device_type == "mps":
        allocated = torch.mps.current_allocated_memory() / 1024 ** 3
        return f"mem: {allocated:.2f}GB"
    return ""


def make_progress_bar(current, total, bar_length=20):
    """
    生成文本进度条
    
    Args:
        current: 当前进度
        total: 总进度
        bar_length: 进度条长度（字符数）
        
    Returns:
        进度条字符串，如 "|████████░░░░░░░░| 40.0%"
    """
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    percent = 100.0 * current / total
    return f"|{bar}| {percent:.1f}%"


def train_epoch(epoch, loader, iters, start_step=0, wandb=None, tb_writer=None):
    """
    执行一个 epoch 的预训练
    
    本函数实现完整的训练循环，包含前向传播、反向传播、梯度累积、优化器更新等核心步骤。
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
        tb_writer: TensorBoard SummaryWriter，用于本地可视化（可选）
        
    Returns:
        None。训练状态通过全局变量 optimizer、scaler、model 等隐式更新。
        
    Note:
        - 使用 running_loss_sum 在 GPU 上累加 loss，避免每步 .item() 导致的 GPU→CPU 同步开销
        - 只在日志步（is_log_step）进行一次 GPU→CPU 同步，显著提升训练速度
        - epoch 结束时若最后一步未完成梯度累积，会强制执行一次优化器更新
    """
    # ===== 初始化训练状态变量 =====
    epoch_start_time = time.time()
    last_step = start_step
    log_start_time = time.time()
    log_step_count = 0
    running_loss_sum = torch.tensor(0.0, device=args.device)

    # 预计算每 batch 的有效 token 数（固定序列长度下近似恒定，用于计算 tokens/sec）
    tokens_per_batch = args.batch_size * args.max_seq_len

    first_step_logged = False

    # ===== 主训练循环：遍历每个 batch =====
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 首个 batch 加载时打印提示，便于确认数据加载正常
        if not first_step_logged:
            Logger(f'📦 First batch loaded, starting forward pass (step {step})...')
            first_step_logged = True

        # 将数据移至训练设备，non_blocking=True 允许与 CUDA kernel 异步重叠，提升吞吐
        input_ids = input_ids.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        last_step = step

        # 【学习率调度】根据全局 step 计算当前学习率（线性衰减策略）
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 【前向传播】在混合精度上下文（autocast）中执行，自动选择合适的数据类型加速计算
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            # 总 loss = logits 预测损失 + aux 辅助损失（MoE 架构下的负载均衡项）
            loss = res.loss + res.aux_loss
            # 梯度累积：将 loss 缩小，使得累加 accumulation_steps 次后等效于一次大 batch
            scaled_loss = loss / args.accumulation_steps

        # 【反向传播】根据是否使用 GradScaler 选择不同的 backward 路径
        if use_scaler:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # 【优化器更新】每 accumulation_steps 步执行一次梯度裁剪和参数更新
        if step % args.accumulation_steps == 0:
            if use_scaler:
                # GradScaler 路径：先 unscale 恢复真实梯度值，再裁剪和更新
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 非 GradScaler 路径（如 bfloat16 或 CPU）：直接裁剪和更新
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            # 清空梯度，set_to_none=True 比 zero_() 更节省内存
            optimizer.zero_grad(set_to_none=True)

        # 【Loss 累加】在 GPU 上累加 loss，避免每步 .item() 导致的 GPU→CPU 同步开销
        running_loss_sum += loss.detach()
        log_step_count += 1

        # 判断是否为日志步：每隔 log_interval、最后一步、或起始步时记录
        is_log_step = (step % args.log_interval == 0 or step == iters or step == start_step + 1)

        # 【日志记录】在日志步执行指标计算和上报
        if is_log_step:
            # 只在日志步做一次 GPU→CPU 同步，减少通信开销
            now = time.time()
            elapsed_since_log = now - log_start_time
            elapsed_total = now - epoch_start_time
            steps_done = step - start_step

            # 计算平均 loss 和当前 step 的各项指标
            avg_loss = (running_loss_sum / max(log_step_count, 1)).item()
            current_loss = loss.item()
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss  # 分离 logits loss 和 aux loss
            current_lr = optimizer.param_groups[-1]['lr']

            # 计算吞吐量：tokens/sec，衡量训练速度
            log_tokens = log_step_count * tokens_per_batch
            tokens_per_sec = log_tokens / max(elapsed_since_log, 1e-6)
            avg_step_time = elapsed_total / max(steps_done, 1)
            eta_seconds = avg_step_time * (iters - step)  # 预计剩余时间

            global_step = epoch * iters + step
            global_total = args.epochs * iters
            progress_bar = make_progress_bar(global_step, global_total)

            mem_info = get_memory_usage(device_type)

            # 控制台日志输出
            Logger(
                f'{progress_bar} '
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) | '
                f'loss: {current_loss:.4f} (avg: {avg_loss:.4f}) | '
                f'logits: {current_logits_loss:.4f} aux: {current_aux_loss:.4f} | '
                f'lr: {current_lr:.2e} | '
                f'{tokens_per_sec:.0f} tok/s | '
                f'{avg_step_time * 1000:.0f}ms/step | '
                f'ETA: {format_duration(eta_seconds)} | '
                f'{mem_info}'
            )

            # WandB 日志上报
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "avg_loss": avg_loss,
                    "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "tokens_per_sec": tokens_per_sec,
                    "epoch_eta_min": eta_seconds / 60,
                })

            # TensorBoard 日志上报
            if tb_writer:
                tb_writer.add_scalar('Loss/current', current_loss, global_step)
                tb_writer.add_scalar('Loss/average', avg_loss, global_step)
                tb_writer.add_scalar('Loss/logits', current_logits_loss, global_step)
                tb_writer.add_scalar('Loss/aux', current_aux_loss, global_step)
                tb_writer.add_scalar('Training/learning_rate', current_lr, global_step)
                tb_writer.add_scalar('Training/tokens_per_sec', tokens_per_sec, global_step)
                tb_writer.add_scalar('Training/ms_per_step', avg_step_time * 1000, global_step)
                tb_writer.add_scalar('Training/eta_minutes', eta_seconds / 60, global_step)

            # 重置日志计数器
            log_start_time = now
            log_step_count = 0
            running_loss_sum.zero_()

        # 【模型保存】定期保存检查点，仅主进程执行
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            # 统一保存：推理权重 + 训练状态（自动处理 DDP / torch.compile 包装）
            save_checkpoint(model, lm_config, args.save_dir, args.save_weight,
                            optimizer=optimizer, scaler=scaler,
                            epoch=epoch, step=step, wandb=wandb)
            model.train()
            # MPS 设备手动清理缓存，避免内存泄漏
            if device_type == "mps":
                torch.mps.empty_cache()

        # 显式删除中间变量，帮助 Python GC 及时回收内存
        del input_ids, labels, res, loss, scaled_loss

    # ===== Epoch 结束汇总 =====
    epoch_elapsed = time.time() - epoch_start_time
    total_tokens = (last_step - start_step) * tokens_per_batch
    avg_tokens_per_sec = total_tokens / max(epoch_elapsed, 1e-6)
    Logger(
        f'✅ Epoch {epoch + 1}/{args.epochs} complete | '
        f'steps: {last_step - start_step} | '
        f'time: {format_duration(epoch_elapsed)} | '
        f'avg: {avg_tokens_per_sec:.0f} tok/s | '
        f'tokens: {total_tokens:,}'
    )

    # 【尾部梯度刷新】若最后一步未完成梯度累积，强制执行一次优化器更新，确保所有梯度都被应用
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        if use_scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # ===== 主程序入口 =====
    
    # 项目根目录（trainer/ 的上一级），用于构建默认路径
    _project_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    # ===== 阶段一：参数解析 =====
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default=os.path.join(_project_root, 'out'), help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default=get_default_device(), help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=32, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=50, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str,
                        default=os.path.join(_project_root, '.dataset', 'pretrain_t2t_mini.jsonl'),
                        help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--use_tb", action="store_true", help="是否使用TensorBoard可视化")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1],
                        help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # 将路径参数转为基于脚本目录的绝对路径，避免 cwd 差异和第三方库路径校验问题
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    for _attr in ('save_dir', 'data_path'):
        _path = getattr(args, _attr)
        if not os.path.isabs(_path):
            _path = os.path.join(_script_dir, _path)
        setattr(args, _attr, os.path.normpath(_path))

    # ===== 阶段二：初始化分布式环境和随机种子 =====
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    # 为每个进程设置不同的随机种子，确保数据打乱的多样性
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ===== 阶段三：配置目录、模型参数、检查点 =====
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               use_moe=bool(args.use_moe), num_key_value_heads=2)
    # 若启用断点续训，加载最近的检查点数据
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

    # ===== 阶段四：设置混合精度与设备优化 =====
    device_type = get_device_type(args.device)

    if device_type == "cuda":
        # TF32：在 Ampere+ GPU 上用 TF32 替代 FP32 做矩阵乘法和卷积，精度损失极小但速度提升显著
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # cuDNN benchmark：自动为当前输入尺寸选择最快的卷积算法（固定输入尺寸时效果最佳）
        torch.backends.cudnn.benchmark = True
        # 设置 CUDA 内存分配器为可扩展段模式，减少内存碎片
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        Logger('⚡ CUDA: TF32 + cuDNN benchmark + expandable_segments enabled')

    # MPS 上 F.scaled_dot_product_attention 性能极差（forward 慢 15x，backward 慢 100x+），
    # 强制关闭 flash_attn，使用手动 attention 实现
    if device_type == "mps" and lm_config.flash_attn:
        lm_config.flash_attn = False
        Logger('⚡ MPS: flash_attn disabled (SDPA is extremely slow on MPS, using manual attention)')

    # 根据设备类型选择混合精度策略
    if device_type == "mps":
        # MPS 上 autocast + GradScaler 有巨大开销（实测慢 3x+），直接用 fp32 原生计算最快
        Logger('⚡ MPS: autocast/scaler disabled (native fp32 is fastest on Apple Silicon)')
        autocast_ctx = nullcontext()
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    elif device_type == "cuda":
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
    else:
        autocast_ctx = nullcontext()

    # ===== 阶段五：打印训练环境摘要 =====
    Logger('=' * 60)
    Logger(f'  MiniMind Pretraining')
    Logger(f'  Device: {args.device} | dtype: {args.dtype}')
    Logger(f'  Epochs: {args.epochs} | Batch: {args.batch_size} | Accum: {args.accumulation_steps}')
    Logger(f'  Effective batch: {args.batch_size * args.accumulation_steps}')
    Logger(f'  LR: {args.learning_rate} | Grad clip: {args.grad_clip}')
    Logger(f'  Max seq len: {args.max_seq_len} | Workers: {args.num_workers}')
    Logger(f'  Model: hidden={args.hidden_size}, layers={args.num_hidden_layers}, MoE={bool(args.use_moe)}')
    Logger('=' * 60)

    # ===== 阶段六：配置日志系统（WandB + TensorBoard）=====
    wandb = setup_wandb(args, ckp_data, run_name_prefix="MiniMind-Pretrain")

    tb_writer = None
    if args.use_tb and is_main_process():
        from torch.utils.tensorboard import SummaryWriter

        tb_log_dir = os.path.join(args.save_dir, 'tb_logs', time.strftime('%Y%m%d_%H%M%S'))
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        Logger(f'📊 TensorBoard enabled → {tb_log_dir}')
        Logger(f'   Run: tensorboard --logdir {os.path.abspath(tb_log_dir)}')

    # ===== 阶段七：初始化模型、数据集、优化器 =====
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 统一内存架构：MPS 设备上数据直接放 GPU，训练时零拷贝
    dataset_device = args.device if device_type == "mps" else None
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len, device=dataset_device)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # 配置 GradScaler：仅 CUDA + float16 时启用
    use_scaler = (device_type == "cuda" and args.dtype == "float16")
    if device_type == "cuda":
        scaler = torch.amp.GradScaler(device="cuda", enabled=use_scaler)
    else:
        scaler = torch.amp.GradScaler(enabled=False)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    Logger(f'Dataset size: {len(train_ds):,} samples')
    total_steps_per_epoch = (len(train_ds) + args.batch_size - 1) // args.batch_size
    total_tokens_estimate = len(train_ds) * args.max_seq_len * args.epochs
    Logger(f'Steps/epoch: ~{total_steps_per_epoch:,} | Total tokens (est): ~{total_tokens_estimate:,}')

    # ===== 阶段八：从检查点恢复训练状态 =====
    start_epoch, start_step = restore_training_state(ckp_data, model, optimizer=optimizer, scaler=scaler)
    if ckp_data:
        Logger(f'Resumed from epoch {start_epoch}, step {start_step}')

    # ===== 阶段九：编译和分布式包装 =====
    model = wrap_model_for_training(model, use_compile=bool(args.use_compile), local_rank=local_rank)

    # ===== 阶段十：DataLoader 性能调优 =====
    # MPS 统一内存：数据已在 GPU 上，num_workers=0 避免跨进程拷贝 GPU tensor
    # CUDA：保留 multi-worker + pin_memory 的标准优化
    if device_type == "mps":
        Logger(f'⚡ MPS unified memory: num_workers → 0 (data already on GPU, zero-copy)')
        args.num_workers = 0
    pin_memory = (device_type == "cuda")
    use_persistent_workers = (args.num_workers > 0)
    prefetch_factor = 2 if args.num_workers > 0 else None

    # ===== 阶段十一：开始训练循环 =====
    training_start_time = time.time()
    Logger(f'\n🚀 Training started at {time.strftime("%Y-%m-%d %H:%M:%S")}')

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # DDP 模式下，每个 epoch 重新打乱数据
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=use_persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb, tb_writer)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb, tb_writer)

    # ===== 阶段十二：训练完成汇总 =====
    total_training_time = time.time() - training_start_time
    Logger(f'\n🎉 Training complete! Total time: {format_duration(total_training_time)}')
    Logger(f'   Finished at {time.strftime("%Y-%m-%d %H:%M:%S")}')

    # ===== 阶段十三：资源清理 =====
    if tb_writer:
        tb_writer.close()
    if device_type == "mps":
        torch.mps.empty_cache()
    if dist.is_initialized():
        dist.destroy_process_group()
