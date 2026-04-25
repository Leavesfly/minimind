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
from model.model_lora import apply_lora, save_lora
from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import (
    Logger, SkipBatchSampler, build_train_dataloader, get_default_device, get_device_type, get_lr,
    init_distributed_mode, init_model, is_main_process, lm_checkpoint,
    restore_training_state, save_checkpoint, setup_precision_context, setup_seed,
    setup_wandb, wrap_model_for_training,
)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    """
    执行一个 epoch 的 LoRA 微调训练
    
    LoRA 微调的关键特点：
    1. 仅优化 LoRA 参数（lora_params），冻结基础模型参数
    2. 梯度裁剪仅应用于 LoRA 参数
    3. 保存时只保存 LoRA 权重，大幅减少存储空间
    
    训练循环包含以下关键步骤：
    1. 前向传播计算 loss（包含 logits loss 和 aux loss）
    2. 反向传播计算梯度
    3. 梯度累积（accumulation_steps）
    4. 梯度裁剪（仅针对 lora_params）
    5. 优化器更新（仅更新 LoRA 参数）
    6. 定期记录日志和保存 LoRA 权重
    
    Args:
        epoch: 当前 epoch 索引
        loader: 数据加载器
        iters: 本 epoch 总步数
        lora_params: LoRA 参数列表（用于梯度裁剪）
        start_step: 起始 step（用于恢复训练）
        wandb: WandB 日志记录器（可选）
    """
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # LoRA 特有：仅对 LoRA 参数进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss,
                                 "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth'
            # LoRA只保存LoRA权重
            save_lora(model, lora_save_path)
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        # LoRA 特有：仅对 LoRA 参数进行梯度裁剪
        torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # ========== 参数解析 ==========
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--lora_name", type=str, default="lora_medical",
                        help="LoRA权重名称(如lora_identity/lora_medical等)")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default=get_default_device(), help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../.dataset/lora_medical.jsonl", help="LoRA训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练，默认full_sft")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1],
                        help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    Logger(f'Training device: {args.device}')

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               use_moe=bool(args.use_moe), num_key_value_heads=2)
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度（统一工具，自动处理 cuda/mps/cpu 差异） ==========
    device_type = get_device_type(args.device)
    autocast_ctx, scaler, use_scaler = setup_precision_context(device_type, args.dtype, lm_config)

    # ========== 4. 配 wandb（统一工具，自动支持断点续训） ==========
    wandb = setup_wandb(args, ckp_data, run_name_prefix=f"MiniMind-LoRA-{args.lora_name}")

    # ========== 5. 定义模型、应用LoRA、冻结非LoRA参数 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # LoRA 特有：在模型中插入 LoRA 适配器
    apply_lora(model)

    # LoRA 特有：统计参数量，展示 LoRA 的参数效率
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")

    # LoRA 特有：冻结非LoRA参数，仅收集LoRA参数用于优化
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    # ========== 6. 定义数据和优化器 ==========
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    use_scaler = (device_type == "cuda" and args.dtype == "float16")
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_scaler) if device_type == "cuda" else torch.amp.GradScaler(
        enabled=False)
    # LoRA 特有：优化器仅优化 LoRA 参数
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)

    # ========== 7. 从 ckp 恢复状态（LoRA 用 strict=False 允许部分加载） ==========
    start_epoch, start_step = restore_training_state(
        ckp_data, model, optimizer=optimizer, scaler=scaler, strict=False)

    # ========== 8. 编译和分布式包装（统一工具） ==========
    model = wrap_model_for_training(model, use_compile=bool(args.use_compile), local_rank=local_rank)

    # ========== 9. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers,
                            pin_memory=(device_type == "cuda"))
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)

    # ========== 10. 清理分布进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
