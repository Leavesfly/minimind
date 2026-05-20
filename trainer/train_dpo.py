"""
MiniMind DPO（Direct Preference Optimization）训练脚本

本脚本实现基于 DPO 算法的偏好对齐训练，用于优化模型使其更符合人类偏好。

## 功能说明
DPO 是一种无需显式奖励模型的偏好优化方法。传统 RLHF 需要训练一个独立的奖励模型（RM），
然后通过 PPO 等强化学习算法优化策略模型，流程复杂且不稳定。DPO 通过数学推导，将偏好优化
转化为直接对策略模型进行监督式训练，简化了训练流程并提高了稳定性。

## DPO 算法原理
DPO 的核心思想是通过隐式奖励建模，直接优化策略模型使其更偏好 chosen（优选）响应而非 
rejected（拒绝）响应。

### 数学推导
1. **Bradley-Terry 模型**：假设人类偏好服从 BT 模型，即对于给定 prompt x，chosen 响应 y_w 
   优于 rejected 响应 y_l 的概率为：
   P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))
   其中 r(x, y) 是隐式奖励函数，σ 是 sigmoid 函数。

2. **奖励与策略的关系**：根据最大熵 RL 理论，最优策略 π* 与奖励函数 r 的关系为：
   r(x, y) = β log(π*(y|x) / π_ref(y|x)) + β log Z(x)
   其中 π_ref 是参考策略（通常是 SFT 模型），β 是温度参数，Z(x) 是归一化常数。

3. **DPO 损失函数**：将奖励表达式代入 BT 模型，消去未知的 Z(x)，得到：
   P(y_w ≻ y_l | x) = σ(β log(π*(y_w|x)/π_ref(y_w|x)) - β log(π*(y_l|x)/π_ref(y_l|x)))
   
   DPO 通过最大化上述概率来优化策略模型，对应的损失函数为：
   L_DPO = -E[log σ(β (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x))))]

### 关键概念
- **策略模型（Policy Model）**：当前正在训练的模型 π_θ
- **参考模型（Reference Model）**：冻结的 SFT 模型 π_ref，用于计算基线对数概率
- **Chosen/Rejected 数据对**：每个样本包含同一个 prompt 的两个响应，一个是人类偏好的 
  chosen 响应，另一个是不被偏好的 rejected 响应
- **Beta 参数**：控制 KL 散度正则化的强度，beta 越大，策略模型偏离参考模型的程度越小

## 训练流程
1. **数据准备**：加载包含 (prompt, chosen, rejected) 三元组的数据集
2. **前向传播**：
   - 使用参考模型（no_grad）计算 chosen 和 rejected 的对数概率作为基线
   - 使用策略模型计算 chosen 和 rejected 的对数概率
3. **损失计算**：
   - 计算策略模型与参考模型的 log-ratio 差值
   - 应用 DPO 损失函数：loss = -log_sigmoid(beta * (pi_logratio - ref_logratio))
4. **反向传播**：仅更新策略模型参数，参考模型保持冻结

## 使用方式
```bash
# 基本用法
python trainer/train_dpo.py \
    --data_path ../.dataset/dpo.jsonl \
    --from_weight full_sft \
    --epochs 1 \
    --batch_size 4 \
    --learning_rate 4e-8 \
    --beta 0.15

# 从检查点恢复训练
python trainer/train_dpo.py \
    --from_resume 1 \
    --save_weight dpo

# 启用 WandB 日志
python trainer/train_dpo.py \
    --use_wandb \
    --wandb_project MiniMind-DPO
```

## 关键参数说明
- `--beta`: DPO 温度参数，控制策略模型偏离参考模型的程度。典型值范围：0.1~0.5
- `--learning_rate`: 建议设置为较小的值（如 4e-8 ~ 5e-8），避免过度优化导致遗忘
- `--from_weight`: 通常使用 full_sft（监督微调后的模型）作为起点
- `--batch_size`: DPO 需要成对数据，实际 batch size 应为偶数（一半 chosen，一半 rejected）
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
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler

from dataset.lm_dataset import DPODataset
from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import (
    Logger, SkipBatchSampler, build_train_dataloader, get_default_device, get_device_type, get_lr,
    init_distributed_mode, init_model, is_main_process, lm_checkpoint,
    restore_training_state, save_checkpoint, setup_precision_context, setup_seed,
    setup_wandb, wrap_model_for_training,
)

warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels):
    """
    从 logits 计算 token 级别的对数概率
    
    这是 DPO 算法中的关键辅助函数，用于将模型输出的 logits 转换为对数概率。
    
    计算步骤：
    1. 对 logits 沿词汇表维度（dim=2）进行 log_softmax，得到每个 token 在所有词汇上的对数概率分布
    2. 使用 gather 操作，根据 labels 中指定的 token ID，提取对应位置的对数概率值
    
    为什么需要这个函数？
    - DPO 损失函数依赖于策略模型和参考模型的对数概率比值
    - 直接使用 logits 无法正确计算概率比值，必须先转换为对数概率空间
    - log_softmax 确保数值稳定性，避免 softmax 中的指数溢出问题
    
    Args:
        logits: 模型输出的 logits，shape: (batch_size, seq_len, vocab_size)
                表示每个位置上每个词汇的未归一化得分
        labels: 目标标签，shape: (batch_size, seq_len)
                表示每个位置上的真实 token ID
        
    Returns:
        每个位置标签的对数概率，shape: (batch_size, seq_len)
        返回值中的每个元素 log_probs[i, j] 表示第 i 个样本第 j 个位置上，
        真实标签 tokens[j] 的对数概率 log P(tokens[j] | context)
    """
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # log_probs shape: (batch_size, seq_len)
    
    # 对 logits 进行 log_softmax，得到每个 token 在所有词汇上的对数概率分布
    # dim=2 表示沿词汇表维度进行归一化，确保每个位置上的概率之和为 1
    log_probs = F.log_softmax(logits, dim=2)
    
    # 使用 gather 操作提取对应标签位置的对数概率
    # labels.unsqueeze(2) 将 labels 从 (batch_size, seq_len) 扩展为 (batch_size, seq_len, 1)
    # gather 在 dim=2 上根据 labels 的值索引，提取每个位置上真实 token 的对数概率
    # squeeze(-1) 移除最后一维，恢复为 (batch_size, seq_len)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    计算 DPO（Direct Preference Optimization）损失
    
    DPO 的核心思想：通过直接优化策略模型，使其更偏好 chosen（优选）而非 rejected（拒绝）响应，
    而无需显式训练奖励模型。这是通过隐式奖励建模实现的。
    
    ## 算法原理详解
    
    ### 1. Bradley-Terry 偏好模型
    假设人类对两个响应 y_w（chosen）和 y_l（rejected）的偏好服从 BT 模型：
    P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))
    其中 r(x, y) 是隐式奖励函数，σ 是 sigmoid 函数。
    
    ### 2. 奖励与策略的关系（从最大熵 RL 推导）
    最优策略 π* 与奖励函数 r 的关系为：
    r(x, y) = β log(π*(y|x) / π_ref(y|x)) + β log Z(x)
    
    将这个关系代入 BT 模型，归一化常数 Z(x) 会被消去，得到：
    P(y_w ≻ y_l | x) = σ(β [log(π*(y_w|x)/π_ref(y_w|x)) - log(π*(y_l|x)/π_ref(y_l|x))])
    
    ### 3. DPO 损失函数
    为了最大化上述偏好概率，我们最小化负对数似然：
    L_DPO = -E[log σ(β (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x))))]
    
    定义：
    - π_logratio = log(π(y_w|x)) - log(π(y_l|x))  （策略模型的 chosen/rejected 对数概率差）
    - ref_logratio = log(π_ref(y_w|x)) - log(π_ref(y_l|x))  （参考模型的 chosen/rejected 对数概率差）
    - DPO logits = π_logratio - ref_logratio
    
    则损失函数简化为：
    L_DPO = -E[log σ(β * DPO logits)]
    
    ### 4. 为什么需要参考模型？
    - 防止策略模型过度优化：如果没有参考模型，策略模型可能会通过简单地增加所有 token 的概率
      来"作弊"，而不是真正学习偏好差异
    - KL 散度正则化：参考模型提供了 KL 散度约束，确保策略模型不会偏离初始 SFT 模型太远
    - Beta 参数控制正则化强度：beta 越大，策略模型越接近参考模型；beta 越小，策略模型有更大自由度
    
    ### 5. Mask 的作用
    - 序列中可能包含 padding token（值为 -100 或其他特殊值）
    - Mask 将这些位置的贡献设为 0，确保只有有效 token 参与损失计算
    - 通过对每个序列的有效 token 对数概率求和，得到整个序列的对数概率
    
    Args:
        ref_log_probs: 参考模型的对数概率，shape: (batch_size, seq_len)
                       包含 chosen 和 rejected 数据的拼接结果
        policy_log_probs: 策略模型的对数概率，shape: (batch_size, seq_len)
                          同样包含 chosen 和 rejected 数据的拼接结果
        mask: 掩码，用于过滤 padding token，shape: (batch_size, seq_len)
              有效位置为 1，padding 位置为 0
        beta: DPO 温度参数，控制优化强度和对参考模型的偏离程度
              - beta 较大（如 0.5）：策略模型更接近参考模型，优化更保守
              - beta 较小（如 0.1）：策略模型有更大自由度，可能获得更高偏好分数但风险更大
        
    Returns:
        DPO 损失的均值，标量张量
        损失值越小，表示策略模型对 chosen 响应的偏好越强
    """
    # ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)
    # 通过 mask 过滤 padding token，对每个序列的有效 token 对数概率求和
    # 得到每个样本的总对数概率（标量）
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # 将 chosen 和 rejected 数据分开
    # 输入数据格式：前一半是 chosen 样本，后一半是 rejected 样本
    # 例如 batch_size=4 时，索引 0,1 是 chosen，索引 2,3 是 rejected
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # 计算策略模型和参考模型的 log-ratio
    # pi_logratios = log(π(y_chosen|x)) - log(π(y_rejected|x))
    # 正值表示策略模型更偏好 chosen，负值表示更偏好 rejected
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    
    # ref_logratios = log(π_ref(y_chosen|x)) - log(π_ref(y_rejected|x))
    # 作为基线，衡量参考模型对 chosen/rejected 的偏好程度
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    
    # DPO logits = pi_logratios - ref_logratios
    # 这相当于：log(π(y_chosen)/π_ref(y_chosen)) - log(π(y_rejected)/π_ref(y_rejected))
    # 即策略模型相对于参考模型的"相对偏好改进"
    logits = pi_logratios - ref_logratios
    
    # DPO 损失：-log_sigmoid(beta * logits)
    # 当 logits > 0 时（策略模型比参考模型更偏好 chosen），损失较小
    # 当 logits < 0 时（策略模型比参考模型更偏好 rejected），损失较大
    # sigmoid 将 logits 映射到 (0, 1) 区间，log 将其映射到 (-∞, 0)
    # 取负号后，损失范围为 (0, +∞)，最小化损失等价于最大化偏好概率
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    """
    执行一个 epoch 的 DPO 训练
    
    DPO 训练的关键特点：
    1. 同时使用策略模型（model）和参考模型（ref_model）
    2. 参考模型冻结，不参与梯度更新
    3. 处理成对的 chosen 和 rejected 数据
    4. 使用 DPO 损失函数优化策略模型
    
    训练循环包含以下关键步骤：
    1. 使用参考模型计算基线对数概率（no_grad）
    2. 使用策略模型计算当前对数概率
    3. 计算 DPO 损失
    4. 反向传播和优化器更新
    
    Args:
        epoch: 当前 epoch 索引
        loader: 数据加载器（返回包含 chosen/rejected 数据的字典）
        iters: 本 epoch 总步数
        ref_model: 参考模型（冻结的 SFT 模型）
        lm_config: 模型配置
        start_step: 起始 step（用于恢复训练）
        wandb: WandB 日志记录器（可选）
        beta: DPO 温度参数
    """
    start_time = time.time()
    last_step = start_step

    for step, batch in enumerate(loader, start=start_step + 1):
        last_step = step
        # DPO 特有：提取 chosen 和 rejected 数据
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        # 将 chosen 和 rejected 拼接在一起
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # DPO 特有：使用参考模型计算基线对数概率（不计算梯度）
            # 为什么用 torch.no_grad()？
            # 1. 参考模型是冻结的，不需要计算梯度，节省显存和计算资源
            # 2. 避免不必要的梯度图构建，加速训练
            # 3. 确保参考模型的对数概率不会受到策略模型梯度的影响
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            # 将参考模型的 logits 转换为对数概率
            # ref_log_probs shape: (batch_size, seq_len)，包含 chosen 和 rejected 的拼接结果
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            # 使用策略模型计算当前对数概率
            # 策略模型需要计算梯度，以便后续反向传播更新参数
            outputs = model(x)
            logits = outputs.logits
            # 将策略模型的 logits 转换为对数概率
            # policy_log_probs shape: (batch_size, seq_len)，与 ref_log_probs 对应
            policy_log_probs = logits_to_log_probs(logits, y)

            # 计算 DPO 损失
            # dpo_loss 内部会：
            # 1. 将 chosen 和 rejected 数据分开
            # 2. 计算策略模型和参考模型的 log-ratio 差值
            # 3. 应用 sigmoid 和对数变换，得到最终的损失值
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            # 总损失 = DPO 损失 + 辅助损失（如 MoE 的负载均衡损失）
            # aux_loss 仅在启用 MoE 架构时非零
            loss = dpo_loss_val + outputs.aux_loss
            # 梯度累积：将损失除以累积步数
            loss = loss / args.accumulation_steps

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
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60

            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')

            if wandb: wandb.log({"loss": current_loss, "dpo_loss": current_dpo_loss, "aux_loss": current_aux_loss,
                                 "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            # 统一保存：推理权重 + 训练状态（自动处理 DDP / torch.compile 包装）
            # save_checkpoint 会保存：
            # 1. 模型权重：策略模型的完整参数（不包括参考模型，因为参考模型是冻结的 SFT 模型）
            # 2. 优化器状态：用于断点续训
            # 3. 训练状态：epoch、step、scaler 状态等
            # 4. WandB 信息：如果启用了 WandB，保存 run_id 以便恢复日志记录
            save_checkpoint(model, lm_config, args.save_dir, args.save_weight,
                            optimizer=optimizer, scaler=scaler,
                            epoch=epoch, step=step, wandb=wandb)
            model.train()

        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # ========== 参数解析 ==========
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--device", type=str, default=get_default_device(), help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../.dataset/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--beta', default=0.15, type=float, help="DPO中的beta参数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
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
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度（统一工具，自动处理 cuda/mps/cpu 差异） ==========
    device_type = get_device_type(args.device)
    autocast_ctx, scaler, use_scaler = setup_precision_context(device_type, args.dtype, lm_config)

    # ========== 4. 配 wandb（统一工具，自动支持断点续训） ==========
    wandb = setup_wandb(args, ckp_data, run_name_prefix="MiniMind-DPO")

    # ========== 5. 定义模型和参考模型 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    
    # DPO 特有：初始化参考模型（ref_model 冻结）
    # 为什么需要参考模型？
    # 1. 提供基线对数概率：DPO 损失依赖于策略模型与参考模型的对数概率比值
    # 2. KL 散度正则化：防止策略模型过度偏离初始 SFT 模型，避免灾难性遗忘
    # 3. 隐式奖励建模：参考模型代表了"中性"策略，策略模型的改进是相对于这个基线的
    #
    # 为什么参考模型要冻结？
    # - 参考模型必须保持不变，才能作为稳定的比较基准
    # - 如果参考模型也参与训练，log-ratio 会失去意义，DPO 损失无法正确优化
    # - 参考模型通常就是 SFT 模型，代表了经过监督微调后的基础能力
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()  # 设置为评估模式，禁用 dropout 等训练特有的行为
    ref_model.requires_grad_(False)  # 冻结所有参数，不参与梯度计算和更新
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')

    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    use_scaler = (device_type == "cuda" and args.dtype == "float16")
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_scaler) if device_type == "cuda" else torch.amp.GradScaler(
        enabled=False)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从 ckp 恢复状态（统一工具，自动处理 model/optimizer/scaler） ==========
    start_epoch, start_step = restore_training_state(ckp_data, model, optimizer=optimizer, scaler=scaler)

    # ========== 7. 编译和分布式包装（统一工具） ==========
    model = wrap_model_for_training(model, use_compile=bool(args.use_compile), local_rank=local_rank)

    # ========== 8. 开始训练 ==========
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
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
