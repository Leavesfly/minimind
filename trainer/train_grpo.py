# -*- coding: utf-8 -*-
"""
MiniMind GRPO (Group Relative Policy Optimization) / CISPO 训练脚本

本脚本实现了基于 GRPO/CISPO 算法的强化学习训练，用于优化语言模型的生成策略。
主要特点：
1. GRPO: Group Relative Policy Optimization，通过组内相对优势估计，无需训练 Critic 模型
2. CISPO: Constrained Importance Sampling Policy Optimization，一种变体的策略优化算法
3. 支持多种奖励机制：长度奖励、思考奖励、重复惩罚、Reward Model 评分
4. 支持多种 rollout 引擎：torch 原生推理、SGLang 高性能推理引擎
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import re
import warnings

from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler

from dataset.lm_dataset import RLAIFDataset
from model.model_minimind import MiniMindConfig
from trainer.reward_utils import compute_repetition_penalty
from trainer.rollout_engine import compute_per_token_logps, create_rollout_engine
from trainer.trainer_utils import (
    LMForRewardModel, Logger, SkipBatchSampler, build_train_dataloader,
    get_default_device, get_device_type, init_distributed_mode, init_model,
    is_main_process, lm_checkpoint, restore_training_state, save_checkpoint,
    setup_precision_context, setup_seed, setup_wandb, wrap_model_for_training,
)

warnings.filterwarnings('ignore')


# n-gram 重复惩罚已统一抽取至 trainer.reward_utils.compute_repetition_penalty，
# 这里保留 rep_penalty 别名以兼容本文件中现有的奖励计算逻辑
rep_penalty = compute_repetition_penalty


def calculate_rewards(prompts, responses, reward_model):
    """计算生成样本的综合奖励值

    奖励由多个维度组成：
    1. 长度奖励：20~800 字符的回答 +0.5，否则 -0.5
    2. 思考奖励：包含 </think> 且思考内容长度合理 +1.0
    3. 思考格式奖励：</think> 只出现一次 +0.25
    4. 重复惩罚：通过 n-gram 重复检测扣分
    5. Reward Model 评分：使用外部奖励模型对回答质量打分

    Args:
        prompts (list[str]): 输入的 prompt 列表，长度为 B（batch size）
        responses (list[str]): 模型生成的回答列表，长度为 B * num_generations
        reward_model: 奖励模型实例，用于对回答质量进行评分

    Returns:
        torch.Tensor: 形状为 [B * num_generations] 的奖励张量
    """
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # === 准备 Reward Model 输入 ===
                # 从 prompt 中解析对话历史，用于 Reward Model 评分
                # 解析 <|im_start|>role content<|im_end|> 格式的对话标记
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer = response
                
                # === 奖励维度 1: 长度奖励 ===
                # 鼓励生成适当长度的回答（20~800 字符），过短或过长都会扣分
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
                
                # === 奖励维度 2 & 3: 思考内容奖励 ===
                if '</think>' in response:
                    # 思考奖励：鼓励包含合理长度的思考内容（20~300 字符）
                    thinking_content, answer_content = response.split('</think>', 1)
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    # 格式奖励：</think> 标记只应出现一次，多次出现说明格式错误
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()
                
                # === 奖励维度 4: 重复惩罚 ===
                # 通过 n-gram 重复检测对冗余内容进行扣分
                rewards[response_idx] -= rep_penalty(answer)

                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None,
                     use_sglang=False):
    """
    GRPO 训练一个 epoch
    
    GRPO 训练流程：
    1. Rollout: 使用当前策略模型生成多个回答样本
    2. 计算奖励: 使用多种奖励机制评估每个样本的质量
    3. 优势估计: 通过组内相对优势计算，无需训练 Critic 模型
    4. 策略更新: 使用 PPO clip 或 CISPO 损失更新策略
    5. KL 散度约束: 通过参考模型约束策略更新幅度
    
    参数:
        epoch (int): 当前训练轮数
        loader (DataLoader): 数据加载器
        iters (int): 总迭代次数
        rollout_engine: Rollout 引擎（支持 torch 和 SGLang）
        ref_model: 参考模型（用于 KL 散度计算）
        reward_model: 奖励模型
        start_step (int): 起始 step
        wandb: wandb 日志记录器
        use_sglang (bool): 是否使用 SGLang 引擎
    """
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']  # list[str], length B
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # ==================== 阶段 1: Rollout 生成 ====================
        # 使用当前策略模型为每个 prompt 生成 num_generations 个回答样本
        # Rollout 引擎支持 torch 原生推理和 SGLang 高性能推理
        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,  # 每个 prompt 生成的样本数
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        # 提取 rollout 结果
        outputs = rollout_result.output_ids  # 完整的 token IDs [B*num_gen, seq_len]
        completion_ids = rollout_result.completion_ids  # 仅生成部分的 token IDs [B*num_gen, gen_len]
        completions = rollout_result.completions  # 生成的文本列表
        # old_per_token_logps: rollout 时策略模型的对数概率，用于后续计算 importance sampling ratio
        old_per_token_logps = rollout_result.per_token_logps.to(args.device)

        # ==================== 阶段 2: 前向传播获取当前策略的对数概率 ====================
        # 获取解包后的模型（处理 DDP 包装）
        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            if use_sglang or lm_config.use_moe:
                # SGLang 或 MoE 架构需要重新计算 logits
                res = model_unwrapped(outputs)
                aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
                logits = res.logits[:, :-1, :]  # 移除最后一个 token 的 logits
                # 计算每个 token 的对数概率：从 logits 中提取生成部分的 log probability
                per_token_logps = F.log_softmax(logits, dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1)[
                    :, -completion_ids.size(1):]
            else:
                # 非 SGLang 且非 MoE 时，直接使用 rollout 时计算的 logps
                aux_loss = torch.tensor(0.0, device=args.device)
                per_token_logps = rollout_result.per_token_logps

        # 使用参考模型计算对数概率，用于 KL 散度约束
        # ref_per_token_logps: 参考模型（冻结的策略副本）的对数概率
        with torch.no_grad():
            ref_per_token_logps = compute_per_token_logps(ref_model, outputs, completion_ids.size(1))

        # ==================== 阶段 3: 计算奖励 ====================
        # 调用奖励函数，综合长度、思考内容、重复惩罚和 Reward Model 评分
        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-' * 100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                Logger('=' * 100)

        # ==================== 阶段 4: 优势估计（GRPO 核心）====================
        # GRPO 通过组内相对优势估计，无需训练 Critic 模型
        # 将奖励重塑为 [B, num_generations] 形状，每个 prompt 对应一组样本
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        # 计算每组内的均值和标准差，用于归一化
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        # 组内归一化优势：(reward - mean) / (std + epsilon)
        # 这种归一化使得同一 prompt 下的不同回答可以相互比较
        advantages = (rewards - mean_r) / (std_r + 1e-4)  # [B*num_gen]

        # ==================== 阶段 5: 构建完成掩码（completion mask）====================
        # 识别每个生成序列中 EOS token 的位置，构建有效 token 的掩码
        # is_eos: 标记哪些位置是 EOS token [B*num_gen, R]
        is_eos = completion_ids == tokenizer.eos_token_id  # [B*num_gen, R]
        # eos_idx: 记录每个序列中第一个 EOS token 的索引位置
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        # completion_mask: 布尔掩码，标记从开始到 EOS（含）之间的所有有效 token 位置
        # 用于在损失计算时屏蔽 padding 和 EOS 之后的无效 token
        completion_mask = (
                torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(
            1)).int()  # [B*num_gen, R]

        # ==================== 阶段 6: 计算 KL 散度惩罚项 ====================
        # KL 散度用于约束新策略与参考策略之间的偏离程度
        # kl_div = ref_logp - new_logp，当新策略偏离参考策略时产生惩罚
        kl_div = ref_per_token_logps - per_token_logps
        # per_token_kl: 使用 KL(p_ref || p_new) 的近似公式: exp(kl_div) - kl_div - 1
        # 这是一个非负值，当两个分布相同时为 0
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]

        # ==================== 阶段 7: 策略更新（核心优化步骤）====================
        # 计算 importance sampling ratio: ratio = exp(new_logp - old_logp)
        # ratio > 1 表示新策略比旧策略更倾向于该动作，ratio < 1 则相反
        ratio = torch.exp(per_token_logps - old_per_token_logps)  # [B*num_gen, R]
        if args.loss_type == "cispo":
            # ==================== CISPO 损失计算 ====================
            # CISPO (Constrained Importance Sampling Policy Optimization) 使用单侧截断策略
            # 与 PPO 的双侧 clip 不同，CISPO 只限制 ratio 的上界，允许下界自由变化
            # clamped_ratio: 将 importance sampling ratio 截断到 [0, epsilon_high] 范围
            # .detach() 确保截断后的 ratio 不参与梯度回传，避免梯度通过截断操作传播
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            # CISPO 损失：-clamped_ratio * advantage * logp + beta * KL
            # 注意：这里乘以 per_token_logps 而非传统的 1，是 CISPO 的变体设计
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            # ==================== PPO Clip 损失计算 ====================
            # 标准 PPO 使用双侧 clip 来限制策略更新幅度
            # clipped_ratio: 将 ratio 截断到 [1-epsilon, 1+epsilon] 范围
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            # per_token_loss1: 未截断的损失项
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            # per_token_loss2: 截断后的损失项
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            # 取两者中的较小值（因为前面有负号，实际上是取较大的保守估计）
            # 最终损失：-min(ratio * adv, clipped_ratio * adv) + beta * KL
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)

        # ==================== 阶段 8: 计算最终损失并反向传播 ====================
        # policy_loss: 对每个样本的 per_token_loss 按有效 token 掩码加权平均
        # 先对每个样本内部求平均（除以有效 token 数），再对所有样本求均值
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # 总损失 = 策略损失 + 辅助损失（MoE 的负载均衡损失），除以梯度累积步数
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
        loss.backward()

        # ==================== 阶段 9: 梯度累积与优化器更新 ====================
        # 每 accumulation_steps 步执行一次真正的参数更新
        if step % args.accumulation_steps == 0:
            # 梯度裁剪：防止梯度过大导致训练不稳定
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 更新模型参数
            optimizer.step()
            # 更新学习率调度器
            scheduler.step()
            # 清空梯度
            optimizer.zero_grad()
            # 定期同步 rollout 引擎中的策略模型（确保生成时使用最新策略）
            if is_main_process() and step % args.save_interval == 0: rollout_engine.update_policy(model)

        # ==================== 阶段 10: 日志记录与监控 ====================
        # 定期打印训练指标到控制台和 wandb（如果启用）
        if step % args.log_interval == 0 or step == iters:
            # 还原实际的策略损失值（乘以累积步数）
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            # 平均奖励：反映生成样本的整体质量
            avg_reward_val = rewards.mean().item()
            # 平均响应长度：监控生成长度的变化趋势
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            # KL 散度均值：衡量新策略与参考策略的偏离程度
            kl_ref_val = ((
                                  ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / completion_mask.sum().item()
            # 优势的均值和标准差：用于监控奖励分布的稳定性
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()
            # 当前学习率
            current_lr = optimizer.param_groups[0]['lr']

            # 控制台日志输出
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {policy_loss_val:.4f}, Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            # WandB 日志记录（仅主进程）
            if wandb and is_main_process():
                wandb.log({
                    "reward": avg_reward_val,
                    "kl_ref": kl_ref_val,
                    "advantages_std": advantages_std_val,
                    "advantages_mean": advantages_mean_val,
                    "policy_loss": policy_loss_val,
                    "avg_response_len": avg_len_val,
                    "learning_rate": current_lr
                })

        # ==================== 阶段 11: 模型检查点保存 ====================
        # 定期保存模型权重和训练状态，支持断点续训
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()  # 切换到评估模式以确保保存的权重一致
            # 统一保存：推理权重 + 训练状态（GRPO 需保存 scheduler 以支持学习率恢复）
            save_checkpoint(model, lm_config, args.save_dir, args.save_weight,
                            optimizer=optimizer, scheduler=scheduler,
                            epoch=epoch, step=step, wandb=wandb)
            model.train()  # 恢复训练模式

    # ==================== 处理剩余的梯度累积 ====================
    # 如果最后一个 batch 没有达到 accumulation_steps，仍需执行一次参数更新
    if step > start_step and step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if is_main_process() and step % args.save_interval == 0: rollout_engine.update_policy(model)

        # 释放中间变量以节省内存
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default=get_default_device(), help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../.dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=6, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1],
                        help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="sglang", choices=["torch", "sglang"],
                        help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8996", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ==================== 主程序入口：初始化与配置 ====================
    
    # ---------- 步骤 1: 初始化分布式环境和随机种子 ----------
    # init_distributed_mode: 初始化分布式训练模式（DDP），返回本地 rank
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        # 分布式环境下，每个进程使用不同的 CUDA 设备
        args.device = f"cuda:{local_rank}"
    # 设置随机种子，确保可复现性；分布式环境下不同 rank 使用不同种子
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    Logger(f'Training device: {args.device}')

    # ---------- 步骤 2: 配置模型参数和检查点 ----------
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    # 初始化 MiniMind 模型配置
    # max_seq_len 需要覆盖 prompt + 生成的总长度
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe),
                               num_key_value_heads=2)
    # 如果启用断点续训，加载之前的检查点数据
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

    # ---------- 步骤 3: 设置混合精度上下文 ----------
    # 根据设备类型（cuda/mps/cpu）和数据类型（bfloat16/float16）自动配置 autocast
    device_type = get_device_type(args.device)
    autocast_ctx, _scaler_unused, _ = setup_precision_context(device_type, args.dtype, lm_config)

    # ---------- 步骤 4: 配置 WandB 日志 ----------
    # setup_wandb 自动支持从检查点恢复 wandb run_id，实现断点续训时的日志连续性
    wandb = setup_wandb(args, ckp_data, run_name_prefix="MiniMind-GRPO")

    # ---------- 步骤 5: 初始化模型、Rollout 引擎和数据加载器 ----------
    base_weight = args.from_weight
    
    # Policy 模型：当前正在训练的策略模型
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    
    # Reference 模型：冻结的策略副本，用于 KL 散度约束
    # 参考模型不参与梯度更新，仅用于计算与当前策略的偏离程度
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    
    # Reward 模型：用于对生成回答的质量进行评分
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    
    # Rollout 引擎：负责使用当前策略生成回答样本
    # 支持 torch 原生推理和 SGLang 高性能推理引擎，可插拔替换
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    
    # 初始化 RLAIF 数据集
    # thinking_ratio: 控制生成思考内容的概率（0.0~1.0）
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len,
                            thinking_ratio=args.thinking_ratio)
    
    # 分布式采样器：确保每个进程处理不同的数据子集
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # 优化器：使用 AdamW 优化策略模型参数
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 计算总迭代次数和学习率调度器
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    # 总优化步数 = (总样本数 / 累积步数) * 轮数
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    # 余弦退火学习率调度器，最小学习率为初始值的 1/10
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    # ---------- 步骤 6: 从检查点恢复训练状态 ----------
    # restore_training_state 自动恢复 model、optimizer、scheduler 的状态
    # 返回起始 epoch 和 step，支持断点续训
    start_epoch, start_step = restore_training_state(
        ckp_data, model, optimizer=optimizer, scheduler=scheduler)

    # ---------- 步骤 7: 模型编译和分布式包装 ----------
    # NOTE: 此处不复用 trainer_utils.wrap_model_for_training，因为 GRPO 需要在
    # torch.compile 完成后立即把"已编译模型"注入 rollout_engine（保证生成时使用编译版），
    # 这一耦合无法通过通用 helper 表达，故保留显式实现。
    
    # 启用 torch.compile 加速（可选）
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        # 编译后立即更新 rollout 引擎，确保生成时使用加速版本
        rollout_engine.update_policy(model)
    
    # 分布式数据并行（DDP）包装
    if dist.is_initialized():
        # MiniMind 的旋转位置编码 buffer（freqs_cos/freqs_sin）是确定性计算的，不需要 DDP 同步
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # 主进程更新 rollout 引擎中的策略模型引用
    if is_main_process():
        rollout_engine.update_policy(model)

    # ---------- 步骤 8: 开始训练循环 ----------
    for epoch in range(start_epoch, args.epochs):
        # 设置分布式采样器的 epoch，确保每个 epoch 数据shuffle不同
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # 每个 epoch 重新设置随机种子，确保可复现性
        setup_seed(42 + epoch)
        
        # 生成随机索引并创建跳过式 batch sampler（支持断点续训时跳过已处理的 batch）
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        
        # 创建数据加载器
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers,
                            pin_memory=(device_type == "cuda"))  # CUDA 设备启用 pinned memory 加速数据传输
        
        # 如果是断点续训且当前是起始 epoch，跳过已处理的 batch
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step,
                             wandb, use_sglang=(args.rollout_engine == "sglang"))
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb,
                             use_sglang=(args.rollout_engine == "sglang"))

    # ---------- 步骤 9: 清理分布式进程 ----------
    # 训练结束后销毁分布式进程组，释放资源
    if dist.is_initialized():
        dist.destroy_process_group()
