# -*- coding: utf-8 -*-
"""
MLX Rollout Engine - 推理引擎
=============================================
本模块实现基于 MLX 的推理引擎，为 RL 训练提供 rollout 生成能力。

与 PyTorch 版本的主要区别：
- 使用 MLX 原生自回归生成（无需 DDP、device 管理）
- 统一内存下零拷贝权重同步
- 无需混合精度上下文（MLX 原生支持低精度）

功能特性：
- 批量自回归采样生成
- 自动计算每个 token 的 log probability
- Top-p / temperature 采样控制
- EOS 终止检测
"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn


# =====================================================================================
# 数据结构
# =====================================================================================

@dataclass
class RolloutResult:
    """Rollout 引擎的生成结果

    Attributes:
        output_ids: 完整序列（prompt + completion），形状 [B*G, P+R]
        completion_ids: 仅 completion 部分，形状 [B*G, R]
        per_token_logps: completion 中每个 token 的 logp，形状 [B*G, R]
        completions: 解码后的纯文本列表，长度 B*G
    """
    output_ids: mx.array
    completion_ids: mx.array
    per_token_logps: mx.array
    completions: List[str]


# =====================================================================================
# 工具函数
# =====================================================================================

def compute_per_token_logps(model, input_ids: mx.array, n_keep: int) -> mx.array:
    """计算序列尾部 n_keep 个 token 的对数概率

    用于 RL 训练中的：
    - 策略梯度中的 π(a|s) 项
    - 新旧策略之间的 KL 散度（GRPO / PPO）
    - Importance sampling ratio 的分子分母

    Args:
        model: 语言模型
        input_ids: 完整输入序列 [batch_size, seq_len]
        n_keep: 需要计算 logp 的尾部 token 数量

    Returns:
        形状 [batch_size, n_keep] 的 logp 数组
    """
    if n_keep <= 0:
        return mx.zeros((input_ids.shape[0], 0))

    output = model(input_ids)
    # 取尾部 n_keep+1 个位置的 logits（偏移对齐 next-token 预测）
    logits = output.logits[:, -(n_keep + 1):-1, :]
    labels = input_ids[:, -n_keep:]

    # log_softmax + gather
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    B, L, V = log_probs.shape
    batch_idx = mx.arange(B)[:, None]
    seq_idx = mx.arange(L)[None, :]
    per_token_logps = log_probs[batch_idx, seq_idx, labels]

    return per_token_logps


# =====================================================================================
# 抽象基类
# =====================================================================================

class RolloutEngine(ABC):
    """Rollout 引擎抽象基类"""

    @abstractmethod
    def rollout(self, prompt_ids: mx.array, num_generations: int,
                max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        """执行批量 rollout（采样生成）

        Args:
            prompt_ids: prompt token IDs [B, P]
            num_generations: 每个 prompt 采样的候选数 G
            max_new_tokens: 单条回答最大生成 token 数
            temperature: 采样温度

        Returns:
            RolloutResult 实例
        """

    @abstractmethod
    def update_policy(self, model: nn.Module):
        """将最新的训练模型权重同步到推理引擎"""


# =====================================================================================
# MLX 原生推理引擎
# =====================================================================================

class MLXRolloutEngine(RolloutEngine):
    """MLX 原生推理引擎——直接在 Apple Silicon 上自回归采样

    特性：
    - 利用 MLX 统一内存，权重同步零拷贝
    - 逐 token 采样，支持 EOS 早停
    - 自动记录每个 token 的 log probability
    """

    def __init__(self, policy_model: nn.Module, tokenizer,
                 eos_token_id: int = 2, pad_token_id: int = 0):
        """初始化 MLX 推理引擎

        Args:
            policy_model: 当前策略模型
            tokenizer: 分词器
            eos_token_id: EOS token ID
            pad_token_id: PAD token ID
        """
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def rollout(self, prompt_ids: mx.array, num_generations: int = 1,
                max_new_tokens: int = 256, temperature: float = 0.8) -> RolloutResult:
        """批量自回归采样生成

        流程：
        1. 将 prompt 复制 G 份
        2. 逐 token 采样（带温度）
        3. 全局 EOS 检测（所有序列结束则停止）
        4. 收集 log probs 和生成结果
        """
        B, P = prompt_ids.shape
        G = num_generations
        total = B * G

        # 复制 prompt
        expanded_ids = mx.repeat(prompt_ids, G, axis=0)  # [B*G, P]
        generated = expanded_ids

        all_logps = []
        finished = mx.zeros((total,), dtype=mx.bool_)  # EOS 终止追踪

        for _ in range(max_new_tokens):
            output = self.policy_model(generated)
            next_logits = output.logits[:, -1, :] / temperature

            # log_softmax
            log_probs = next_logits - mx.logsumexp(next_logits, axis=-1, keepdims=True)

            # 采样
            next_token = mx.random.categorical(log_probs)[:, None]  # [B*G, 1]

            # 记录该 token 的 log prob
            token_logp = mx.take_along_axis(log_probs, next_token, axis=-1)
            # 已结束的序列 logp 置 0
            token_logp = mx.where(finished[:, None], mx.zeros_like(token_logp), token_logp)
            all_logps.append(token_logp)

            # 拼接
            generated = mx.concatenate([generated, next_token], axis=-1)
            mx.eval(generated)

            # 更新终止状态
            finished = finished | (next_token.reshape(-1) == self.eos_token_id)
            if mx.all(finished).item():
                break

        # 收集结果
        completion_ids = generated[:, P:]  # [B*G, R]
        per_token_logps = mx.concatenate(all_logps, axis=-1) if all_logps else mx.zeros((total, 0))

        # 解码文本
        completions = []
        for i in range(total):
            ids = completion_ids[i].tolist()
            # 截取到第一个 EOS
            if self.eos_token_id in ids:
                ids = ids[:ids.index(self.eos_token_id)]
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            completions.append(text)

        return RolloutResult(
            output_ids=generated,
            completion_ids=completion_ids,
            per_token_logps=per_token_logps,
            completions=completions,
        )

    def update_policy(self, model: nn.Module):
        """零拷贝替换策略模型引用（MLX 统一内存下无需特殊处理）"""
        self.policy_model = model


# =====================================================================================
# 工厂函数
# =====================================================================================

def create_rollout_engine(policy_model: nn.Module, tokenizer,
                          eos_token_id: int = 2,
                          pad_token_id: int = 0) -> RolloutEngine:
    """创建 MLX Rollout 引擎

    Args:
        policy_model: 策略模型
        tokenizer: 分词器
        eos_token_id: EOS token ID
        pad_token_id: PAD token ID

    Returns:
        MLXRolloutEngine 实例
    """
    return MLXRolloutEngine(
        policy_model=policy_model,
        tokenizer=tokenizer,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
