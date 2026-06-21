# -*- coding: utf-8 -*-
"""
MLX RL 训练通用奖励工具
=============================================
本模块为所有强化学习训练脚本提供奖励计算与文本评估工具：
- compute_repetition_penalty: 基于 n-gram 统计的重复惩罚
- score_response_format: 思考标签 / 长度等基础格式分
- compute_basic_rewards: 单条 (prompt, response) 的基础奖励
- compute_batch_rewards: 批量计算综合奖励

设计原则：
- 与具体训练算法无关，只接受字符串作为输入
- 所有可调参数显式作为函数参数传入
- 纯 Python 计算，不依赖 MLX 张量（奖励计算无需加速）
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

import mlx.core as mx


# =====================================================================================
# ChatML Prompt 解析
# =====================================================================================

_CHAT_MESSAGE_PATTERN = re.compile(
    r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>",
    re.DOTALL,
)


def parse_messages_from_chat_prompt(prompt: str) -> list:
    """从 ChatML 格式的 prompt 字符串中解析 [{role, content}, ...] 列表"""
    return [
        {"role": role, "content": content.strip()}
        for role, content in _CHAT_MESSAGE_PATTERN.findall(prompt)
    ]


# =====================================================================================
# 阈值常量
# =====================================================================================

DEFAULT_NGRAM = 3
DEFAULT_REPETITION_CAP = 0.5

DEFAULT_LENGTH_MIN = 20
DEFAULT_LENGTH_MAX = 800
LENGTH_REWARD = 0.5
LENGTH_PENALTY = -0.5

THINK_CONTENT_MIN = 20
THINK_CONTENT_MAX = 300
THINK_REWARD = 1.0
THINK_PENALTY = -0.5

THINK_FORMAT_REWARD = 0.25
THINK_FORMAT_PENALTY = -0.25

THINK_END_TAG = "</think>"


# =====================================================================================
# 核心奖励函数
# =====================================================================================

def compute_repetition_penalty(text: str, n: int = DEFAULT_NGRAM,
                               cap: float = DEFAULT_REPETITION_CAP) -> float:
    """基于 n-gram 重复率的惩罚分数

    将文本切分为词级 token，统计 n-gram 中重复出现的比例。

    Args:
        text: 待评估文本
        n: n-gram 窗口大小，默认 3
        cap: 惩罚上限

    Returns:
        [0, cap] 之间的浮点惩罚值
    """
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    duplicate_ratio = (len(grams) - len(set(grams))) * cap * 2 / len(grams)
    return min(cap, duplicate_ratio)


def _length_score(response: str,
                  min_len: int = DEFAULT_LENGTH_MIN,
                  max_len: int = DEFAULT_LENGTH_MAX) -> float:
    """长度奖励：在 [min_len, max_len] 内得正分，否则得负分"""
    return LENGTH_REWARD if min_len <= len(response.strip()) <= max_len else LENGTH_PENALTY


def _think_section_score(response: str) -> tuple:
    """对思考标签进行格式与长度评分

    Returns:
        (score, answer): 思考块总分 + 去掉思考块后的纯回答
    """
    if THINK_END_TAG not in response:
        return 0.0, response

    thinking_content, answer_content = response.split(THINK_END_TAG, 1)
    score = (THINK_REWARD if THINK_CONTENT_MIN <= len(thinking_content.strip()) <= THINK_CONTENT_MAX
             else THINK_PENALTY)
    score += (THINK_FORMAT_REWARD if response.count(THINK_END_TAG) == 1
              else THINK_FORMAT_PENALTY)
    return score, answer_content.strip()


def compute_basic_rewards(prompt: str, response: str) -> tuple:
    """计算单条样本的"格式 / 长度 / 重复"基础奖励

    Returns:
        (format_reward, answer_clean, messages)
    """
    score = _length_score(response)
    think_score, answer_clean = _think_section_score(response)
    score += think_score
    score -= compute_repetition_penalty(answer_clean)
    messages = parse_messages_from_chat_prompt(prompt)
    return score, answer_clean, messages


def compute_batch_rewards(prompts: Sequence[str], responses: Sequence[str],
                          reward_model=None, *, num_generations: int = 1) -> mx.array:
    """批量计算综合奖励：基础格式分 + Reward Model 分

    Args:
        prompts: 提示词列表（去重后），长度 B
        responses: 回答列表，长度 B * num_generations
        reward_model: 具有 get_score(messages, answer) -> float 接口的奖励模型
        num_generations: 每个 prompt 生成的样本数

    Returns:
        形状为 [B * num_generations] 的奖励数组
    """
    rewards = [0.0] * len(responses)

    for prompt_idx, prompt in enumerate(prompts):
        for gen_idx in range(num_generations):
            response_idx = prompt_idx * num_generations + gen_idx
            response = responses[response_idx]

            format_reward, answer_clean, messages = compute_basic_rewards(prompt, response)
            rewards[response_idx] += format_reward

            if reward_model is not None:
                rm_score = reward_model.get_score(messages, answer_clean)
                rewards[response_idx] += rm_score

    return mx.array(rewards)
