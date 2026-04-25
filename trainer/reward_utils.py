# -*- coding: utf-8 -*-
"""
RL 训练通用奖励工具
================================================
本模块抽取了 ``train_grpo.py`` / ``train_ppo.py`` / ``train_agent.py`` 中
重复出现的奖励计算与文本评估工具，供所有强化学习训练脚本复用。

主要功能：
- ``compute_repetition_penalty``: 基于 n-gram 统计的重复惩罚。
- ``score_response_format``: 思考标签 / 长度等基础格式分。
- ``compute_basic_rewards``: 把上述组件组合为单条 (prompt, response) 的基础奖励。
- ``compute_batch_rewards``: 批量对 ``(prompts, responses)`` 计算综合奖励，
  自动调用 reward model 并合并基础格式分。

设计原则：
- 与具体训练算法无关，只接受张量 / 字符串作为输入。
- 不强依赖 ``args``，所有可调参数显式作为函数参数传入，便于单元测试。
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

import torch

# ---- ChatML prompt 解析 ----
# 该函数曾位于 ``trainer.trainer_utils``，为了让 ``reward_utils`` 自包含、
# 避免与 ``trainer_utils`` 形成"工具 ↔ 工具"互引依赖，这里直接定义于此。
_CHAT_MESSAGE_PATTERN = re.compile(
    r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>",
    re.DOTALL,
)

def parse_messages_from_chat_prompt(prompt: str) -> list:
    """从 ChatML 格式的 prompt 字符串中解析 ``[{role, content}, ...]`` 列表。"""
    return [{"role": role, "content": content.strip()}
            for role, content in _CHAT_MESSAGE_PATTERN.findall(prompt)]

# ----- 阈值常量：单独提取出来便于调参与测试 -----
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


def compute_repetition_penalty(text: str, n: int = DEFAULT_NGRAM,
                               cap: float = DEFAULT_REPETITION_CAP) -> float:
    """基于 n-gram 重复率的惩罚分数。

    将文本切分为词级 token，统计 n-gram 中重复出现的比例，并截断到 ``cap``。
    该函数曾以 ``rep_penalty`` 之名重复出现在 GRPO/PPO/Agent 三个训练脚本中，
    本次重构统一抽取，避免行为漂移。

    Args:
        text: 待评估文本。
        n: n-gram 的窗口大小，默认 3。
        cap: 惩罚上限。

    Returns:
        ``[0, cap]`` 之间的浮点惩罚值；当无法构造 n-gram 时返回 0。
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
    """长度奖励：在 ``[min_len, max_len]`` 内得正分，否则得负分。"""
    return LENGTH_REWARD if min_len <= len(response.strip()) <= max_len else LENGTH_PENALTY


def _think_section_score(response: str) -> tuple[float, str]:
    """对思考标签进行格式与长度评分，并返回去掉思考块后的回答。

    评分规则：
    1. 出现 ``</think>`` 时检查思考内容长度是否落在合理区间（20~300）。
    2. ``</think>`` 必须只出现一次，否则给予小幅扣分。

    Args:
        response: 原始回答字符串。

    Returns:
        ``(score, answer)``：
            - score: 思考块带来的总分（含正负）。
            - answer: 去掉思考块后的纯回答；若不含思考标签则返回原 response。
    """
    if THINK_END_TAG not in response:
        return 0.0, response

    thinking_content, answer_content = response.split(THINK_END_TAG, 1)
    score = (THINK_REWARD if THINK_CONTENT_MIN <= len(thinking_content.strip()) <= THINK_CONTENT_MAX
             else THINK_PENALTY)
    score += (THINK_FORMAT_REWARD if response.count(THINK_END_TAG) == 1
              else THINK_FORMAT_PENALTY)
    return score, answer_content.strip()


def compute_basic_rewards(prompt: str, response: str) -> tuple[float, str, list]:
    """计算单条样本的"格式 / 长度 / 重复"基础奖励。

    本函数只产出与文本结构相关的奖励，不调用任何外部模型。
    Reward Model 评分由调用方另行获取并相加。

    Args:
        prompt: ChatML 格式的 prompt 字符串。
        response: 待评分的回答字符串。

    Returns:
        ``(format_reward, answer_clean, messages)``：
            - format_reward: 长度 + 思考 + 重复惩罚之和。
            - answer_clean: 去除思考块后的回答（用于 reward model 评分时更聚焦）。
            - messages: 从 prompt 中解析出的对话历史。
    """
    score = _length_score(response)
    think_score, answer_clean = _think_section_score(response)
    score += think_score
    score -= compute_repetition_penalty(answer_clean)
    messages = parse_messages_from_chat_prompt(prompt)
    return score, answer_clean, messages

# 为保持对旧 import 路径的兼容，将该函数也作为 ``trainer.trainer_utils.parse_messages_from_chat_prompt``
# 的等价实现导出。后续如果还有外部模块从 trainer_utils 导入它，行为完全一致。


def compute_batch_rewards(prompts: Sequence[str], responses: Sequence[str],
                          reward_model, *, num_generations: int = 1,
                          device: Optional[str] = None) -> torch.Tensor:
    """批量计算综合奖励：``基础格式分 + Reward Model 分``。

    支持两种调用约定：
    - ``num_generations == 1``：``prompts`` 与 ``responses`` 长度一致，一一对应。
    - ``num_generations > 1``：每条 prompt 对应连续 ``num_generations`` 个回答，
      因此 ``len(responses) == len(prompts) * num_generations``。

    Args:
        prompts: 提示词列表（去重后），长度为 ``B``。
        responses: 模型生成的回答列表，长度为 ``B * num_generations``。
        reward_model: 拥有 ``get_score(messages, answer) -> float`` 接口的奖励模型。
        num_generations: 每个 prompt 生成的样本数。
        device: 结果张量所在的设备；为 None 时使用 CPU。

    Returns:
        形状为 ``[B * num_generations]`` 的奖励张量。
    """
    target_device = device if device is not None else "cpu"
    rewards = torch.zeros(len(responses), device=target_device)
    rm_scores: List[float] = []

    with torch.no_grad():
        for prompt_idx, prompt in enumerate(prompts):
            for gen_idx in range(num_generations):
                response_idx = prompt_idx * num_generations + gen_idx
                response = responses[response_idx]

                format_reward, answer_clean, messages = compute_basic_rewards(prompt, response)
                rewards[response_idx] += format_reward

                rm_scores.append(reward_model.get_score(messages, answer_clean))

        rewards += torch.tensor(rm_scores, device=target_device)

    return rewards
