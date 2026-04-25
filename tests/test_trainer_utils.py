# -*- coding: utf-8 -*-
"""trainer 模块通用工具的单元测试。

本测试聚焦于纯函数 / 纯数据结构，避免依赖真实的模型权重、CUDA、分布式环境，
方便在任何机器（包括 CPU-only 的 CI）上快速回归。

覆盖范围：
- ``trainer.reward_utils.compute_repetition_penalty`` 的边界与典型情况
- ``trainer.reward_utils.compute_basic_rewards`` 的格式分逻辑
- ``trainer.trainer_utils.get_lr`` 余弦学习率调度
- ``trainer.trainer_utils.format_duration`` 时长格式化
- ``trainer.trainer_utils.SkipBatchSampler`` 断点续训采样器
- ``trainer.trainer_utils.get_device_type`` 设备类型识别
"""

from __future__ import annotations

import math
import os
import sys
import unittest

# 让本测试可以独立从仓库根目录运行：python -m unittest tests/test_trainer_utils.py
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from trainer.reward_utils import (  # noqa: E402  延迟到 sys.path 调整后再导入
    compute_basic_rewards,
    compute_repetition_penalty,
)
from trainer.trainer_utils import (  # noqa: E402
    SkipBatchSampler,
    format_duration,
    get_device_type,
    get_lr,
)


class ComputeRepetitionPenaltyTests(unittest.TestCase):
    """``compute_repetition_penalty`` 的边界与典型情况。"""

    def test_empty_text_returns_zero(self):
        """空文本应返回 0，避免除零。"""
        self.assertEqual(compute_repetition_penalty(""), 0.0)

    def test_short_text_no_ngram_returns_zero(self):
        """文本长度不足以形成 n-gram 时（例如 2 个 token，n=3）应返回 0。"""
        self.assertEqual(compute_repetition_penalty("hello world", n=3), 0.0)

    def test_no_repetition_returns_zero(self):
        """完全无重复 n-gram 时应返回 0。"""
        text = "the quick brown fox jumps over a lazy dog"
        self.assertAlmostEqual(compute_repetition_penalty(text, n=3), 0.0)

    def test_full_repetition_returns_cap(self):
        """高度重复的文本不应超过 cap。"""
        text = "abc abc abc abc abc abc abc abc"
        penalty = compute_repetition_penalty(text, n=3, cap=0.5)
        self.assertLessEqual(penalty, 0.5)
        self.assertGreater(penalty, 0.0)

    def test_cap_is_respected(self):
        """传入更小的 cap，结果上限应跟随调整。"""
        text = "x x x x x x x x x x"
        penalty = compute_repetition_penalty(text, n=2, cap=0.1)
        self.assertLessEqual(penalty, 0.1)


class ComputeBasicRewardsTests(unittest.TestCase):
    """``compute_basic_rewards`` 的奖励项组合验证。"""

    def test_normal_response_gets_length_bonus(self):
        """长度合适且无 n-gram 重复的回答应获得正向长度奖励。"""
        # 注意：避免使用 ``"xxx" * N`` 这类构造高重复 n-gram 的样本，
        # 否则 compute_repetition_penalty 会抵消掉 LENGTH_REWARD。
        response = (
            "Spring frames green willows by the river bank, while distant "
            "mountains slowly fade into faint clouds at dusk every evening."
        )
        reward, answer, _ = compute_basic_rewards(
            prompt="<|im_start|>user\nhi<|im_end|>",
            response=response,
        )
        self.assertGreater(reward, 0.0)
        self.assertEqual(answer, response)

    def test_too_short_response_is_penalised(self):
        """过短回答应被扣长度分。"""
        reward, _, _ = compute_basic_rewards(prompt="x", response="hi")
        self.assertLess(reward, 0.0)

    def test_thinking_block_is_extracted(self):
        """``</think>`` 之后的部分才是 answer，``answer`` 应只含答案部分。"""
        thinking = "let me think " * 5
        answer_text = "final answer" * 5
        response = f"{thinking}</think>{answer_text}"
        reward, answer, _ = compute_basic_rewards(prompt="x", response=response)
        self.assertEqual(answer, answer_text)
        # 包含 <think> 标签且数量为 1 时，应有正向加分
        self.assertGreater(reward, 0.0)


class GetLrTests(unittest.TestCase):
    """余弦学习率调度的单调性与边界。"""

    def test_at_step_zero_close_to_max(self):
        """step=0 时学习率应等于初始 lr（公式：lr*(0.1 + 0.45*(1+cos(0))) = lr*1.0）。"""
        lr = get_lr(current_step=0, total_steps=1000, lr=1e-3)
        self.assertAlmostEqual(lr, 1e-3, places=6)

    def test_at_final_step_close_to_min(self):
        """step=total_steps 时应接近最小值（lr/10）。"""
        lr = get_lr(current_step=1000, total_steps=1000, lr=1e-3)
        self.assertAlmostEqual(lr, 1e-3 / 10, places=6)

    def test_monotonic_decrease(self):
        """学习率应单调递减。"""
        prev = math.inf
        for step in range(0, 1000, 100):
            cur = get_lr(step, 1000, 1e-3)
            self.assertLess(cur, prev)
            prev = cur


class FormatDurationTests(unittest.TestCase):
    """``format_duration`` 的格式化分支。"""

    def test_seconds_branch(self):
        self.assertEqual(format_duration(30), "30s")

    def test_minutes_branch(self):
        # 5.5 分钟
        self.assertEqual(format_duration(330), "5.5min")

    def test_hours_branch(self):
        # 2 小时 30 分钟
        self.assertEqual(format_duration(2 * 3600 + 30 * 60), "2h30m")


class SkipBatchSamplerTests(unittest.TestCase):
    """断点续训用的批采样器。"""

    def test_yields_correct_number_of_batches(self):
        """无跳过时应产出 ⌈len/batch⌉ 个批次。"""
        sampler = SkipBatchSampler(list(range(10)), batch_size=3, skip_batches=0)
        batches = list(sampler)
        # 10 个样本，batch_size=3 -> 4 个批（最后一个长度为 1）
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[0], [0, 1, 2])
        self.assertEqual(batches[-1], [9])

    def test_skip_batches_reduces_count(self):
        """跳过 N 个批次后应少返回 N 个批次。"""
        sampler = SkipBatchSampler(list(range(10)), batch_size=3, skip_batches=2)
        batches = list(sampler)
        self.assertEqual(len(batches), 2)  # 4 - 2 = 2
        self.assertEqual(batches[0], [6, 7, 8])

    def test_len_reflects_skipped(self):
        sampler = SkipBatchSampler(list(range(10)), batch_size=3, skip_batches=1)
        self.assertEqual(len(sampler), 3)


class GetDeviceTypeTests(unittest.TestCase):
    """根据设备字符串识别后端类型。"""

    def test_cuda(self):
        self.assertEqual(get_device_type("cuda"), "cuda")
        self.assertEqual(get_device_type("cuda:0"), "cuda")

    def test_mps(self):
        self.assertEqual(get_device_type("mps"), "mps")

    def test_cpu(self):
        self.assertEqual(get_device_type("cpu"), "cpu")


if __name__ == "__main__":
    unittest.main()
