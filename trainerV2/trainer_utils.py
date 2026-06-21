# -*- coding: utf-8 -*-
"""
MLX 训练工具函数集合

本模块提供 MLX 框架下模型训练所需的工具函数和类：
- 学习率调度（余弦退火）
- 模型初始化与权重加载
- 检查点管理（保存/恢复）
- 梯度裁剪
- 日志与进度展示

设计理念：
- MLX 使用统一内存，无需设备管理
- 惰性求值天然支持计算图融合，无需 autocast
- 函数式梯度计算取代命令式 backward()
"""
import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np
from transformers import AutoTokenizer


# =====================================================================================
# 日志与格式化工具
# =====================================================================================

def logger(content: str):
    """打印带时间戳的日志"""
    print(f"[{time.strftime('%H:%M:%S')}] {content}")


def format_duration(seconds: float) -> str:
    """将秒数格式化为人类可读的时间字符串"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h{minutes:02d}m"


def make_progress_bar(current: int, total: int, bar_length: int = 20) -> str:
    """生成文本进度条"""
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    percent = 100.0 * current / total
    return f"|{bar}| {percent:.1f}%"


def get_memory_usage() -> str:
    """获取 Apple Silicon 统一内存使用情况"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        return f"mem: {used_gb:.1f}/{total_gb:.1f}GB"
    except ImportError:
        return ""


# =====================================================================================
# 学习率调度
# =====================================================================================

def get_lr(current_step: int, total_steps: int, lr: float) -> float:
    """余弦退火学习率调度

    学习率从初始值平滑衰减到 10% 的初始值。
    公式：lr_t = lr * (0.1 + 0.45 * (1 + cos(π * t / T)))

    Args:
        current_step: 当前训练步数
        total_steps: 总训练步数
        lr: 初始学习率

    Returns:
        当前步数对应的学习率
    """
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


# =====================================================================================
# 随机种子
# =====================================================================================

def setup_seed(seed: int):
    """设置全局随机种子，确保训练可复现"""
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


# =====================================================================================
# 模型参数统计
# =====================================================================================

def get_model_params(model: nn.Module) -> Tuple[float, float]:
    """统计模型参数量

    Args:
        model: MLX 模型实例

    Returns:
        (总参数量M, 可训练参数量M) 元组
    """
    total = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    trainable = sum(
        p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters())
    )
    return total / 1e6, trainable / 1e6


def log_model_params(model: nn.Module, config=None):
    """打印模型参数量信息"""
    total_m, trainable_m = get_model_params(model)
    logger(f"Model Params: {total_m:.2f}M | Trainable: {trainable_m:.3f}M")


# =====================================================================================
# 梯度裁剪
# =====================================================================================

def clip_grad_norm(grads, max_norm: float = 1.0):
    """全局梯度范数裁剪（MLX 版本）

    计算所有梯度的全局 L2 范数，若超过 max_norm 则等比缩放。
    使用批量计算避免多次 mx.eval() 同步。

    Args:
        grads: 梯度树（与模型参数结构对应）
        max_norm: 最大梯度范数

    Returns:
        裁剪后的梯度树
    """
    flat_grads = nn.utils.tree_flatten(grads)
    # 批量计算所有梯度的平方和（单次求值，避免逐个 .item() 破坏 lazy eval）
    norms = mx.array([mx.sum(g * g) for _, g in flat_grads])
    total_norm_sq = mx.sum(norms)
    # 使用 MLX 原生条件缩放，无需 .item() 同步
    scale = mx.minimum(mx.array(max_norm) / (mx.sqrt(total_norm_sq) + 1e-6), mx.array(1.0))
    grads = mlx.utils.tree_map(lambda g: g * scale, grads)
    return grads


# =====================================================================================
# 检查点管理
# =====================================================================================

def save_checkpoint(model: nn.Module, optimizer,
                    save_dir: str, save_weight: str, *,
                    epoch: int = 0, step: int = 0,
                    config: Optional[Any] = None):
    """保存模型检查点

    保存内容包括：
    1. 模型权重（npz 格式，MLX 原生）
    2. 训练状态（optimizer state、epoch、step）

    Args:
        model: MLX 模型实例
        optimizer: 优化器状态
        save_dir: 保存目录
        save_weight: 权重文件前缀名
        epoch: 当前 epoch
        step: 当前 step
        config: 模型配置对象（可选）
    """
    os.makedirs(save_dir, exist_ok=True)

    # 保存模型权重
    weight_path = os.path.join(save_dir, f"{save_weight}.safetensors")
    nn.utils.save_safetensors(weight_path, dict(nn.utils.tree_flatten(model.parameters())))

    # 保存训练状态
    state_path = os.path.join(save_dir, f"{save_weight}_state.npz")
    state = {
        "epoch": mx.array(epoch),
        "step": mx.array(step),
    }
    mx.savez(state_path, **state)
    logger(f"Checkpoint saved: {weight_path}")


def load_checkpoint(model: nn.Module, save_dir: str, save_weight: str,
                    ) -> Tuple[int, int]:
    """加载模型检查点

    Args:
        model: MLX 模型实例
        save_dir: 检查点目录
        save_weight: 权重文件前缀名

    Returns:
        (start_epoch, start_step) 元组
    """
    weight_path = os.path.join(save_dir, f"{save_weight}.safetensors")
    state_path = os.path.join(save_dir, f"{save_weight}_state.npz")

    if not os.path.exists(weight_path):
        return 0, 0

    # 加载模型权重
    weights = nn.utils.load_safetensors(weight_path)
    model.load_weights(list(weights.items()))
    logger(f"Weights loaded from: {weight_path}")

    # 加载训练状态
    start_epoch, start_step = 0, 0
    if os.path.exists(state_path):
        state = mx.load(state_path)
        start_epoch = int(state["epoch"].item())
        start_step = int(state["step"].item())
        logger(f"Resumed from epoch {start_epoch}, step {start_step}")

    return start_epoch, start_step


# =====================================================================================
# 数据加载工具
# =====================================================================================

@dataclass
class Batch:
    """训练批次数据容器"""
    input_ids: mx.array
    labels: mx.array


def numpy_to_mlx(arr) -> mx.array:
    """将 numpy 数组转为 MLX 数组（零拷贝）"""
    if isinstance(arr, mx.array):
        return arr
    return mx.array(np.asarray(arr))


class DataIterator:
    """高效数据迭代器

    针对 MLX 的统一内存架构设计：
    - 数据直接存在内存中，无需 pin_memory 或设备传输
    - 支持随机打乱和跳步续训

    Args:
        dataset: 数据集（需实现 __len__ 和 __getitem__）
        batch_size: 批次大小
        shuffle: 是否打乱数据
        skip_steps: 跳过的步数（用于续训）
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = True,
                 skip_steps: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.skip_steps = skip_steps

        # 生成索引
        self.indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self.indices)

    def __len__(self) -> int:
        total = (len(self.indices) + self.batch_size - 1) // self.batch_size
        return max(0, total - self.skip_steps)

    def __iter__(self):
        batches_yielded = 0
        batch = []
        for idx in self.indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batches_yielded += 1
                if batches_yielded <= self.skip_steps:
                    batch = []
                    continue
                yield self._collate(batch)
                batch = []
        # 处理最后一个不完整的 batch
        if batch and batches_yielded >= self.skip_steps:
            yield self._collate(batch)

    def _collate(self, indices: List[int]) -> Batch:
        """将样本索引对应的数据收集为一个批次"""
        items = [self.dataset[i] for i in indices]
        input_ids = mx.array(np.stack([item[0] for item in items]))
        labels = mx.array(np.stack([item[1] for item in items]))
        return Batch(input_ids=input_ids, labels=labels)


# =====================================================================================
# ChatML 解析工具
# =====================================================================================

_CHAT_MESSAGE_PATTERN = re.compile(
    r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>",
    re.DOTALL,
)


def parse_messages_from_chat_prompt(prompt: str) -> List[Dict[str, str]]:
    """从 ChatML 格式的 prompt 字符串中解析 messages 列表"""
    return [{"role": role, "content": content.strip()}
            for role, content in _CHAT_MESSAGE_PATTERN.findall(prompt)]


# =====================================================================================
# WandB / SwanLab 日志集成
# =====================================================================================

def setup_wandb(args, run_name_prefix: str = "MiniMind-MLX"):
    """初始化 wandb/swanlab 日志记录器

    Args:
        args: 命令行参数对象
        run_name_prefix: run 名前缀

    Returns:
        wandb 模块对象或 None
    """
    if not getattr(args, "use_wandb", False):
        return None

    try:
        import swanlab as wandb
        epochs = getattr(args, "epochs", "?")
        batch_size = getattr(args, "batch_size", "?")
        lr = getattr(args, "learning_rate", "?")
        run_name = f"{run_name_prefix}-Epoch-{epochs}-BS-{batch_size}-LR-{lr}"
        wandb.init(
            project=getattr(args, "wandb_project", "MiniMind-MLX"),
            name=run_name,
        )
        return wandb
    except ImportError:
        logger("Warning: swanlab not installed, wandb logging disabled")
        return None
