# -*- coding: utf-8 -*-
"""
Rollout Engine - 可插拔的推理引擎
==============================================
本模块实现了可插拔的推理引擎，支持多种后端：
- PyTorch 原生推理引擎（TorchRolloutEngine）
- SGLang HTTP API 推理引擎（SGLangRolloutEngine）

功能特点：
- 统一的 rollout 接口，支持批量生成
- 自动计算每个 token 的 logprob
- 支持动态更新策略模型权重
- 支持流式输出和非流式输出

使用示例：
```python
# SGLang 服务器启动命令
python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998
```

作者：MiniMind Team
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer


# =====================================================================================
# 核心工具函数
# =====================================================================================

def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
    """计算序列尾部 ``n_keep`` 个 token 的对数概率（log probability）。

    本函数是 RL 训练的核心组件，用于：
    - 计算策略梯度中的 π(a|s) 项；
    - 计算新旧策略之间的 KL 散度（GRPO / PPO）；
    - 为 importance sampling ratio 提供分子分母。

    实现细节：
    1. 自动解包 DDP 包装，直接在底层模型上调用 forward。
    2. 利用 ``logits_to_keep`` 参数让模型只输出最后 ``n_keep + 1`` 个位置的 logits，
       节省显存（MiniMind 模型支持该优化参数）。
    3. 对每一行用 ``torch.gather`` 从 log_softmax 分布中取出对应 token 的 logp，
       避免物化完整的 softmax 概率矩阵。

    Args:
        model: 语言模型，可以是 DDP 包装过的。
        input_ids: 完整输入序列（prompt + completion），形状 ``[batch_size, seq_len]``。
        n_keep: 需要保留 logp 的尾部 token 数量（通常等于 completion 长度）。
        attention_mask: 注意力掩码（可选），用于处理左侧 padding。

    Returns:
        形状为 ``[batch_size, n_keep]`` 的 logp 张量；若 ``n_keep <= 0`` 则返回空张量。
    """
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)

    # 解包 DDP，获取底层模型
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model

    # inference mode 的 tensor 不允许梯度操作，需要 detach+clone
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids

    # forward 只保留尾部 logits，节省显存：logits shape = [B, n_keep+1, vocab]
    # 取 [:-1] 是因为 logits[t] 预测 input_ids[t+1]，对齐后正好覆盖最后 n_keep 个 token
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]

    # 逐行 gather：从 log_softmax 分布中取出实际 token 对应的 logp
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)


# =====================================================================================
# 数据结构
# =====================================================================================

@dataclass
class RolloutResult:
    """Rollout 引擎的生成结果，封装了 RL 训练所需的全部信息。

    在一次 rollout 中，引擎对每个 prompt 生成 ``num_generations`` 条候选回答，
    本结构将所有候选展平为一维 batch（大小 = batch_size × num_generations）。

    Attributes:
        output_ids: 完整序列（prompt + completion），形状 ``[B*G, P+R]``。
            G = num_generations, P = prompt_len, R = max_response_len。
        completion_ids: 仅 completion 部分，形状 ``[B*G, R]``；
            可直接与 ``per_token_logps`` 一一对应。
        per_token_logps: 对应 ``completion_ids`` 中每个 token 的 logp，
            形状 ``[B*G, R]``；用于计算 importance sampling ratio。
        completions: 解码后的纯文本列表，长度 ``B*G``；
            用于传给 reward model 或规则奖励函数评分。
    """
    output_ids: Tensor
    completion_ids: Tensor
    per_token_logps: Tensor
    completions: List[str]


# =====================================================================================
# 抽象基类
# =====================================================================================

class RolloutEngine(ABC):
    """Rollout 引擎抽象基类，定义了 RL 训练中"生成 + 权重同步"的统一接口。

    所有后端（本地 PyTorch、远程 SGLang 等）均继承此类并实现以下两个方法：
    - ``rollout``: 给定 prompt 批量生成候选回答；
    - ``update_policy``: 在训练步结束后将最新策略权重同步到推理引擎。

    属性：
        tokenizer: 分词器实例；Torch 引擎由外部传入，SGLang 引擎自行加载。
    """
    tokenizer = None

    @abstractmethod
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int,
                temperature: float = 0.8) -> RolloutResult:
        """执行批量 rollout（采样生成）。

        Args:
            prompt_ids: 左填充对齐后的 prompt token IDs，形状 ``[B, P]``。
            attention_mask: 与 prompt_ids 等形状的掩码（1=有效, 0=padding）。
            num_generations: 每个 prompt 采样的候选回答数量 G。
            max_new_tokens: 单条回答的最大生成 token 数。
            temperature: 采样温度，越高越随机。

        Returns:
            ``RolloutResult`` 实例。
        """

    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        """将最新的训练模型权重同步到推理引擎。

        Torch 引擎：直接替换内部模型引用（零拷贝）。
        SGLang 引擎：通过共享磁盘路径 + HTTP API 触发远程权重重载。

        Args:
            model: 最新的策略模型（可能被 DDP / torch.compile 包装）。
        """


# =====================================================================================
# PyTorch 原生推理引擎
# =====================================================================================

class TorchRolloutEngine(RolloutEngine):
    """PyTorch 原生推理引擎——直接在本地 GPU 上用 ``model.generate`` 采样。

    适用场景：
    - 小规模模型（≤1B），单卡即可完成 rollout；
    - 本地调试和快速迭代（无需启动外部服务）；
    - 对延迟不敏感的场景。

    权重同步方式：直接替换内部模型引用（``update_policy``），零拷贝零延迟。
    """

    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):
        """初始化 Torch 推理引擎。

        Args:
            policy_model: 当前策略模型（可被 DDP 包装）。
            tokenizer: 分词器，需含 pad_token_id 和 eos_token_id。
            device: 运行设备标识。
            autocast_ctx: 混合精度上下文（如 torch.cuda.amp.autocast）；
                若为 None 则以 fp32 推理。
        """
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int,
                temperature: float = 0.8) -> RolloutResult:
        # 解包 DDP，generate 需要底层模型
        model = self.policy_model.module if isinstance(self.policy_model,
                                                       DistributedDataParallel) else self.policy_model

        # ---- 阶段 1：无梯度采样生成 ----
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=num_generations,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )  # shape: [B * num_generations, prompt_len + response_len]

        # ---- 阶段 2：切分 completion 并计算 logp ----
        prompt_len = prompt_ids.size(1)
        completion_ids = output_ids[:, prompt_len:]  # [B*G, R]

        from contextlib import nullcontext
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
        with ctx:
            # 在混合精度下计算 logp，与训练时的精度对齐
            per_token_logps = compute_per_token_logps(self.policy_model, output_ids, completion_ids.size(1))

        # ---- 阶段 3：解码文本（供 reward model 评分） ----
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(output_ids, completion_ids, per_token_logps, completions)

    def update_policy(self, model: torch.nn.Module):
        """零拷贝替换策略模型引用——训练更新后立即对后续 rollout 生效。"""
        self.policy_model = model


# =====================================================================================
# SGLang HTTP API 推理引擎
# =====================================================================================

class SGLangRolloutEngine(RolloutEngine):
    """SGLang HTTP API 推理引擎——通过 HTTP 与外部 SGLang 服务器通信。

    适用场景：
    - 大规模模型，推理需要独立 GPU 池化部署；
    - 需要高吞吐、连续 batching、PagedAttention 等高级优化；
    - 训练节点与推理节点分离的分布式架构。

    权重同步方式（``update_policy``）：
    1. 训练端将最新权重写入共享磁盘路径 ``shared_ckpt_path``；
    2. 向 SGLang 服务器发送 ``/update_weights_from_disk`` HTTP POST；
    3. 服务器热加载新权重并刷新 KV Cache。

    前提条件：
    需要先启动 SGLang 推理服务器：
    ``python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998``
    """

    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
        """初始化 SGLang 引擎。

        Args:
            base_url: SGLang 服务器地址，如 ``http://localhost:8998``。
            model_path: 用于加载 tokenizer 的本地模型路径。
            shared_ckpt_path: 训练端与推理端共享的磁盘路径，用于传递新权重。
            timeout: 单次 HTTP 请求超时秒数（生成长文本时可能需要较大值）。
        """
        self.base_url = base_url.rstrip('/')
        self.shared_ckpt_path = shared_ckpt_path
        self.timeout = timeout
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.http = requests

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int,
                temperature: float = 0.8) -> RolloutResult:
        # ---- 阶段 1：准备请求 ----
        # prompt_ids 是左 padding 对齐的，需要用 attention_mask 去除 pad token
        input_ids_list = []
        for ids, mask in zip(prompt_ids, attention_mask):
            valid_ids = ids[mask.bool()].tolist()
            input_ids_list.append(valid_ids)
        # 每个 prompt 复制 G 份（SGLang 不支持 num_return_sequences，需要手动展开）
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]

        payload = {
            "input_ids": all_input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],
            },
            "return_logprob": True,  # 请求服务端返回每个 token 的 logp
        }

        # ---- 阶段 2：发送 HTTP 请求 ----
        resp = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()

        # ---- 阶段 3：解析响应 ----
        results = resp.json()
        if not isinstance(results, list):
            results = [results]

        all_output_ids, all_completion_ids, all_logprobs = [], [], []
        completions = []
        prompt_len = prompt_ids.size(1)

        for i, result in enumerate(results):
            # SGLang 返回格式：meta_info 中包含 output_ids 和 output_token_logprobs
            meta = result.get("meta_info", {})
            completion_ids = meta.get("output_ids", result.get("output_ids", []))
            raw_logprobs = meta.get("output_token_logprobs", [])

            # logprobs 可能是 [logp] 的列表或直接的数值，统一提取为 float 列表
            logprobs = []
            for item in raw_logprobs:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    logprobs.append(item[0])
                elif isinstance(item, (int, float)):
                    logprobs.append(item)

            # 拼接完整序列：prompt + completion（与 TorchRolloutEngine 保持一致的输出格式）
            prompt = all_input_ids[i]
            full_output = prompt + completion_ids
            all_output_ids.append(full_output)
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs)
            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))

        # ---- 阶段 4：右 padding 对齐并转为 Tensor ----
        # 不同样本的 completion 长度不同，统一右填充到 max_len 后 stack 为 batch
        device = prompt_ids.device
        max_out_len = max(len(ids) for ids in all_output_ids)
        max_comp_len = max(len(ids) for ids in all_completion_ids)
        max_logp_len = max(len(lp) for lp in all_logprobs)

        def pad_to_tensor(seqs, max_len, pad_val=0):
            """将变长列表右填充并转为张量。"""
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs], device=device)

        return RolloutResult(
            output_ids=pad_to_tensor(all_output_ids, max_out_len),
            completion_ids=pad_to_tensor(all_completion_ids, max_comp_len),
            per_token_logps=pad_to_tensor(all_logprobs, max_logp_len, pad_val=0.0),
            completions=completions,
        )

    def update_policy(self, model: torch.nn.Module):
        """将训练最新权重落盘并通知 SGLang 服务器热加载。

        流程：
        1. 解包 DDP，获取底层模型。
        2. 断开 lm_head / embed_tokens 的权重共享（clone），避免 save 时修改训练中的参数。
        3. 将 state_dict 以 fp16 落盘到 ``shared_ckpt_path``。
        4. 恢复权重共享（tie weights）。
        5. 向 SGLang 发送 ``/update_weights_from_disk`` 请求触发重载。
        """
        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        abs_path = os.path.abspath(self.shared_ckpt_path)

        # 断开权重共享：lm_head.weight 和 embed_tokens.weight 通常是同一 tensor，
        # 这里 clone 一份独立副本，避免 half() 修改影响训练进程中的 fp32 参数
        unwrapped.lm_head.weight = torch.nn.Parameter(unwrapped.lm_head.weight.clone())

        # 以 fp16 保存到共享磁盘（节省空间 + 加快传输）
        state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
        unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)

        # 恢复权重共享（tie weights）：让 embed_tokens 和 lm_head 再次指向同一 tensor
        unwrapped.model.embed_tokens.weight = unwrapped.lm_head.weight
        self.tokenizer.save_pretrained(abs_path)

        # 通知 SGLang 从磁盘加载新权重
        resp = self.http.post(
            f"{self.base_url}/update_weights_from_disk",
            json={"model_path": abs_path},
            timeout=self.timeout
        )
        if resp.status_code != 200:
            print(f"[SGLANG WARNING] update_weights 失败: {resp.status_code}, {resp.text}")
        return resp.status_code == 200

    def flush_cache(self) -> bool:
        """清空 SGLang 服务器的 KV Cache（权重更新后通常需要调用）。"""
        resp = self.http.post(f"{self.base_url}/flush_cache", timeout=30)
        return resp.status_code == 200

    def health(self) -> bool:
        """探活检查：SGLang 服务器是否正常运行。"""
        try:
            resp = self.http.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


# =====================================================================================
# 工厂函数
# =====================================================================================

def create_rollout_engine(
        engine_type: str = "torch",
        policy_model: torch.nn.Module = None,
        tokenizer=None,
        device: str = "cuda",
        autocast_ctx=None,
        sglang_base_url: str = None,
        sglang_model_path: str = None,
        sglang_shared_path: str = None,
) -> RolloutEngine:
    """创建 Rollout 引擎的工厂函数

    根据配置创建对应的推理引擎实例，支持 PyTorch 原生推理和 SGLang HTTP API 两种方式。

    Args:
        engine_type: 引擎类型，可选 "torch" 或 "sglang"
        policy_model: 策略模型（torch 引擎需要）
        tokenizer: 分词器（torch 引擎需要）
        device: 运行设备
        autocast_ctx: 自动混合精度上下文
        sglang_base_url: SGLang 服务器 URL（sglang 引擎需要）
        sglang_model_path: SGLang 模型路径（sglang 引擎需要）
        sglang_shared_path: SGLang 共享存储路径（sglang 引擎需要）

    Returns:
        RolloutEngine 实例
    """
    if engine_type == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    elif engine_type == "sglang":
        return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")
