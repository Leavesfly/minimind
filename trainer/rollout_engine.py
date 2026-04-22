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


# ===== 计算每个 token 的 logprob =====
def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
    """计算每个 token 的对数概率（log probability）
    
    用于强化学习训练时计算策略梯度和 KL 散度
    
    Args:
        model: 语言模型
        input_ids: 输入 token IDs，形状为 [batch_size, seq_len]
        n_keep: 保留最后 n_keep 个 token 的 logprob
        attention_mask: 注意力掩码（可选）
    
    Returns:
        每个 token 的 logprob，形状为 [batch_size, n_keep]
    """
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)


# ===== Rollout 结果 =====
@dataclass
class RolloutResult:
    """Rollout 结果数据类
    
    封装了推理引擎生成的所有必要信息，用于后续的强化学习训练
    
    Attributes:
        output_ids: 完整的输出 token IDs（包括 prompt 和 completion）
        completion_ids: 仅 completion 部分的 token IDs
        per_token_logps: 每个 token 的对数概率
        completions: 解码后的文本列表
    """
    output_ids: Tensor
    completion_ids: Tensor
    per_token_logps: Tensor
    completions: List[str]


# ===== Rollout 引擎抽象基类 =====
class RolloutEngine(ABC):
    """Rollout 引擎抽象基类
    
    定义了所有推理引擎必须实现的接口，支持不同后端的可插拔设计
    """
    tokenizer = None

    @abstractmethod
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int,
                temperature: float = 0.8) -> RolloutResult:
        pass

    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        pass


# ===== PyTorch 原生推理引擎 =====
class TorchRolloutEngine(RolloutEngine):
    """PyTorch 原生推理引擎
    
    使用 PyTorch 原生的 generate 方法进行推理，适合小规模模型和本地部署
    
    特点：
    - 直接使用模型权重，无需额外服务
    - 支持 GPU 加速
    - 支持混合精度推理
    """

    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int,
                temperature: float = 0.8) -> RolloutResult:
        model = self.policy_model.module if isinstance(self.policy_model,
                                                       DistributedDataParallel) else self.policy_model

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
            )  # [B*num_gen, P+R]

        prompt_len = prompt_ids.size(1)
        completion_ids = output_ids[:, prompt_len:]  # [B*num_gen, R]

        from contextlib import nullcontext
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
        with ctx:
            per_token_logps = compute_per_token_logps(self.policy_model, output_ids, completion_ids.size(1))

        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(output_ids, completion_ids, per_token_logps, completions)

    def update_policy(self, model: torch.nn.Module):
        self.policy_model = model


# ===== SGLang HTTP API 推理引擎 =====
class SGLangRolloutEngine(RolloutEngine):
    """SGLang HTTP API 推理引擎
    
    通过 HTTP API 与 SGLang 服务器通信，适合大规模模型和服务化部署
    
    特点：
    - 支持批量推理和流式输出
    - 高性能推理引擎
    - 支持动态权重更新
    - 自动处理 KV Cache
    
    注意：
    需要先启动 SGLang 服务器：
    python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998
    """

    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.shared_ckpt_path = shared_ckpt_path
        self.timeout = timeout
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.http = requests

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int,
                temperature: float = 0.8) -> RolloutResult:
        # 去除左侧 padding tokens，只保留有效 token
        input_ids_list = []
        for ids, mask in zip(prompt_ids, attention_mask):
            valid_ids = ids[mask.bool()].tolist()
            input_ids_list.append(valid_ids)
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]

        payload = {
            "input_ids": all_input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],
            },
            "return_logprob": True,
        }

        resp = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()

        results = resp.json()
        if not isinstance(results, list):
            results = [results]

        all_output_ids, all_completion_ids, all_logprobs = [], [], []
        completions = []
        prompt_len = prompt_ids.size(1)

        for i, result in enumerate(results):
            meta = result.get("meta_info", {})
            completion_ids = meta.get("output_ids", result.get("output_ids", []))
            raw_logprobs = meta.get("output_token_logprobs", [])

            logprobs = []
            for item in raw_logprobs:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    logprobs.append(item[0])
                elif isinstance(item, (int, float)):
                    logprobs.append(item)

            prompt = all_input_ids[i]
            full_output = prompt + completion_ids
            all_output_ids.append(full_output)
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs)
            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))

        device = prompt_ids.device
        max_out_len = max(len(ids) for ids in all_output_ids)
        max_comp_len = max(len(ids) for ids in all_completion_ids)
        max_logp_len = max(len(lp) for lp in all_logprobs)

        def pad_to_tensor(seqs, max_len, pad_val=0):
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs], device=device)

        return RolloutResult(
            output_ids=pad_to_tensor(all_output_ids, max_out_len),
            completion_ids=pad_to_tensor(all_completion_ids, max_comp_len),
            per_token_logps=pad_to_tensor(all_logprobs, max_logp_len, pad_val=0.0),
            completions=completions,
        )

    def update_policy(self, model: torch.nn.Module):
        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        abs_path = os.path.abspath(self.shared_ckpt_path)
        unwrapped.lm_head.weight = torch.nn.Parameter(unwrapped.lm_head.weight.clone())
        state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
        unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
        unwrapped.model.embed_tokens.weight = unwrapped.lm_head.weight
        self.tokenizer.save_pretrained(abs_path)
        resp = self.http.post(
            f"{self.base_url}/update_weights_from_disk",
            json={"model_path": abs_path},
            timeout=self.timeout
        )
        if resp.status_code != 200:
            print(f"[SGLANG WARNING] update_weights 失败: {resp.status_code}, {resp.text}")
        return resp.status_code == 200

    def flush_cache(self) -> bool:
        resp = self.http.post(f"{self.base_url}/flush_cache", timeout=30)
        return resp.status_code == 200

    def health(self) -> bool:
        try:
            resp = self.http.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


# ===== 工厂函数 =====
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
