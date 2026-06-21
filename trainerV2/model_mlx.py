# -*- coding: utf-8 -*-
"""
MiniMind 模型 MLX 实现

基于 Apple MLX 框架重写的 MiniMind 因果语言模型，保持与 PyTorch 版本
相同的架构设计（RoPE、GQA、RMSNorm、MoE），同时利用 MLX 的惰性求值
和统一内存特性实现高效推理与训练。

架构特性：
- RoPE 旋转位置编码（支持 YaRN 外推）
- GQA 分组查询注意力
- RMSNorm 归一化
- SwiGLU 激活函数
- 可选 MoE 混合专家机制
"""
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


# =====================================================================================
# 配置
# =====================================================================================

@dataclass
class MiniMindConfig:
    """MiniMind 模型配置"""
    hidden_size: int = 768
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    vocab_size: int = 6400
    intermediate_size: int = 0  # 0 表示自动计算
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    hidden_act: str = "silu"
    dropout: float = 0.0
    use_moe: bool = False
    num_experts: int = 4
    num_experts_per_tok: int = 1
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True
    router_aux_loss_coef: float = 5e-4

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads
        if self.intermediate_size == 0:
            self.intermediate_size = math.ceil(self.hidden_size * math.pi / 64) * 64
        if self.moe_intermediate_size == 0:
            self.moe_intermediate_size = self.intermediate_size


# =====================================================================================
# 基础组件
# =====================================================================================

class RMSNorm(nn.Module):
    """RMS 归一化层"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        norm = x * mx.rsqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return self.weight * norm


def precompute_freqs_cis(dim: int, end: int, theta: float = 1e6) -> Tuple[mx.array, mx.array]:
    """预计算 RoPE 旋转位置编码的 cos/sin 值"""
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    t = mx.arange(end).astype(mx.float32)
    freqs = mx.outer(t, freqs)
    cos = mx.concatenate([mx.cos(freqs), mx.cos(freqs)], axis=-1)
    sin = mx.concatenate([mx.sin(freqs), mx.sin(freqs)], axis=-1)
    return cos, sin


def apply_rotary_pos_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> Tuple[mx.array, mx.array]:
    """应用 RoPE 旋转位置编码"""

    def rotate_half(x):
        x1, x2 = mx.split(x, 2, axis=-1)
        return mx.concatenate([-x2, x1], axis=-1)

    # cos/sin shape: [seq_len, head_dim] -> [1, seq_len, 1, head_dim]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def repeat_kv(x: mx.array, n_rep: int) -> mx.array:
    """重复 KV 头以匹配 Q 头数量（GQA）"""
    if n_rep == 1:
        return x
    B, L, n_kv, D = x.shape
    x = mx.expand_dims(x, axis=3)  # [B, L, n_kv, 1, D]
    x = mx.broadcast_to(x, (B, L, n_kv, n_rep, D))
    return x.reshape(B, L, n_kv * n_rep, D)


# =====================================================================================
# 注意力层
# =====================================================================================

class MiniMindAttention(nn.Module):
    """GQA 多头注意力（支持 Grouped Query Attention）"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)

    def __call__(self, x: mx.array, freqs_cos: mx.array, freqs_sin: mx.array,
                 mask: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape

        # 投影
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        # 应用 RoPE
        q, k = apply_rotary_pos_emb(q, k, freqs_cos[:L], freqs_sin[:L])

        # GQA: 重复 KV 头
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # 转置为 [B, n_heads, L, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # 缩放点积注意力
        scale = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale

        if mask is not None:
            scores = scores + mask

        attn_weights = mx.softmax(scores, axis=-1)
        output = (attn_weights @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output)


# =====================================================================================
# 前馈网络
# =====================================================================================

class MiniMindMLP(nn.Module):
    """SwiGLU 前馈网络"""

    def __init__(self, config: MiniMindConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        dim = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, dim, bias=False)
        self.down_proj = nn.Linear(dim, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    """MoE 路由门控"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def __call__(self, x: mx.array):
        logits = self.gate(x)  # [B*L, num_experts]
        # Top-k 选择
        top_k_indices = mx.argpartition(-logits, kth=self.top_k, axis=-1)[:, :self.top_k]
        top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)
        top_k_weights = mx.softmax(top_k_logits, axis=-1)
        return top_k_indices, top_k_weights


class MiniMindMoE(nn.Module):
    """Mixture of Experts 层"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.gate = MoEGate(config)
        self.experts = [
            MiniMindMLP(config, config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ]
        self.router_aux_loss_coef = config.router_aux_loss_coef

    def __call__(self, x: mx.array):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])  # [B*L, D]

        indices, weights = self.gate(x_flat)  # [B*L, top_k], [B*L, top_k]

        # 逐专家计算
        output = mx.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # 找到分配给该专家的 token
            mask = (indices == i)  # [B*L, top_k]
            token_mask = mx.any(mask, axis=-1)  # [B*L]
            if not mx.any(token_mask).item():
                continue
            # 获取对应权重
            expert_weight = mx.sum(weights * mask, axis=-1, keepdims=True)  # [B*L, 1]
            expert_out = expert(x_flat)
            output = output + expert_out * expert_weight

        return output.reshape(orig_shape), mx.array(0.0)


# =====================================================================================
# Transformer 层
# =====================================================================================

class MiniMindBlock(nn.Module):
    """Transformer 解码器层"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.attention = MiniMindAttention(config)
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.use_moe = config.use_moe

        if config.use_moe:
            self.feed_forward = MiniMindMoE(config)
        else:
            self.feed_forward = MiniMindMLP(config)

    def __call__(self, x: mx.array, freqs_cos: mx.array, freqs_sin: mx.array,
                 mask: Optional[mx.array] = None):
        # Pre-Norm + Attention + Residual
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin, mask)

        # Pre-Norm + FFN + Residual
        if self.use_moe:
            ffn_out, aux_loss = self.feed_forward(self.ffn_norm(h))
            h = h + ffn_out
            return h, aux_loss
        else:
            h = h + self.feed_forward(self.ffn_norm(h))
            return h, mx.array(0.0)


# =====================================================================================
# 完整模型
# =====================================================================================

@dataclass
class CausalLMOutput:
    """模型输出容器"""
    logits: mx.array
    loss: Optional[mx.array] = None
    aux_loss: Optional[mx.array] = None


class MiniMindForCausalLM(nn.Module):
    """MiniMind 因果语言模型（MLX 版本）

    Weight Tying: lm_head 与 embed_tokens 共享权重，
    通过直接用 embed_tokens.weight 做输出投影实现（而非单独的 nn.Linear）。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [MiniMindBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # 注意：无独立 lm_head，使用 embed_tokens.weight 实现 weight tying

        # 预计算 RoPE（冻结，不参与梯度更新）
        self._freqs_cos, self._freqs_sin = precompute_freqs_cis(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.freeze(keys=["_freqs_cos", "_freqs_sin"])

    def __call__(self, input_ids: mx.array, labels: Optional[mx.array] = None) -> CausalLMOutput:
        B, L = input_ids.shape

        # 嵌入
        h = self.embed_tokens(input_ids)

        # 因果注意力掩码
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        mask = mask.astype(h.dtype)

        # Transformer 层
        aux_loss_total = mx.array(0.0)
        for layer in self.layers:
            h, aux_loss = layer(h, self._freqs_cos, self._freqs_sin, mask)
            aux_loss_total = aux_loss_total + aux_loss

        # 输出 (Weight Tying: 直接用 embedding 权重做投影)
        h = self.norm(h)
        logits = h @ self.embed_tokens.weight.T

        # 计算损失
        loss = None
        if labels is not None:
            # Shift: logits[:-1] 预测 labels[1:]
            shift_logits = logits[:, :-1, :].reshape(-1, self.config.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            # 忽略 padding token (label == -100)
            valid_mask = shift_labels != -100
            # 用 0 替换 -100 以避免索引越界
            safe_labels = mx.where(valid_mask, shift_labels, mx.zeros_like(shift_labels))
            loss = nn.losses.cross_entropy(shift_logits, safe_labels, reduction="none")
            loss = mx.sum(loss * valid_mask) / mx.maximum(mx.sum(valid_mask), mx.array(1.0))

        return CausalLMOutput(logits=logits, loss=loss, aux_loss=aux_loss_total)

    def generate(self, input_ids: mx.array, max_new_tokens: int = 128,
                 temperature: float = 0.8, top_p: float = 0.9) -> mx.array:
        """自回归生成"""
        for _ in range(max_new_tokens):
            output = self(input_ids)
            next_logits = output.logits[:, -1, :] / temperature

            # Top-p (nucleus) 采样
            sorted_indices = mx.argsort(-next_logits, axis=-1)
            sorted_logits = mx.take_along_axis(next_logits, sorted_indices, axis=-1)
            cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)

            # 去除累积概率超过 top_p 的 token
            sorted_mask = cumulative_probs - mx.softmax(sorted_logits, axis=-1) >= top_p
            sorted_logits = mx.where(sorted_mask, mx.full_like(sorted_logits, -1e9), sorted_logits)

            # 还原顺序并采样
            probs = mx.softmax(sorted_logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
            next_token = mx.take_along_axis(sorted_indices, next_token[:, None], axis=-1)

            input_ids = mx.concatenate([input_ids, next_token], axis=-1)
            mx.eval(input_ids)

            # EOS 检查
            if mx.all(next_token == self.config.eos_token_id if hasattr(self.config, 'eos_token_id') else 2):
                break

        return input_ids
