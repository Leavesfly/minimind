import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏

class MiniMindConfig(PretrainedConfig):
    """
    MiniMind 模型配置类
    
    继承自 PretrainedConfig，定义模型的所有超参数和架构配置。
    
    参数说明：
        hidden_size (int): 隐藏层维度，默认 768
        num_hidden_layers (int): Transformer 层数，默认 8
        use_moe (bool): 是否使用 MoE 混合专家机制，默认 False
        dropout (float): Dropout 概率，默认 0.0
        vocab_size (int): 词汇表大小，默认 6400
        bos_token_id (int): 开始标记 ID，默认 1
        eos_token_id (int): 结束标记 ID，默认 2
        flash_attn (bool): 是否使用 Flash Attention，默认 True
        num_attention_heads (int): 注意力头数，默认 8
        num_key_value_heads (int): GQA 中 KV 头数，默认 4（用于分组查询注意力）
        head_dim (int): 每个注意力头的维度，默认 hidden_size // num_attention_heads
        hidden_act (str): 激活函数类型，默认 'silu'
        intermediate_size (int): 前馈网络中间层维度，默认为 hidden_size * π / 64 的倍数
        max_position_embeddings (int): 最大位置编码长度，默认 32768
        rms_norm_eps (float): RMSNorm 的 epsilon 稳定项，默认 1e-6
        rope_theta (float): RoPE 的基频参数，默认 1e6
        inference_rope_scaling (bool): 推理时是否使用 YaRN 外推，默认 False
        rope_scaling (dict): YaRN 外推配置，包含 beta_fast、beta_slow、factor 等参数
        num_experts (int): MoE 专家数量，默认 4
        num_experts_per_tok (int): 每个 token 选择的专家数量，默认 1
        moe_intermediate_size (int): MoE 专家的中间层维度
        norm_topk_prob (bool): 是否归一化 top-k 概率，默认 True
        router_aux_loss_coef (float): 负载均衡损失系数，默认 5e-4
    """
    model_type = "minimind"

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏

class RMSNorm(torch.nn.Module):
    """
    RMS 归一化层（Root Mean Square Normalization）
    
    RMSNorm 是 LayerNorm 的简化版本，去掉了均值归一化，仅保留均方根归一化。
    相比 LayerNorm，RMSNorm 计算更简单、速度更快，且在语言模型中表现相当。
    
    公式：RMSNorm(x) = x / sqrt(mean(x²) + eps) * γ
    
    参数说明：
        dim (int): 归一化的维度
        eps (float): 数值稳定项，防止除零，默认 1e-5
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        # 计算 RMS: sqrt(mean(x²) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 先转换为 float32 保证精度，再转回原类型
        return (self.weight * self.norm(x.float())).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """
    预计算 RoPE 位置编码的余弦和正弦值
    
    RoPE（Rotary Position Embedding）通过旋转矩阵将位置信息注入到 query 和 key 中，
    使模型能够感知 token 的相对位置关系。
    
    原理：将位置索引映射到旋转角度，通过复数乘法实现旋转。
    对于维度 d 的向量，位置 m 的旋转角度为：θ_i = base^(-2i/d)
    
    YaRN 外推机制：
    当序列长度超过训练长度时，通过调整频率来扩展位置编码的有效范围。
    核心思想：对低频维度保持不变，对高频维度进行缩放，实现平滑外推。
    公式：f'(i) = f(i)((1-γ) + γ/s)，其中 γ 是线性斜坡函数
    
    参数说明：
        dim (int): 每个注意力头的维度
        end (int): 最大序列长度，默认 32K
        rope_base (float): RoPE 基频，默认 1e6
        rope_scaling (dict): YaRN 外推配置，包含 beta_fast、beta_slow、factor 等
    
    返回：
        freqs_cos: 余弦位置编码 [max_len, head_dim]
        freqs_sin: 正弦位置编码 [max_len, head_dim]
    """
    # 计算基础频率：f(i) = 1 / (base^(2i/d))
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0

    # YaRN 外推：对超出原始长度的序列进行频率调整
    if rope_scaling is not None:  # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # 计算需要调整的维度范围
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # 构建线性斜坡函数 γ
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0,
                               1)
            # 应用 YaRN 缩放：低频不变，高频缩放
            freqs = freqs * (1 - ramp + ramp / factor)

    # 生成位置索引并计算频率：freqs[m, i] = m * f(i)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    # 计算余弦和正弦编码，每个维度复制一次以匹配 head_dim
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    应用旋转位置编码到 query 和 key
    
    通过复数乘法实现旋转：(a + bi) * (cosθ + i*sinθ) = (a*cosθ - b*sinθ) + (a*sinθ + b*cosθ)
    实现时将向量分为两半，分别对应复数的实部和虚部。
    
    参数说明：
        q: query 张量 [batch, seq_len, n_heads, head_dim]
        k: key 张量 [batch, seq_len, n_kv_heads, head_dim]
        cos: 余弦位置编码 [seq_len, head_dim]
        sin: 正弦位置编码 [seq_len, head_dim]
        unsqueeze_dim: 在哪个维度添加位置编码维度
    
    返回：
        q_embed: 应用 RoPE 后的 query
        k_embed: 应用 RoPE 后的 key
    """

    # 将向量旋转 90 度：[x1, x2, ..., x_n/2, x_n/2+1, ..., x_n] -> [-x_n/2+1, ..., -x_n, x1, ..., x_n/2]
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 应用旋转：q_rot = q * cos + rotate_half(q) * sin
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复 KV 头以匹配 Q 头数量（用于 GQA 分组查询注意力）
    
    GQA（Grouped Query Attention）通过共享 KV 头来减少显存和计算开销。
    例如：8 个 Q 头，2 个 KV 头，则每个 KV 头需要复制 4 次来匹配 Q 头。
    
    参数说明：
        x: KV 张量 [batch, seq_len, n_kv_heads, head_dim]
        n_rep: 重复次数（每个 KV 头需要复制的次数）
    
    返回：
        重复后的 KV 张量 [batch, seq_len, n_kv_heads * n_rep, head_dim]
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    # 先添加一个维度，然后扩展，最后重塑形状
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen,
                                                                                               num_key_value_heads * n_rep,
                                                                                               head_dim))


class Attention(nn.Module):
    """
    多头注意力层（支持 GQA 和 Flash Attention）
    
    实现了标准的多头注意力机制，并支持以下优化：
    - GQA（Grouped Query Attention）：分组查询注意力，减少显存和计算
    - QK Norm：对 Q 和 K 进行归一化，提升训练稳定性
    - Flash Attention：优化的注意力实现，大幅提升训练速度
    
    参数说明：
        config (MiniMindConfig): 模型配置对象
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个 KV 头需要复制的次数
        self.head_dim = config.head_dim
        self.is_causal = True  # 因果注意力（只能看到之前的 token）

        # QKV 投影层
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # QK 归一化（提升训练稳定性）
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # 检查是否支持 Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        前向传播
        
        参数说明：
            x: 输入张量 [batch, seq_len, hidden_size]
            position_embeddings: 位置编码 (cos, sin)
            past_key_value: 缓存的 KV（用于加速生成）
            use_cache: 是否缓存 KV
            attention_mask: 注意力掩码
        
        返回：
            output: 注意力输出 [batch, seq_len, hidden_size]
            past_kv: 缓存的 KV（如果 use_cache=True）
        """
        bsz, seq_len, _ = x.shape

        # QKV 投影并重塑为多头格式
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # QK 归一化
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        # 应用 RoPE 位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 拼接缓存的 KV（用于加速生成）
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 重复 KV 头以匹配 Q 头（GQA）
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2),
                      repeat_kv(xv, self.n_rep).transpose(1, 2))

        # 使用 Flash Attention 或标准注意力
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (
                attention_mask is None or torch.all(attention_mask == 1)):
            # Flash Attention：优化的注意力实现
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0,
                                                    is_causal=self.is_causal)
        else:
            # 标准注意力实现
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # 注意力分数
            if self.is_causal:
                # 因果掩码：只能看到之前的 token
                scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(
                    1)
            if attention_mask is not None:
                # 应用注意力掩码（如 padding mask）
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv

        # 输出投影
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """
    SwiGLU 前馈网络层
    
    SwiGLU 是一种高效的前馈网络结构，相比标准的 ReLU MLP 有更好的性能。
    结构：x -> gate_proj(x) * up_proj(x) -> down_proj
    其中 gate_proj 和 up_proj 使用相同的激活函数（SiLU），然后逐元素相乘。
    
    公式：FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    
    参数说明：
        config (MiniMindConfig): 模型配置对象
        intermediate_size (int): 中间层维度，默认使用 config.intermediate_size
    """

    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # SwiGLU: SiLU(gate) * up，然后通过 down_proj 投影回 hidden_size
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MOEFeedForward(nn.Module):
    """
    混合专家（MoE）前馈网络层
    
    MoE（Mixture of Experts）通过将输入路由到不同的专家网络来提升模型容量，
    同时保持计算效率。每个 token 只选择 top-k 个专家进行处理。
    
    核心组件：
    - Gate（门控网络）：决定每个 token 选择哪些专家
    - Experts（专家网络）：多个独立的前馈网络
    - 负载均衡损失：确保专家负载均衡，避免某些专家过载
    
    参数说明：
        config (MiniMindConfig): 模型配置对象
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 门控网络：输出每个 token 对每个专家的选择概率
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # 专家网络列表
        self.experts = nn.ModuleList(
            [FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # 展平为 [batch*seq_len, hidden_dim]

        # 门控网络计算专家选择概率
        scores = F.softmax(self.gate(x_flat), dim=-1)

        # 选择 top-k 个专家
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)

        # 归一化 top-k 概率（可选）
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        # 初始化输出
        y = torch.zeros_like(x_flat)

        # 对每个专家进行处理
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)  # 找出选择当前专家的 token
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()  # 选择当前专家的 token 索引
                weight = topk_weight[mask].view(-1, 1)  # 对应的权重
                # 累加专家输出（加权求和）
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # 训练时确保所有专家参与反向传播（避免专家失效）
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        # 计算负载均衡损失（训练时）
        if self.training and self.config.router_aux_loss_coef > 0:
            # 统计每个专家被选择的频率
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            # 负载均衡损失 = 专家负载 * 门控概率，鼓励均匀分配
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()

        return y.view(batch_size, seq_len, hidden_dim)


class MiniMindBlock(nn.Module):
    """
    Transformer Block（Pre-Norm 架构）
    
    实现了标准的 Transformer Block，使用 Pre-Norm 结构（先归一化再计算）。
    Pre-Norm 相比 Post-Norm 训练更稳定，适合深层网络。
    
    结构：
    - 输入 -> RMSNorm -> Attention -> 残差连接
    - -> RMSNorm -> MLP/MoE -> 残差连接 -> 输出
    
    参数说明：
        layer_id (int): 层索引
        config (MiniMindConfig): 模型配置对象
    """

    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 根据配置选择标准 MLP 或 MoE
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # Pre-Norm 注意力层：先归一化，再计算注意力，最后残差连接
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual

        # Pre-Norm MLP/MoE 层：先归一化，再计算前馈，最后残差连接
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    MiniMind 完整模型
    
    包含词嵌入、多层 Transformer Block 和最终的归一化层。
    
    结构：
    - 词嵌入层
    - Dropout
    - N 层 Transformer Block
    - RMSNorm 归一化
    - 预计算 RoPE 位置编码
    
    参数说明：
        config (MiniMindConfig): 模型配置对象
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers

        # 词嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer Block 层
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])

        # 最终归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算 RoPE 位置编码
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings,
                                                    rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        """
        前向传播
        
        参数说明：
            input_ids: 输入 token ID [batch, seq_len]
            attention_mask: 注意力掩码
            past_key_values: 缓存的 KV（用于加速生成）
            use_cache: 是否缓存 KV
        
        返回：
            hidden_states: 最终隐藏状态 [batch, seq_len, hidden_size]
            presents: 缓存的 KV（如果 use_cache=True）
            aux_loss: MoE 负载均衡损失
        """
        batch_size, seq_length = input_ids.shape

        # 处理 past_key_values
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)

        # 计算当前序列的起始位置（用于增量生成）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 词嵌入 + Dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 获取位置编码（根据起始位置切片）
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length],
                               self.freqs_sin[start_pos:start_pos + seq_length])

        # 逐层前向传播
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # 最终归一化
        hidden_states = self.norm(hidden_states)

        # 累加所有 MoE 层的负载均衡损失
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
                       hidden_states.new_zeros(1).squeeze())

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    因果语言模型（用于文本生成）
    
    在 MiniMindModel 基础上添加语言模型头（lm_head），
    支持训练时的损失计算和推理时的文本生成。
    
    特性：
    - 词嵌入和输出层共享权重（减少参数）
    - 支持多种采样策略（top-k、top-p、温度采样）
    - 支持重复惩罚
    - 支持流式输出
    
    参数说明：
        config (MiniMindConfig): 模型配置对象
    """
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 词嵌入和输出层共享权重
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0,
                labels=None, **kwargs):
        """
        前向传播（训练模式）
        
        参数说明：
            input_ids: 输入 token ID [batch, seq_len]
            attention_mask: 注意力掩码
            past_key_values: 缓存的 KV
            use_cache: 是否缓存 KV
            logits_to_keep: 只保留最后 N 个位置的 logits（用于节省显存）
            labels: 训练标签 [batch, seq_len]
        
        返回：
            MoeCausalLMOutputWithPast: 包含 loss、aux_loss、logits、past_key_values、hidden_states
        """
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache,
                                                              **kwargs)

        # 只保留最后 N 个位置的 logits（用于节省显存）
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # 计算交叉熵损失
        loss = None
        if labels is not None:
            # 预测下一个 token：logits[:, :-1] vs labels[:, 1:]
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)

        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values,
                                         hidden_states=hidden_states)

    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50,
                 eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True,
                 repetition_penalty=1.0, **kwargs):
        """
        文本生成（推理模式）
        
        支持多种采样策略：
        - 温度采样：控制输出的随机性
        - Top-k 采样：只从概率最高的 k 个 token 中采样
        - Top-p（nucleus）采样：累积概率达到 p 的 token 集合中采样
        - 重复惩罚：惩罚重复出现的 token
        
        参数说明：
            inputs: 输入 token ID [batch, seq_len]
            attention_mask: 注意力掩码
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度，越高越随机
            top_p: nucleus 采样阈值
            top_k: top-k 采样阈值
            eos_token_id: 结束标记 ID
            streamer: 流式输出处理器
            use_cache: 是否使用 KV 缓存
            num_return_sequences: 生成序列数量
            do_sample: 是否使用采样（False 则使用贪心解码）
            repetition_penalty: 重复惩罚系数
        
        返回：
            generated_ids: 生成的 token ID
        """
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        if streamer: streamer.put(input_ids.cpu())

        # 自回归生成循环
        for _ in range(max_new_tokens):
            # 计算当前需要处理的位置
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0

            # 前向传播
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache,
                                   **kwargs)

            # 更新注意力掩码
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)],
                                       -1) if attention_mask is not None else None

            # 获取下一个 token 的 logits
            logits = outputs.logits[:, -1, :] / temperature

            # 重复惩罚：降低已出现 token 的概率
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]): logits[i, torch.unique(input_ids[i])] /= repetition_penalty

            # Top-k 采样：只保留概率最高的 k 个 token
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')

            # Top-p（nucleus）采样：累积概率达到 p 的 token 集合
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')

            # 采样下一个 token
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(
                logits, dim=-1, keepdim=True)

            # 处理已完成的序列
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1),
                                                                  next_token.new_full((next_token.shape[0], 1),
                                                                                      eos_token_id), next_token)

            # 拼接新生成的 token
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None

            # 流式输出
            if streamer: streamer.put(next_token.cpu())

            # 检查是否所有序列都已完成
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break

        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
