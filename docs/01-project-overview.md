# 01 - 项目概览

## 1.1 项目定位

**MiniMind3** 是一个完全从 0 实现的极小语言模型（约 64M 参数，MoE 版 198M / A64M），目标是让普通个人 GPU 也能在 2 小时内完成全流程训练。它不是一个"封装框架"，而是一份覆盖以下完整链路的**可读、可改、可复现**的代码工程：

1. **Tokenizer 训练**（`trainer/train_tokenizer.py`，BPE + ByteLevel）
2. **预训练**（`trainer/train_pretrain.py`）
3. **监督微调**（`trainer/train_full_sft.py`，已混入 Tool Call）
4. **LoRA 高效微调**（`model/model_lora.py` + `trainer/train_lora.py`，从 0 实现）
5. **DPO 偏好对齐**（`trainer/train_dpo.py`，PyTorch 原生）
6. **RLAIF 强化学习**（`train_ppo.py` / `train_grpo.py`，含 CISPO 变体）
7. **Agentic RL**（`trainer/train_agent.py`，多轮 Tool Use）
8. **知识蒸馏**（`trainer/train_distillation.py`，支持 MoE→Dense）
9. **OpenAI 兼容服务**（`scripts/serve_openai_api.py`）
10. **Streamlit Web Demo**（`scripts/web_demo.py`）

## 1.2 核心特性

### 模型层（对齐 Qwen3 / Qwen3-MoE 生态）

- **RMSNorm + RoPE**：标准 Qwen3 风格归一化与位置编码
- **GQA（Grouped Query Attention）**：默认 `num_kv_heads=2`，显著减少 KV Cache
- **QK Norm**：对 Q/K 做 RMSNorm，提升训练稳定性
- **Flash Attention**：CUDA 上启用 `F.scaled_dot_product_attention`，MPS 上自动回退手动实现
- **YaRN**：通过 `inference_rope_scaling=True` 实现 RoPE 长文本外推
- **MoE**：`num_experts=4`、Top-1 路由、负载均衡 aux loss（系数 5e-4）

### 训练层

- **统一工具链**：`trainer/trainer_utils.py` 抽象出 DDP 初始化、混合精度、Ckpt 保存/恢复、`SkipBatchSampler`、Reward Model 等公共能力
- **跨设备适配**：CUDA / MPS / CPU 自动检测，MPS 上专门做了 attention/AMP/DataLoader 的差异化优化
- **断点续训**：`lm_checkpoint` + `restore_training_state` 支持自动 Resume，跨 GPU 数量自动调整 step
- **可视化双轨**：同时支持 WandB 和 TensorBoard，国内可用 SwanLab

### 推理与生态

- **OpenAI API 协议**：`/v1/chat/completions` 完整兼容，支持 `reasoning_content`、`tool_calls`、流式
- **思考开关**：`open_thinking` 自适应控制 `` |
| 工具调用标记 | `<tool_call>` / `</tool_call>` |
| 工具响应标记 | `<tool_response>` / `</tool_response>` |
| 系统标记 | `<|system|>` 等 |

> 小词表 + 紧凑 Embedding 是 MiniMind 能在小模型尺寸下保持效率的核心因素之一。Tokenizer 与 Chat Template 的详细解读见 [04 - Tokenizer 与 Chat Template](./04-tokenizer-and-chat-template.md)。
