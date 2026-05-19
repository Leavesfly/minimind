# 01 - 项目概览

> 如果 GPT-4 是一栋摩天大楼，MiniMind3 就是一套完整的乐高积木——它让你亲手从第一块砖开始搭建，从而真正理解大楼是怎么站起来的。

## 1.1 这个项目为什么存在？

### 核心矛盾

大语言模型（LLM）正在改变世界，但理解它的门槛极高：

- **看论文**：公式密密麻麻，但你不知道工程上到底怎么跑通
- **看开源大模型代码**：动辄几十万行，抽象层层叠叠，看完不知道从哪里开始
- **用框架训练**：一行 `Trainer.train()` 跑完了，但"里面到底发生了什么"一无所知

### MiniMind3 的定位

**MiniMind3** 是一个完全从 0 实现的极小语言模型（约 64M 参数，MoE 版 198M / A64M）。它的核心理念是：

> **把大模型训练的每一步都摊开在阳光下，让你用一台普通 GPU 在 2 小时内走完整条路。**

它不是一个"封装框架"，而是一份**可读、可改、可复现**的代码工程，覆盖 LLM 的完整生命周期：

| 阶段 | 类比 | 对应代码 |
|------|------|---------|
| 1. 造词典 | 给模型发明一套"字母表" | `trainer/train_tokenizer.py`（BPE + ByteLevel） |
| 2. 预训练 | 让模型通读所有书籍，学会语言本身 | `trainer/train_pretrain.py` |
| 3. 监督微调 | 教模型"听问题→答问题"的对话格式 | `trainer/train_full_sft.py`（含 Tool Call） |
| 4. LoRA 适配 | 用极少参数让模型学会新领域技能 | `model/model_lora.py` + `trainer/train_lora.py` |
| 5. DPO 偏好对齐 | 告诉模型"这个答案好、那个答案差" | `trainer/train_dpo.py` |
| 6. RLAIF 强化学习 | 让模型通过试错自我进化 | `train_ppo.py` / `train_grpo.py`（含 CISPO） |
| 7. Agentic RL | 教模型使用工具解决复杂问题 | `trainer/train_agent.py` |
| 8. 知识蒸馏 | 让大模型把"聪明"传给小模型 | `trainer/train_distillation.py`（MoE→Dense） |
| 9. API 服务 | 模型毕业后去"上班" | `scripts/serve_openai_api.py` |
| 10. Web Demo | 模型毕业后的"面试展示" | `scripts/web_demo.py` |

## 1.2 核心设计哲学

### 模型层：对齐 Qwen3 生态，但足够小

想象你在学开车——你不需要一辆法拉利，你需要一辆**结构完整但体积迷你**的教练车。MiniMind3 的模型架构与 Qwen3 / Qwen3-MoE 完全对齐，但参数只有 64M：

- **RMSNorm + RoPE**：标准归一化与位置编码（像车的方向盘和仪表盘，必须有但可以简化）
- **GQA**（`num_kv_heads=2`）：多个注意力头共用同一份"笔记"，省内存不损精度
- **QK Norm**：防止训练初期数值跑飞的"安全带"
- **Flash Attention**：CUDA 上自动加速，MPS 上智能回退
- **YaRN**：推理时将上下文从 2K 扩展到 32K 的"望远镜"
- **MoE**（4 专家、Top-1 路由）：不同问题分给不同专家处理的"分诊台"

### 训练层：一切显式可见

不用 `transformers.Trainer`，不用 `accelerate`——每一行训练循环都写在明面上，你能看到梯度怎么流动、学习率怎么衰减、断点怎么恢复：

- **统一底座**：`trainer/trainer_utils.py` 抽象出 DDP、混合精度、检查点管理等公共能力
- **跨设备适配**：CUDA / MPS / CPU 自动检测，各有针对性优化
- **断点续训**：训练中断后从上次停下的地方继续，甚至换 GPU 数量也能无缝接续
- **可视化双轨**：WandB / TensorBoard / SwanLab 随你选

### 推理与生态：训完即可用

模型训完不是终点——它需要"上班"。MiniMind3 提供完整的部署链路：

- **OpenAI 兼容 API**：`/v1/chat/completions` 完整实现，支持 `reasoning_content`、`tool_calls`、流式输出
- **思考开关**：`open_thinking` 控制模型是否先"想一想"再回答
- **Tool Call**：基于 `<tool_call>` 标签的 OpenAI Function Calling 兼容协议
- **一键转换**：输出 Qwen3 兼容格式，可直接被 vLLM / llama.cpp / Ollama 加载

> 小词表（6400）+ 紧凑 Embedding 是 MiniMind 能在小模型尺寸下保持效率的核心因素之一。详见 [04 - Tokenizer 与 Chat Template](./04-tokenizer-and-chat-template.md)。
