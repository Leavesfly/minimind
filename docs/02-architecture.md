# 02 - 项目架构与目录结构

## 2.1 顶层目录速览

```
minimind3/
├── model/                     # 🧠 模型层
│   ├── model_minimind.py      #   核心模型: Config + Dense/MoE Transformer
│   ├── model_lora.py          #   LoRA 注入/保存/加载/合并
│   ├── tokenizer_config.json  #   分词器配置（含 chat_template Jinja）
│   └── __init__.py
│
├── dataset/                   # 📦 数据层
│   ├── lm_dataset.py          #   5 类 Dataset + chat 预/后处理
│   └── dataset.md             #   数据集说明（JSONL 格式约定）
│
├── trainer/                   # 🎓 训练层
│   ├── train_pretrain.py      #   预训练
│   ├── train_full_sft.py      #   全量 SFT
│   ├── train_lora.py          #   LoRA 微调
│   ├── train_dpo.py           #   DPO 偏好对齐
│   ├── train_distillation.py  #   知识蒸馏
│   ├── train_ppo.py           #   PPO 强化学习
│   ├── train_grpo.py          #   GRPO / CISPO 强化学习
│   ├── train_agent.py         #   多轮 Tool Use Agentic RL
│   ├── train_tokenizer.py     #   BPE 分词器训练
│   ├── trainer_utils.py       #   共享: DDP / AMP / Ckpt / Sampler / RM
│   ├── reward_utils.py        #   RL 奖励工具: 重复惩罚 / 思考评分
│   └── rollout_engine.py      #   可插拔推理引擎: Torch / SGLang
│
├── scripts/                   # 🔌 推理与服务
│   ├── serve_openai_api.py    #   OpenAI 兼容 FastAPI 服务
│   ├── chat_api.py            #   命令行 OpenAI Client
│   ├── web_demo.py            #   Streamlit Web Demo
│   ├── eval_toolcall.py       #   工具调用评测
│   └── convert_model.py       #   权重格式转换 / LoRA 合并
│
├── tests/                     # 🧪 测试
│   └── test_trainer_utils.py
│
├── docs/                      # 📚 本技术 Wiki
├── eval_llm.py                # 命令行推理与多轮对话入口
├── requirements.txt           # PyTorch 2.6 + Transformers 4.57 等
├── README.md / README_en.md   # 项目主页文档
└── LICENSE                    # Apache 2.0
```

## 2.2 模块依赖关系

```mermaid
graph TD
    subgraph Layer1[底层]
        A1[model/model_minimind.py<br/>MiniMindConfig + ForCausalLM]
        A2[model/model_lora.py<br/>LoRA]
        A3[model/tokenizer_config.json<br/>BPE + chat_template]
    end

    subgraph Layer2[数据层]
        B1[dataset/lm_dataset.py<br/>5 类 Dataset]
    end

    subgraph Layer3[训练共享层]
        C1[trainer/trainer_utils.py<br/>DDP / AMP / Ckpt / RM]
        C2[trainer/reward_utils.py<br/>奖励组件]
        C3[trainer/rollout_engine.py<br/>Rollout 抽象]
    end

    subgraph Layer4[训练脚本]
        D1[train_pretrain.py]
        D2[train_full_sft.py]
        D3[train_lora.py]
        D4[train_dpo.py]
        D5[train_distillation.py]
        D6[train_ppo.py]
        D7[train_grpo.py]
        D8[train_agent.py]
    end

    subgraph Layer5[推理服务]
        E1[eval_llm.py]
        E2[scripts/serve_openai_api.py]
        E3[scripts/web_demo.py]
        E4[scripts/chat_api.py]
    end

    A1 --> B1
    A3 --> B1
    A1 --> C1
    A2 --> D3
    B1 --> D1 & D2 & D3 & D4 & D5 & D6 & D7 & D8
    C1 --> D1 & D2 & D3 & D4 & D5 & D6 & D7 & D8
    C2 --> D6 & D7 & D8
    C3 --> D6 & D7 & D8
    A1 --> E1 & E2 & E3
    A2 --> E1 & E2
```

## 2.3 训练-推理全景数据流

```mermaid
flowchart TB
    subgraph DataPipeline[数据管道]
        DS1[原始 JSONL 数据]
        DS2[Tokenizer<br/>BPE + ByteLevel]
        DS3[5 类 Dataset<br/>多进程预 tokenize]
        DS1 --> DS2 --> DS3
    end

    subgraph Training[训练阶段]
        T1[Pretrain<br/>pretrain_t2t]
        T2[SFT<br/>sft_t2t + tool_call]
        T3a[LoRA Adapter]
        T3b[DPO<br/>chosen/rejected]
        T3c[RLAIF<br/>rlaif.jsonl]
        T3d[Agentic RL<br/>agent_rl.jsonl]
        T3e[Distillation MoE to Dense]
        DS3 --> T1 --> T2
        T2 --> T3a & T3b & T3c & T3d & T3e
    end

    subgraph CkptStore[权重存储 out/]
        CKP1[pretrain_*.pth]
        CKP2[full_sft_*.pth]
        CKP3[lora_*.pth]
        CKP4[dpo_*.pth]
        CKP5[grpo_*.pth]
        CKP6[full_dist_*.pth]
        T1 --> CKP1
        T2 --> CKP2
        T3a --> CKP3
        T3b --> CKP4
        T3c --> CKP5
        T3e --> CKP6
    end

    subgraph Inference[推理与服务]
        I1[eval_llm.py 命令行]
        I2[serve_openai_api.py FastAPI]
        I3[web_demo.py Streamlit]
        I4[第三方: vllm/llama.cpp/ollama]
        CKP1 & CKP2 & CKP3 & CKP4 & CKP5 & CKP6 --> I1 & I2 & I3 & I4
    end
```

## 2.4 训练阶段串联关系

| 阶段 | 默认 `from_weight` | 默认 `save_weight` | 数据 |
|------|-------------------|-------------------|------|
| Pretrain | `none`（从随机初始化） | `pretrain` | `pretrain_t2t_mini.jsonl` |
| Full SFT | `pretrain` | `full_sft` | `sft_t2t_mini.jsonl` |
| LoRA | `full_sft` | `lora_*`（独立保存） | `lora_*.jsonl` |
| DPO | `full_sft` | `dpo` | `dpo.jsonl` |
| Distillation | `full_sft`（学生）+ `full_sft_moe`（教师） | `full_dist` | `sft_t2t_mini.jsonl` |
| PPO/GRPO | `full_sft` | `ppo_actor` / `grpo` | `rlaif.jsonl` |
| Agent RL | `full_sft` | `agent_rl` | `agent_rl.jsonl` |

> 💡 **工程要点**：所有训练脚本通过 `--from_weight` 参数指定起点权重，`--save_weight` 指定保存前缀。权重文件名约定：`{save_weight}_{hidden_size}{_moe}.pth`。

## 2.5 设备适配策略

`trainer_utils.py` 中的 `get_default_device()` 会按优先级 `cuda > mps > cpu` 自动选择，并针对不同设备做以下差异化处理：

| 设备 | Flash Attn | AMP | DataLoader workers | 数据存放 |
|------|-----------|-----|-------------------|---------|
| CUDA | ✅ 启用 | ✅ bf16/fp16 + GradScaler | 多进程 + pin_memory | CPU pinned |
| MPS | ❌ 强制关闭（SDPA on MPS 极慢） | ❌ 禁用（fp32 native 最快） | 0（数据已在 GPU） | GPU 零拷贝 |
| CPU | 不适用 | 关闭 | 默认 | CPU |

具体实现见 `trainer/train_pretrain.py` 中第 200~240 行的设备分支与 [14 - 训练工具链](./14-trainer-utils.md)。

## 2.6 后续阅读

- 想理解 `MiniMindForCausalLM` 的每一层实现 → [03 - 模型架构](./03-model-architecture.md)
- 想理解数据是如何变成 input_ids 的 → [05 - 数据管道](./05-dataset-pipeline.md)
- 想理解断点续训和 DDP 是如何统一的 → [14 - 训练工具链](./14-trainer-utils.md)
