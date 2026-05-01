# 13 - 评测体系

> 对应代码：`eval_llm.py` + `scripts/eval_toolcall.py` + `tests/test_trainer_utils.py`

## 13.1 MiniMind3 的评测策略

MiniMind3 作为**学习导向**的小模型项目，没有强行追求 Open LLM Leaderboard 等重量级 benchmark，而是聚焦**实用性 + 可解释性**：

| 评测维度 | 工具 | 数据 |
|---------|------|------|
| 通用对话 | `eval_llm.py` 交互/批测 | 自定义问题集 |
| Tool Call | `scripts/eval_toolcall.py` | 内置工具用例 |
| 工程正确性 | `tests/test_trainer_utils.py` | pytest |
| 训练监控指标 | wandb / tensorboard | 训练日志 |

> 对于 C-Eval / MMLU / HumanEval 这类 academic benchmark，建议使用 **HuggingFace lm-eval-harness** 配合 `convert_model.py --to_hf` 后接入。

## 13.2 通用对话评测：`eval_llm.py --batch_test`

### 13.2.1 数据格式

`eval/test_questions.jsonl`（项目自定义）：

```json
{"question": "请用一句话解释什么是黑洞？"}
{"question": "中国的四大发明是？"}
{"question": "写一首关于春天的五言绝句"}
```

### 13.2.2 启动

```bash
python eval_llm.py \
    --weight full_sft --hidden_size 512 \
    --batch_test --test_file ./eval/test_questions.jsonl \
    --temperature 0.7 --top_p 0.9
```

### 13.2.3 输出

`./eval/results_{timestamp}.jsonl`：

```json
{"question": "...", "answer": "...", "latency_ms": 1234, "tokens": 89}
```

可直接 diff 不同 checkpoint 的回答质量。

## 13.3 Tool Call 评测：`scripts/eval_toolcall.py`

### 13.3.1 评测范围

- **Format**：tool_call JSON 是否合法
- **Selection**：是否选择了正确的工具
- **Argument**：参数是否正确（含必填项）
- **Multi-turn**：是否能根据 tool_response 继续推理

### 13.3.2 内置用例

```python
TEST_CASES = [
    {
        "user": "查一下北京今天的天气",
        "tools": [WEATHER_TOOL],
        "expected_tool": "get_weather",
        "expected_args": {"location": "北京"}
    },
    {
        "user": "1234*5678 等于多少？",
        "tools": [CALCULATOR_TOOL],
        "expected_tool": "calculator",
        "expected_args_check": lambda args: "1234" in args["expression"]
    },
    ...
]
```

### 13.3.3 启动

```bash
python scripts/eval_toolcall.py --weight agent --hidden_size 512
```

输出：

```
Format Pass:    19/20  (95.0%)
Selection Pass: 17/20  (85.0%)
Argument Pass:  16/20  (80.0%)
Overall Score:  86.7%
```

## 13.4 工程单元测试：`tests/test_trainer_utils.py`

```bash
pytest tests/test_trainer_utils.py -v
```

覆盖核心训练工具：
- `get_lr`：学习率调度曲线
- `format_duration`：人类可读时间格式
- `get_model_params`：参数量统计
- `SkipBatchSampler`：断点续训跳过逻辑
- `restore_training_state`：状态恢复正确性
- `lm_checkpoint`：路径生成规则

> 这是 MiniMind3 唯一的 unit test 文件，体现了"工具链需要可靠，业务逻辑可手测"的取舍。

## 13.5 训练期监控指标

详见各训练章节，关键指标汇总：

### 13.5.1 通用

| 指标 | 含义 | 健康范围 |
|------|------|---------|
| `loss` | 当前 batch 损失 | 平滑下降 |
| `avg_loss` | 滑动平均 | 整体趋势 |
| `learning_rate` | 当前 lr | 余弦下降 |
| `tokens_per_sec` | 训练吞吐 | 越高越好 |
| `epoch_eta_min` | 当前 epoch 剩余分钟 | 反映节奏 |

### 13.5.2 MoE 专用

| 指标 | 含义 |
|------|------|
| `aux_loss` | 负载均衡损失 |
| `expert_load_std` | 各专家负载标准差（需自行打日志） |

### 13.5.3 RL 专用

| 指标 | 含义 |
|------|------|
| `mean_reward` | 一批 rollout 的平均 reward |
| `kl_div` | policy 与 ref 的 KL |
| `clip_frac` | PPO clip 触发比例 |
| `response_length` | 平均生成长度 |
| `reward_acc` (DPO) | chosen > rejected 的比例 |

## 13.6 经验性效果对比

> 以下为社区报告的近似值，仅供参考，不代表精确数字。

| 模型 | 中文理解 | 英文理解 | Code | Math | Tool Call |
|------|---------|---------|------|------|----------|
| MiniMind3-Pretrain | ★ | ★ | × | × | × |
| MiniMind3-SFT | ★★★ | ★★ | ★ | ★ | ★★ |
| MiniMind3-DPO | ★★★ | ★★ | ★ | ★ | ★★ |
| MiniMind3-GRPO | ★★★ | ★★ | ★★ | ★★★ | ★★ |
| MiniMind3-Agent | ★★★ | ★★ | ★★ | ★★★ | ★★★ |
| MoE-A64M-Full-SFT | ★★★ | ★★★ | ★★ | ★★ | ★★ |

## 13.7 与官方 benchmark 接入

如需 lm-eval-harness：

```bash
# 1. 转 HF 格式
python scripts/convert_model.py --to_hf \
    --weight full_sft --output_dir ./hf_minimind

# 2. 运行 lm-eval
lm_eval --model hf \
        --model_args pretrained=./hf_minimind \
        --tasks ceval-valid,mmlu,gsm8k \
        --batch_size 8
```

## 13.8 推荐评测流程

```mermaid
flowchart LR
    A[训练完成] --> B[batch_test 通用对话]
    B --> C[eval_toolcall Tool 能力]
    C --> D[手动 web_demo 体感]
    D --> E[lm-eval-harness 横评]
    E --> F[发布 / 进入下一阶段]
```
