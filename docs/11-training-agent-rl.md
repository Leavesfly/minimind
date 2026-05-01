# 11 - Agentic RL（多轮 Tool Use 强化学习）

> 对应代码：`trainer/train_agent.py` + `dataset/lm_dataset.py:AgentRLDataset` + `trainer/rollout_engine.py`（多轮分支）

## 11.1 Agentic RL 定位

普通 RLAIF 只优化"一次性回答"，而 **Agentic RL** 让模型在回合内**多次调用工具、读取工具结果、再作答**，最终用 task-level reward 优化整条轨迹（trajectory）。

MiniMind3 的 Agentic RL 是 **GRPO 的多轮扩展**：

```
Step 0: model 生成 → <tool_call>{...}</tool_call>
Step 1: 工具执行 → <tool_response>{...}</tool_response>  注入 prompt
Step 2: model 生成 → 可能继续 tool_call 或最终回答
...
Step N: model 输出最终答案 → reward
```

整条轨迹共享同一个 reward，反向传播只在 **assistant 生成的 token** 上回传。

## 11.2 数据集：`AgentRLDataset`

```json
{
  "messages": [
    {"role": "system", "content": "你可以调用以下工具…"},
    {"role": "user", "content": "查询北京和上海的天气，对比"}
  ],
  "tools": [
    {"type":"function","function":{"name":"get_weather", "parameters":{...}}}
  ],
  "answer_check": {
    "type": "regex",
    "pattern": "北京.*?\\d+°C.*?上海.*?\\d+°C"
  }
}
```

字段说明：
- `messages` + `tools`：构成初始 prompt
- `answer_check`：用于轨迹完成后判定 reward

## 11.3 多轮 Rollout

`RolloutEngine` 在 Agent 模式下进入循环：

```python
def agent_rollout(prompt, tools, max_turns=6):
    history = prompt
    trajectory = []
    for turn in range(max_turns):
        out = model.generate(history)
        trajectory.append(out)
        if "<tool_call>" in out:
            tool_result = execute_tool(parse_tool_call(out), tools)
            history = history + out + render_tool_response(tool_result)
        else:
            break  # 最终回答
    return history, trajectory
```

### 工具执行

`trainer/train_agent.py` 提供本地 tool dispatch 表（mock + sandbox）：

```python
TOOL_REGISTRY = {
    "get_weather": mock_get_weather,
    "calculator":  safe_eval,
    "search":      mock_search,
    ...
}
```

支持注册自定义 Python 函数，签名约定 `def tool(**kwargs) -> str`。

## 11.4 Reward 设计

Agentic 任务的 reward 是**多目标加权**：

| 子 reward | 权重 | 说明 |
|----------|------|------|
| `task_success` | 1.0 | 最终答案是否正确（regex / exact_match） |
| `tool_call_format` | 0.2 | tool_call JSON 是否合法 |
| `tool_call_efficiency` | 0.1 | 是否在最少步数内完成 |
| `format_reward` | 0.1 | ChatML 闭合性 |
| `length_penalty` | -0.05 | 过长惩罚 |

**注**：task_success 占主导，其它都是 shaping reward。

## 11.5 Loss 计算（Token-Level Mask）

由于轨迹中既有 assistant 生成的 token，也有工具响应注入的 token，**只能在 assistant 区段反传梯度**：

```python
# 标记每个 token 是否由 policy 生成
loss_mask = torch.zeros_like(input_ids)
for assistant_span in find_assistant_spans(input_ids):
    loss_mask[assistant_span.start : assistant_span.end] = 1

# GRPO loss
adv = group_normalize(rewards)
log_ratio = (new_logp - old_logp.detach()) * loss_mask
ratio = exp(log_ratio.clamp(-clip_log, clip_log))
loss = -(ratio * adv.unsqueeze(-1) * loss_mask).sum() / loss_mask.sum()
```

## 11.6 训练循环

```mermaid
sequenceDiagram
    participant DS
    participant Engine as Rollout Engine
    participant Tools
    participant Reward
    participant Policy

    DS->>Engine: prompts + tools (B)
    Engine->>Engine: copy N 份 → B×N
    loop for each rollout
        Engine->>Policy: generate (turn 1)
        Policy-->>Engine: tool_call
        Engine->>Tools: execute
        Tools-->>Engine: tool_response
        Engine->>Policy: generate (turn 2)
        ...
    end
    Engine->>Reward: trajectory + answer_check
    Reward-->>Policy: rewards
    Policy->>Policy: GRPO loss + backward
```

## 11.7 启动命令

```bash
python trainer/train_agent.py \
    --from_weight reason \
    --save_weight agent \
    --data_path .dataset/agent_rl.jsonl \
    --num_samples_per_prompt 8 \
    --max_turns 6 \
    --learning_rate 3e-7 \
    --kl_coef 0.04
```

## 11.8 调试技巧

- **先离线验证 reward 函数**：构造若干 (trajectory, expected_reward) 单元测试
- **从短 max_turns 起步**：先 max_turns=2 收敛后再放开
- **冷启动配方**：建议 `pretrain → SFT → DPO → GRPO → Agent RL`，跳级容易崩
- **温度调度**：训练前期 T=1.0~1.2，后期降到 0.7
- **Tool 失败 fallback**：工具异常返回 `<tool_response>{"error": "..."}` 而非中断 rollout

## 11.9 已知限制

1. **本地 tool sandbox 简化**：`safe_eval` 仅支持基础数学，不支持文件 IO/网络
2. **多轮 rollout 显存大**：建议 `num_samples_per_prompt ≤ 4`
3. **没有 retrieval-augmented**：未集成 vector DB，需自行扩展工具
4. **不支持并行多 agent**：单进程串行执行 tool

## 11.10 与业界框架对比

| 框架 | 特性 | MiniMind 对应 |
|------|------|--------------|
| OpenAI Function Calling | API 级 | ChatML `<tool_call>` 标签 |
| LangGraph | 状态机调度 | RolloutEngine 内联循环 |
| AutoGPT / ReAct | 思考-行动循环 | ``
- **`tool_calls` 字段**：自动解析 `<tool_call>{...}</tool_call>` → OpenAI Function Calling 格式
- **`tools` 参数注入**：传入 OpenAI 风格 tools，自动渲染到 chat_template

### 12.4.4 解析逻辑

`parse_response(text)` 把模型输出拆为三段：

```python
def parse_response(text):
    reasoning = re.search(r"", text, re.DOTALL)
    tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    content = re.sub(r"<think>.*?</think>|<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    return content.strip(), reasoning, parsed_tool_calls
```

## 12.5 Streamlit Web Demo

```bash
streamlit run scripts/web_demo.py -- \
    --weight full_sft --hidden_size 512
```

提供：
- 对话历史保留
- 温度 / top_p / max_tokens 滑块
- 思考内容折叠显示
- Tool Call 高亮
- 模型切换（Dense / MoE / LoRA）

## 12.6 极简 SDK：`chat_api.py`

```python
from scripts.chat_api import ChatClient

client = ChatClient(weight="full_sft", hidden_size=512, device="cuda")
reply = client.chat([{"role": "user", "content": "讲个笑话"}])
print(reply)
```

适合嵌入到自动化脚本、unit test。内部封装：tokenizer + model + chat_template + 简易 history 管理。

## 12.7 权重格式转换：`convert_model.py`

支持三类操作：

### 12.7.1 PyTorch → HuggingFace 格式

```bash
python scripts/convert_model.py --to_hf \
    --weight full_sft --hidden_size 512 \
    --output_dir ./hf_minimind
```

输出标准 HuggingFace 目录（含 `config.json`、`pytorch_model.bin`、`tokenizer.json`），可直接 `AutoModelForCausalLM.from_pretrained` 加载，并上传 HF Hub。

### 12.7.2 HuggingFace → PyTorch

```bash
python scripts/convert_model.py --from_hf \
    --hf_dir jingyaogong/MiniMind3 \
    --output_path ./out/from_hf_512.pth
```

### 12.7.3 LoRA 合并（详见 `07-training-lora.md`）

```bash
python scripts/convert_model.py --merge_lora \
    --base_weight full_sft --lora_weight lora_medical \
    --output_path ./out/merged.pth
```

## 12.8 推理性能优化建议

| 优化 | 实现方式 | 预期加速 |
|------|---------|---------|
| Flash Attention | `lm_config.flash_attn=True`（默认开） | 2-3× |
| KV Cache | 默认开（`use_cache=True`） | 长文本必备 |
| `torch.compile` | `model = torch.compile(model)` | 1.3-1.8×（PyTorch ≥ 2.1） |
| bf16 推理 | `model.to(dtype=torch.bfloat16)` | 1.5×（A100/H100） |
| llama.cpp / GGUF | 用 `convert_model.py --to_hf` 后转 GGUF | CPU 友好 |
| vllm | HF 格式直接被 vllm 加载 | 4-10× 吞吐 |

## 12.9 长文本推理（YaRN）

```bash
python eval_llm.py --weight full_sft \
    --inference_rope_scaling \
    --max_new_tokens 30000
```

启用后 `MiniMindConfig.rope_scaling = {"type":"yarn","factor":16,...}`，
有效上下文从 2048 扩展到 32768，无需重训。详见 [03 - 模型架构](./03-model-architecture.md) §3.4.2。

## 12.10 多模态预留

虽然主线模型为纯文本，但 chat_template 已为图文混排预留位置。后续如要扩展 VLM，可参考 README 中提到的 `MiniMind-V` 路线。
