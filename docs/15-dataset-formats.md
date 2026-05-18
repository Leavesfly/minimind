# 15 - 训练数据格式详解

> 本文档详细说明 `.dataset/` 目录下各训练数据文件的格式规范、使用方式，以及为什么采用这样的数据格式设计。

## 15.1 数据文件总览

| 文件名 | 大小 | 训练阶段 | 格式关键字段 |
|--------|------|---------|-------------|
| `pretrain_t2t_mini.jsonl` | 1.16 GB | 预训练 | `text` |
| `sft_t2t_mini.jsonl` | 1.62 GB | 监督微调 (SFT) | `conversations` |
| `dpo.jsonl` | 51 MB | 偏好对齐 (DPO) | `chosen`, `rejected` |
| `rlaif.jsonl` | 22 MB | 强化学习 (RLAIF) | `conversations` |
| `agent_rl.jsonl` | 78 MB | Agent 强化学习 | `conversations`, `gt` |
| `agent_rl_math.jsonl` | 17 MB | 数学 Agent RL | `conversations`, `gt` |
| `lora_identity.jsonl` | 22 KB | LoRA 身份认知 | `conversations` |
| `lora_medical.jsonl` | 32 MB | LoRA 医疗领域 | `conversations` |

所有文件均为 **JSONL 格式**（每行一个独立的 JSON 对象），这是 LLM 训练领域的事实标准格式。

---

## 15.2 预训练数据：`pretrain_t2t_mini.jsonl`

### 格式定义

```json
{"text": "给我生成一首有关秋天的诗歌。秋日早晨，清风拂面。..."}
```

每行仅包含一个 `text` 字段，是一段连续的纯文本。

### 为什么这样设计

| 设计决策 | 原因 |
|---------|------|
| **单字段 `text`** | 预训练目标是 next-token prediction，不需要区分角色或对话结构，只需连续 token 序列 |
| **多段内容拼接在一起** | 样本内容将多个问答/段落拼接为一个长文本，最大化每个训练样本的信息密度，减少 padding 浪费 |
| **无特殊标记** | `bos`/`eos` 由 `PretrainDataset` 在 tokenize 阶段动态添加，保持原始数据的纯净性 |

### 如何使用

```python
from dataset.lm_dataset import PretrainDataset

dataset = PretrainDataset(
    data_path='.dataset/pretrain_t2t_mini.jsonl',
    tokenizer=tokenizer,
    max_length=512,
    device='mps'  # Apple Silicon 可直接放 GPU
)
```

**处理流程**：`text` → tokenize → 添加 `[bos]...[eos]` → padding 到 `max_length` → labels 中 padding 位置设为 `-100`

---

## 15.3 SFT 数据：`sft_t2t_mini.jsonl`

### 格式定义

```json
{
  "conversations": [
    {"role": "user", "content": "你背后的模型是哪个版本？"},
    {"role": "assistant", "content": "我是由jingyaogong开发的高效小参数AI模型。", "reasoning_content": "好的，用户问..."},
    {"role": "user", "content": "你模型的训练数据来源是什么？"},
    {"role": "assistant", "content": "我的训练数据涵盖多领域..."}
  ]
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `role` | string | ✅ | `user` / `assistant` / `system` |
| `content` | string | ✅ | 消息正文内容 |
| `reasoning_content` | string | ❌ | 思维链内容，会被渲染到 `<think>...</think>` 块中 |
| `tools` | string | ❌ | 工具定义 JSON（仅 `system` 角色使用） |
| `tool_calls` | string | ❌ | 工具调用 JSON（仅 `assistant` 角色使用） |

### 为什么这样设计

| 设计决策 | 原因 |
|---------|------|
| **`conversations` 数组** | 支持任意多轮对话，模型需要学习多轮上下文理解能力 |
| **`reasoning_content` 独立字段** | 将思维链与最终回答分离，方便 chat template 将其渲染到 `<think>` 标签内，支持 DeepSeek-R1 风格的"先想后答" |
| **`tools` 放在 system 消息中** | 遵循 OpenAI Function Calling 规范，工具定义是全局配置，属于系统级信息 |
| **`tool_calls` 放在 assistant 中** | 模型的工具调用行为是 assistant 的输出，逻辑上属于 assistant 的回复 |
| **Label Mask 只标注 assistant** | 训练时只让模型学习生成 assistant 的回复，user/system 部分不计入 loss，避免模型学习"提问"而非"回答" |

### 如何使用

```python
from dataset.lm_dataset import SFTDataset

dataset = SFTDataset(
    jsonl_path='.dataset/sft_t2t_mini.jsonl',
    tokenizer=tokenizer,
    max_length=1024
)
```

**处理流程**：
1. `pre_processing_chat`：20% 概率注入随机 system prompt（增强泛化）
2. `apply_chat_template`：将对话渲染为带特殊标记的文本
3. `post_processing_chat`：80% 概率移除空 `<think>` 块
4. tokenize → padding
5. `_sft_generate_labels`：只保留 `<|im_start|>assistant\n` 到 `<|im_end|>\n` 之间的 token 作为 label

---

## 15.4 DPO 数据：`dpo.jsonl`

### 格式定义

```json
{
  "chosen": [
    {"role": "user", "content": "How would you quantify..."},
    {"role": "assistant", "content": "A strong directorial vision can significantly..."}
  ],
  "rejected": [
    {"role": "user", "content": "How would you quantify..."},
    {"role": "assistant", "content": "A strong directorial vision can also have..."}
  ]
}
```

### 为什么这样设计

| 设计决策 | 原因 |
|---------|------|
| **`chosen` / `rejected` 对** | DPO 算法的核心思想是通过对比"好回答"与"差回答"来对齐人类偏好，必须成对出现 |
| **相同 user prompt** | chosen 和 rejected 必须共享相同的用户输入，只有 assistant 回复不同，才能构成有效的偏好对比 |
| **完整对话格式** | 与 SFT 格式兼容，共用 `apply_chat_template` 处理逻辑，保持一致性 |
| **Loss Mask 只看 assistant** | 偏好判断只针对模型的回答部分，用户输入不参与 DPO loss 计算 |

### 如何使用

```python
from dataset.lm_dataset import DPODataset

dataset = DPODataset(
    file_path='.dataset/dpo.jsonl',
    tokenizer=tokenizer,
    max_length=4096
)
# 返回: {x_chosen, y_chosen, mask_chosen, x_rejected, y_rejected, mask_rejected}
```

---

## 15.5 RLAIF 数据：`rlaif.jsonl`

### 格式定义

```json
{
  "conversations": [
    {"role": "user", "content": "基于以下角色信息完成一段对话..."},
    {"role": "assistant", "content": "张明：嗨，刘琳..."},
    {"role": "user", "content": "基于以上对话提出一个问题。"},
    {"role": "assistant", "content": "这些智能家居产品需要哪些前提条件..."},
    {"role": "user", "content": "请回答这个问题。"},
    {"role": "assistant", "content": ""}
  ]
}
```

**关键特征**：最后一轮 assistant 的 `content` 为空字符串 `""`。

### 为什么这样设计

| 设计决策 | 原因 |
|---------|------|
| **最后一轮 answer 为空** | RLAIF 训练中，模型需要自行 rollout 生成回答，然后通过奖励模型评分来学习。空 answer 表示"这里需要模型来补全" |
| **保留完整的历史对话** | 前面的多轮对话作为 prompt 上下文，模型在此基础上生成最终回答 |
| **复用 SFT 的 conversations 格式** | 统一格式降低数据处理复杂度，同一套 chat template 处理逻辑即可应对 |
| **`thinking_ratio=0.5`** | 数据集加载时 50% 概率开启 thinking，让 RL 同时学会"思考"和"直接回答"两种模式 |

### 如何使用

```python
from dataset.lm_dataset import RLAIFDataset

dataset = RLAIFDataset(
    jsonl_path='.dataset/rlaif.jsonl',
    tokenizer=tokenizer,
    max_length=1024,
    thinking_ratio=0.5
)
# 返回: {'prompt': "完整的 prompt 文本", 'answer': ""}
```

**注意**：`RLAIFDataset` 只取 `conversations[:-1]` 构建 prompt（去掉最后一轮空 answer），并设置 `add_generation_prompt=True` 让模型从 assistant 开头继续生成。

---

## 15.6 Agent RL 数据：`agent_rl.jsonl` / `agent_rl_math.jsonl`

### 格式定义

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "",
      "tools": "[{\"function\": {\"name\": \"calculate_math\", \"description\": \"计算数学表达式的结果\", \"parameters\": {...}}}]"
    },
    {"role": "user", "content": "算算7109*2920"},
    {"role": "assistant", "content": ""}
  ],
  "gt": ["20758280"]
}
```

### 字段说明

| 字段 | 说明 |
|------|------|
| `conversations` | 对话上下文，system 消息中的 `tools` 字段定义可用工具 |
| `gt` | Ground Truth 答案列表，用于奖励计算（如判断数学计算结果是否正确） |

### 为什么这样设计

| 设计决策 | 原因 |
|---------|------|
| **`tools` 以 JSON 字符串存储** | 保持 JSONL 单行格式的兼容性，避免嵌套结构导致解析复杂度 |
| **`gt` 作为顶层字段** | Ground Truth 不是对话的一部分，而是用于外部奖励计算的元信息，应与对话内容分离 |
| **`gt` 是数组** | 某些问题可能有多个正确答案（如同义表达），奖励函数只要匹配任一即可 |
| **最后一轮 assistant 为空** | 与 RLAIF 相同，模型需要通过多轮 rollout（调用工具 → 获取结果 → 继续推理）来生成最终答案 |
| **区分 `agent_rl.jsonl` 和 `agent_rl_math.jsonl`** | 数学计算场景专门分文件，方便独立训练或混合采样 |

### 如何使用

```python
from dataset.lm_dataset import AgentRLDataset

dataset = AgentRLDataset(
    jsonl_path='.dataset/agent_rl_math.jsonl',
    tokenizer=tokenizer,
    max_length=1024
)
# 返回: {'messages': [...], 'tools': [...], 'gt': ["20758280"]}
```

**训练流程**：Rollout Engine 根据 `messages` 和 `tools` 驱动模型多轮交互（生成 tool_call → 执行工具 → 拼接结果 → 继续生成），最终用 `gt` 验证答案正确性计算奖励。

---

## 15.7 LoRA 数据：`lora_identity.jsonl` / `lora_medical.jsonl`

### 格式定义

**身份认知**（`lora_identity.jsonl`）：

```json
{"conversations": [{"role": "user", "content": "你是谁"}, {"role": "assistant", "content": "您好，我是 MiniMind，一个由 Jingyao Gong 发明的..."}]}
```

**医疗领域**（`lora_medical.jsonl`）：

```json
{"conversations": [{"role": "user", "content": "头发稀少细软可以植发吗"}, {"role": "assistant", "content": "是的，头发稀少和细软的情况下，植发是一种可能的解决方案..."}]}
```

### 为什么这样设计

| 设计决策 | 原因 |
|---------|------|
| **与 SFT 格式完全一致** | LoRA 微调复用 `SFTDataset` 加载器，无需额外的数据处理逻辑 |
| **独立小文件** | LoRA 适配器训练数据量小，独立文件方便快速迭代和替换 |
| **身份数据仅 22KB（~90 条）** | 身份认知只需少量高质量样本即可植入，过多会导致过拟合或与其他能力冲突 |
| **领域数据规模适中** | 医疗等垂直领域数据 32MB 足够 LoRA 学习领域知识，同时不会覆盖通用能力 |

### 如何使用

```python
from dataset.lm_dataset import SFTDataset

# LoRA 数据直接用 SFTDataset 加载
dataset = SFTDataset(
    jsonl_path='.dataset/lora_identity.jsonl',
    tokenizer=tokenizer,
    max_length=512
)
```

---

## 15.8 格式设计的统一原则

### 为什么选择 JSONL

1. **流式处理友好**：每行独立解析，可以用 `load_dataset('json')` 流式加载，不需要一次性加载全文件到内存
2. **方便追加**：新数据直接 append 到文件末尾，不影响已有数据
3. **工具生态完善**：`datasets` 库原生支持，`jq`/`grep` 等命令行工具可直接处理
4. **Git 友好**：行级别 diff，便于版本管理和 code review

### 为什么 `conversations` 格式统一

```
预训练：text → 纯序列建模
   ↓ 演化
SFT：conversations → 学习对话格式
   ↓ 复用
DPO：chosen/rejected 各自是 conversations → 偏好对比
RLAIF/Agent：conversations + 空 answer → RL rollout
LoRA：conversations → 领域适配
```

统一的 `conversations` 格式带来以下优势：

- **一套 chat template 全覆盖**：`tokenizer.apply_chat_template()` 适用于所有对话类数据
- **Label Mask 逻辑复用**：`_sft_generate_labels` 函数在 SFT、DPO、LoRA 间共享
- **数据可混合**：不同来源的 SFT 数据可以直接 `cat` 合并，无需格式转换

### 为什么 Label 只标注 assistant 区段

```
[user消息]  [assistant回复]  [user消息]  [assistant回复]
  -100          label           -100          label
```

- 模型的训练目标是学习"如何回答"，不是学习"如何提问"
- 用 `-100` 标记 user/system 部分，PyTorch 的 `CrossEntropyLoss(ignore_index=-100)` 会自动跳过这些位置
- 这确保了模型的 loss 100% 来自生成质量的反馈

---

## 15.9 自定义数据集指南

### 预训练数据

如果你想用自己的语料进行预训练：

```bash
# 每行一个 JSON，包含 text 字段
echo '{"text": "你的语料文本..."}' >> .dataset/my_pretrain.jsonl
```

### SFT 数据

如果你想微调模型的对话能力：

```bash
# 基本格式（单轮）
echo '{"conversations": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}]}' >> .dataset/my_sft.jsonl

# 带思维链
echo '{"conversations": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答", "reasoning_content": "思考过程..."}]}' >> .dataset/my_sft.jsonl

# 带工具调用
echo '{"conversations": [{"role": "system", "content": "", "tools": "[{...}]"}, {"role": "user", "content": "问题"}, {"role": "assistant", "content": "结果", "tool_calls": "[{...}]"}]}' >> .dataset/my_sft.jsonl
```

### DPO 数据

如果你想进行偏好对齐训练：

```bash
# chosen 和 rejected 必须共享相同的 user 输入
echo '{"chosen": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "更好的回答"}], "rejected": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "较差的回答"}]}' >> .dataset/my_dpo.jsonl
```

### RLAIF / Agent RL 数据

```bash
# RLAIF：最后一轮 assistant 为空
echo '{"conversations": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": ""}]}' >> .dataset/my_rlaif.jsonl

# Agent RL：需要 tools 和 gt
echo '{"conversations": [{"role": "system", "content": "", "tools": "[...]"}, {"role": "user", "content": "问题"}, {"role": "assistant", "content": ""}], "gt": ["正确答案"]}' >> .dataset/my_agent.jsonl
```

---

## 15.10 常见问题

**Q: 为什么 `reasoning_content` 不直接写在 `content` 里？**

A: 分离设计让 chat template 可以灵活控制是否显示思考过程。训练时通过 `post_processing_chat` 以 80% 概率移除空思考块，20% 保留，让模型同时学会两种输出风格。

**Q: 为什么 `tools` 字段是 JSON 字符串而不是 JSON 对象？**

A: JSONL 要求每行是一个完整的 JSON 对象。如果 `tools` 直接嵌套为对象，`datasets` 库在推断 schema 时可能因为不同样本的 tools 结构不一致而报错。字符串化后统一为 `Value('string')` 类型，加载后再 `json.loads()` 解析。

**Q: 预训练数据为什么不用对话格式？**

A: 预训练阶段的目标是让模型学习语言建模能力（词与词的概率分布），不需要角色区分。使用纯文本可以最大化训练效率，避免 chat template 标记占用有效 token 长度。

**Q: DPO 数据的 chosen/rejected 可以不对称吗？**

A: 不可以。DPO 的数学公式要求 chosen 和 rejected 共享相同的 prompt 前缀，只有 assistant 回复不同。如果 prompt 不同，loss 计算将失去意义。
