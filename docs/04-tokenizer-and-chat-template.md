# 04 - Tokenizer 与 Chat Template

> 对应代码：`model/tokenizer_config.json`（含 chat_template Jinja）+ `trainer/train_tokenizer.py` + `dataset/lm_dataset.py:pre/post_processing_chat`

## 4.1 Tokenizer 概览

MiniMind3 自训了一个 **BPE + ByteLevel** 分词器，词表大小仅 **6400**，专为中英文混合的小模型场景设计。

### 4.1.1 关键属性

| 项 | 值 |
|----|----|
| 类型 | `PreTrainedTokenizerFast` |
| 算法 | BPE (Byte-Pair Encoding) + ByteLevel pre-tokenizer |
| 词表大小 | 6400 |
| `bos_token` | `<|im_start|>`（id=1） |
| `eos_token` | `<|im_end|>`（id=2） |
| `pad_token` | `<|endoftext|>`（id=0） |

### 4.1.2 特殊 Token

| 标记 | 用途 |
|------|------|
| `<|im_start|>` / `<|im_end|>` | ChatML 风格的对话边界 |
| `` | 思考块边界 |
| `<tool_call>` / `</tool_call>` | 工具调用块 |
| `<tool_response>` / `</tool_response>` | 工具响应块 |
| Buffer Tokens | 预留若干 token 便于后续扩展 |

> **设计权衡**：词表越小，Embedding 矩阵越紧凑（6400×768 ≈ 4.9M 参数），但同样的句子需要更多 token 来表示。MiniMind 的取舍是**牺牲压缩率换取参数效率**，符合"小模型"定位。

## 4.2 训练自己的 Tokenizer

参见 `trainer/train_tokenizer.py`：

```bash
python trainer/train_tokenizer.py
```

该脚本基于 HuggingFace `tokenizers` 库训练，主要步骤：
1. 从语料 JSONL 中流式读取 `text` 字段
2. 配置 ByteLevel pre-tokenizer + BPE trainer
3. 注入特殊 token
4. 输出 `tokenizer.json` 和 `tokenizer_config.json` 到 `model/` 目录

如需扩词表（例如 32K），修改其中的 `vocab_size` 后重训。注意：**改词表后需要重新预训练**，否则 Embedding 不匹配。

## 4.3 Chat Template

Chat Template 定义在 `model/tokenizer_config.json` 的 `chat_template` 字段，是一段 **Jinja2 模板**。它负责把 `[{role, content, ...}]` 列表渲染为单个字符串。MiniMind3 模板支持：

- `system / user / assistant / tool` 多角色
- `tools` 函数定义注入
- `<tool_call>` 与 `<tool_response>` 标签
- `open_thinking` 开关：决定是否在 `assistant` 角色之后注入空 `<think>` 占位

### 4.3.1 普通对话

```python
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "你好"}],
    tokenize=False, add_generation_prompt=True)
```

输出：

```
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant

```

### 4.3.2 自适应思考（open_thinking）

```python
prompt = tok.apply_chat_template(msgs, tokenize=False,
                                 add_generation_prompt=True,
                                 open_thinking=True)
```

会在 `assistant` 角色之后追加 `<think>\n`，模型必须先输出思考内容，再用 `</think>` 结束，进入正式回答。

### 4.3.3 Tool Call

```python
tools = [{"type": "function", "function": {
    "name": "get_current_weather",
    "description": "获取天气",
    "parameters": {"type": "object",
                   "properties": {"location": {"type": "string"}},
                   "required": ["location"]}}}]
prompt = tok.apply_chat_template(msgs, tokenize=False,
                                 add_generation_prompt=True,
                                 tools=tools)
```

模型生成的 Tool Call 形如：

```
<tool_call>
{"name": "get_current_weather", "arguments": {"location": "北京"}}
</tool_call>
```

工具结果以 `tool` 角色回填：

```
<|im_start|>tool
<tool_response>
{"city": "北京", "temperature": "28°C"}
</tool_response>
<|im_end|>
```

## 4.4 训练阶段的预/后处理

`dataset/lm_dataset.py` 中的两个关键函数：

### `pre_processing_chat(conversations, add_system_ratio=0.2)`

- 若首条不是 `system`，按 **20% 概率**随机插入一个中英文 system prompt（提升对 system 指令的鲁棒性）
- 含 `tools` 字段时跳过 system 注入，保留原始结构

```python
SYSTEM_PROMPTS = ["你是一个知识丰富的AI...", "You are a helpful AI assistant.", ...]
```

### `post_processing_chat(prompt_content, empty_think_ratio=0.2)`

- 训练时遇到 `<think>\n\n</think>\n\n` 的空思考块，以 **80% 概率移除**
- 这样模型既能学到"思考"也能学到"不思考"，从而支持 `open_thinking` 自适应开关

## 4.5 SFT 标签的 Mask 生成

`_sft_generate_labels` 通过查找两个 marker：

```
bos_marker = tok("<|im_start|>assistant\n").input_ids
eos_marker = tok("<|im_end|>\n").input_ids
```

只把 **assistant 区段**标记为可学习，其它位置（system、user、工具响应）全部置为 `-100`，CE loss 自动忽略：

```
labels = [-100] * len(input_ids)
# 扫描定位每个 <|im_start|>assistant\n ... <|im_end|>\n 区段
# 把这部分 token id 写回 labels
```

> 这是与 `trl.DataCollatorForCompletionOnlyLM` 的关键不同：MiniMind 在 Dataset 阶段**静态生成**好 labels，省去训练时反复扫描的开销。

## 4.6 推理时的 ChatML 解析

`scripts/serve_openai_api.py:parse_response` 把模型生成的纯文本拆成三段，再以 OpenAI 兼容格式返回：

| 模型输出片段 | 字段 |
|------------|------|
| `<think>...</think>` | `reasoning_content` |
| `<tool_call>{...}</tool_call>` | `tool_calls`（OpenAI Function Calling 格式） |
| 其余 | `content` |

这是实现 OpenAI 兼容 `reasoning_content` / `tool_calls` 的关键。

## 4.7 ChatML Prompt 反向解析

`trainer/reward_utils.py:parse_messages_from_chat_prompt` 提供反向能力：把 ChatML 字符串解析回 `[{role, content}]` 列表。在 RL 训练中用于把 prompt 还原后送给 Reward Model 评分。

```python
_CHAT_MESSAGE_PATTERN = re.compile(
    r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>", re.DOTALL)
```
