# 04 - Tokenizer 与 Chat Template

> 对应代码：`model/tokenizer_config.json`（含 chat_template Jinja）+ `trainer/train_tokenizer.py` + `dataset/lm_dataset.py:pre/post_processing_chat`

## 4.1 Tokenizer：模型的翻译词典

**类比理解**：Tokenizer 就像一本"翻译词典"，它的作用是把人类能读懂的自然语言（文字）翻译成模型能理解的数字序列。想象一下，如果你要和只会说数字的外星人交流，你需要一本词典，把每个词都对应到一个数字编号。Tokenizer 就是这本词典——它决定了如何把文本切分成最小的意义单元（token），并为每个单元分配一个唯一的数字 ID。模型实际上只认识这些数字，不认识任何文字。

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

## 4.2 训练自己的 Tokenizer：编纂自己的词典

**为什么需要训练自己的 Tokenizer？** 就像不同领域有不同的专业术语一样，通用词典可能无法很好地处理特定领域的文本。训练自己的 Tokenizer 就是为你的应用场景"量身定制"一本词典，让它更懂你的数据。比如医疗领域有很多专业词汇，法律领域有特定的表达方式，通用的分词器可能把这些词拆得太碎，导致模型理解困难。自己训练的 Tokenizer 能更好地捕捉你数据中的常见模式。

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

## 4.3 Chat Template：对话的信封格式

**类比理解**：Chat Template 就像"信封的格式规范"。想象你要寄一封信，信封上必须清楚地标明：寄件人是谁、收件人是谁、信件内容是什么。如果没有这个格式，邮局（模型）就不知道这封信是谁写的、该回复给谁。Chat Template 的作用就是把结构化的对话数据（[{role: "user", content: "..."}, {role: "assistant", content: "..."}]）转换成模型能理解的单一字符串格式，让模型清楚地知道：哪段话是用户说的、哪段是助手说的、什么时候该开始回答。这样模型才能正确地学习对话的逻辑。

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

## 4.4 训练阶段的预/后处理：备菜环节的小技巧

**类比理解**：训练前的数据处理就像厨师做菜前的"备菜"环节。好的备菜能让烹饪过程更顺利，做出来的菜更好吃。这里的预处理和后处理就是在正式训练前对数据进行一些"调味"和"整理"，让模型学到更鲁棒、更灵活的能力。比如随机添加 system prompt 就像偶尔换个菜谱开头，让模型适应不同的开场方式；移除空思考块就像去掉不需要的食材，让模型学会什么时候该思考、什么时候直接回答。

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

## 4.5 SFT 标签的 Mask 生成：老师只批改答案部分

**类比理解**：在监督微调（SFT）训练中，我们只希望模型学习如何回答用户的问题，而不是学习用户问了什么。这就像老师批改作业时，只给学生写的答案部分打分，而不会去"学习"题目本身。Mask 生成的作用就是告诉模型："这些位置（assistant 的回答）是你需要学习的，那些位置（用户的提问、system 指令）你可以忽略"。通过把非 assistant 部分的 label 设为 -100，损失函数会自动跳过这些位置，只计算 assistant 回答部分的误差。这样模型就能专注于学习"如何回答问题"，而不是记住"用户都问了什么"。

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

## 4.6 推理时的 ChatML 解析：拆信读内容

**类比理解**：当模型生成回答后，输出的是一段包含各种标记的纯文本（ChatML 格式）。推理时的解析就像"拆开信封读取信件内容"——我们需要从这段文本中提取出有用的信息：模型的思考过程是什么？有没有调用工具？最终的回答是什么？这个解析过程把模型输出的原始文本拆解成结构化的数据，以便返回给前端或 API 调用方。这是实现 OpenAI 兼容接口的关键步骤，让用户能以标准的格式获取 reasoning_content（思考内容）、tool_calls（工具调用）和 content（最终回答）。

`scripts/serve_openai_api.py:parse_response` 把模型生成的纯文本拆成三段，再以 OpenAI 兼容格式返回：

| 模型输出片段 | 字段 |
|------------|------|
| `<think>...</think>` | `reasoning_content` |
| `<tool_call>{...}</tool_call>` | `tool_calls`（OpenAI Function Calling 格式） |
| 其余 | `content` |

这是实现 OpenAI 兼容 `reasoning_content` / `tool_calls` 的关键。

## 4.7 ChatML Prompt 反向解析：从信件还原对话

**类比理解**：有时候我们需要把已经格式化好的 ChatML 字符串"逆向工程"回原始的对话结构。这就像收到一封已经按标准格式写好的信，我们需要从中提取出：谁是寄件人、谁是收件人、每段内容分别是谁说的。在强化学习（RL）训练中，这个功能特别有用——我们需要把 prompt 还原成 [{role, content}] 的列表格式，然后送给 Reward Model 进行评分。反向解析让我们能从任何 ChatML 格式的文本中恢复出原始的对话结构。

`trainer/reward_utils.py:parse_messages_from_chat_prompt` 提供反向能力：把 ChatML 字符串解析回 `[{role, content}]` 列表。在 RL 训练中用于把 prompt 还原后送给 Reward Model 评分。

```python
_CHAT_MESSAGE_PATTERN = re.compile(
    r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>", re.DOTALL)
```
