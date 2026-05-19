# 12 - 推理与服务部署：让模型「毕业上班」

> 对应代码：`eval_llm.py` + `scripts/serve_openai_api.py` + `scripts/chat_api.py` + `scripts/web_demo.py` + `scripts/convert_model.py`

训练模型就像培养一个高材生，但只有让它「毕业上班」、真正为人服务，才算完成了闭环。本章介绍如何让 MiniMind3 从实验室走向生产环境，通过标准化的接口、高效的缓存机制和通用的格式转换，让模型在各种平台上稳定工作。

## 12.1 推理入口总览：模型毕业后的多种工作岗位

| 入口 | 场景 | 说明 |
|------|------|------|
| `eval_llm.py` | 命令行交互 / 评测 | 单进程，开发调试用 |
| `scripts/chat_api.py` | 简单 Python API | 嵌入到自己代码里 |
| `scripts/serve_openai_api.py` | **OpenAI 兼容 HTTP 服务** | 生产/工具集成首选 |
| `scripts/web_demo.py` | Streamlit 网页 | 演示 / 用户体验 |
| `scripts/convert_model.py` | 权重转换 | torch → HF / GGUF / LoRA 合并 |

## 12.2 模型加载流程：新员工入职手续

所有入口共用同一份初始化流程，就像为新员工办理入职：准备工牌（Tokenizer）、分配工位（Model）、发放制服（Weights），如果有特殊技能（LoRA）还要额外登记。

```python
def init_model(lm_config, weight, lora_weight=None, device=None):
    tokenizer = AutoTokenizer.from_pretrained("./model")
    model = MiniMindForCausalLM(lm_config)
    state = torch.load(f"./out/{weight}_{hidden}{_moe}.pth", map_location=device)
    model.load_state_dict(state, strict=False)
    if lora_weight:
        apply_lora(model)
        load_lora(model, f"./out/{lora_weight}_{hidden}.pth")
    return model.to(device).eval(), tokenizer
```

要点：
- **权重文件命名约定**：`{weight}_{hidden_size}{'_moe' if moe else ''}.pth`
- **LoRA 二次注入**：base 加载后再 apply_lora
- 自动适配 cuda / mps / cpu

## 12.3 `eval_llm.py`：交互式评测

```bash
# 单轮无历史（评测用）
python eval_llm.py \
    --weight full_sft \
    --hidden_size 512 --num_hidden_layers 8 \
    --history_cnt 0 --temperature 0.7

# 带历史的多轮聊天
python eval_llm.py \
    --weight rlhf \
    --history_cnt 6 --enable_thinking 1
```

主要参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--weight` | full_sft | 加载哪份权重 |
| `--lora_weight` | None | 可选 LoRA |
| `--history_cnt` | 0 | 保留历史轮数（0=单轮评测） |
| `--temperature` | 0.85 | 采样温度 |
| `--top_p` | 0.85 | nucleus sampling |
| `--max_new_tokens` | 8192 | 单轮最大生成 |
| `--enable_thinking` | 0 | 是否启用 `` 思考 |
| `--use_yarn` | 0 | 是否启用 YaRN 长上下文 |

支持 **流式输出**（TextStreamer）和**非流式**两种模式。

## 12.4 `serve_openai_api.py`：标准化的「服务窗口」

这是最重要的部署入口。想象一下，如果每个模型都有自己的「方言」，那接入方就得学无数种语言。OpenAI 兼容 API 就像 **USB 接口标准**——不管你是键盘、鼠标还是 U 盘，只要符合 USB 规范，插上去就能用。基于 FastAPI 实现，**接口完全兼容 OpenAI Chat Completions API**，可被以下生态直接接入：

- LobeChat / NextChat / Open-WebUI
- LangChain / LlamaIndex
- 各种 Function Calling Agent 框架

### 12.4.1 启动

```bash
python scripts/serve_openai_api.py \
    --weight rlhf --hidden_size 512 \
    --host 0.0.0.0 --port 8998
```

### 12.4.2 调用示例

```bash
curl http://localhost:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimind",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```

### 12.4.3 支持的能力

| 能力 | 字段 | 说明 |
|------|------|------|
| 流式输出 | `stream: true` | SSE `data: {chunk}\n\n` |
| 思考模式 | `enable_thinking: true`（自定义字段） | 注入 `` 拆到 `reasoning_content` |
| 多轮对话 | `messages: [...]` | 自动 apply_chat_template |
| 采样参数 | `temperature/top_p/max_tokens` | 标准支持 |

### 12.4.4 响应解析（`parse_response`）

模型生成的纯文本通过正则拆分到三个字段：

```python
content, reasoning_content, tool_calls = parse_response(text)

# 返回示例:
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "今天北京天气晴朗",
      "reasoning_content": "用户想查天气，调用工具...",
      "tool_calls": [
        {"id": "...", "type": "function",
         "function": {"name": "get_weather", "arguments": "{\"location\": \"北京\"}"}}
      ]
    }
  }]
}
```

这是 MiniMind3 兼容 OpenAI Reasoning Models（o1 风格）API 的核心。

## 12.5 `chat_api.py`：OpenAI 兼容客户端示例

`scripts/chat_api.py` **不是**本地推理 SDK，而是一个**调用已部署 OpenAI 兼容 API 服务的命令行客户端示例**，依赖官方 `openai` 库：

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-123",
    base_url="http://localhost:11434/v1",  # 指向 serve_openai_api.py 起的服务
)

while True:
    query = input('[Q]: ')
    conversation_history.append({"role": "user", "content": query})
    response = client.chat.completions.create(
        model="minimind-local:latest",
        messages=conversation_history[-(history_messages_num or 1):],
        stream=True,
        temperature=0.8, max_tokens=2048, top_p=0.8,
        # 传给 MiniMind 自定义 chat_template 的 open_thinking 开关
        extra_body={"chat_template_kwargs": {"open_thinking": True},
                    "reasoning_effort": "medium"},
    )
    # 流式渲染：reasoning_content 显示为灰色
    for chunk in response:
        delta = chunk.choices[0].delta
        r = getattr(delta, 'reasoning_content', None) or ""
        c = delta.content or ""
        if r: print(f'\033[90m{r}\033[0m', end="", flush=True)
        if c: print(c, end="", flush=True)
```

要点：
- `extra_body.chat_template_kwargs.open_thinking`：透传给后端 `apply_chat_template`，控制是否启用 `<think>` 思考块
- `reasoning_effort`：透传给后端，由 `serve_openai_api.py` 解析（low/medium/high 影响生成长度）
- 流式响应中 `delta.reasoning_content` 与 `delta.content` 分别承载思考与正文，便于差异化渲染
- **若需要本地 Python 内嵌推理（不起服务）**，直接复用 `eval_llm.py:init_model` + `model.generate(...)` 即可

## 12.6 `web_demo.py`：Streamlit Web UI

```bash
streamlit run scripts/web_demo.py -- --weight rlhf
```

提供：
- 聊天界面（多轮历史）
- 思考模式开关
- 采样参数滑块
- LoRA 切换下拉

适合 demo 与可视化体验。

## 12.7 `convert_model.py`：给模型办「通用工作签证」

训练好的模型如果只待在自家框架里，就像只会说方言的人，很难去其他平台工作。权重转换就是给模型办一张「通用工作签证」——把它翻译成 HuggingFace、GGUF 等通用格式，让它能去 vLLM、llama.cpp、Ollama 等各种平台「打工」。提供 6 个转换函数，**通过修改 `__main__` 中的注释/调用切换功能**（无 argparse）：

| 函数 | 用途 |
|------|------|
| `convert_torch2transformers_minimind(torch_path, out_dir)` | `.pth` → HuggingFace 格式（保留 MiniMind 自定义类，需 `trust_remote_code=True`） |
| `convert_torch2transformers(torch_path, out_dir)` | `.pth` → **Qwen3 / Qwen3-MoE 标准结构**（更通用，可被 vLLM 等直接加载） |
| `convert_transformers2torch(hf_dir, torch_path)` | HF 格式 → `.pth` |
| `convert_merge_base_lora(base, lora, merged)` | 合并 LoRA 到 base 权重，输出独立 `.pth` |
| `convert_jinja_to_json(jinja_path)` | 把 `chat_template.jinja` 转义后打印到 stdout，方便贴入 `tokenizer_config.json` |
| `convert_json_to_jinja(json_path, out_path)` | 反向操作，便于编辑模板 |

### 12.7.1 默认入口（`if __name__ == '__main__'`）

```python
lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
transformers_path = '../minimind-3'
convert_torch2transformers(torch_path, transformers_path)   # 默认转 Qwen3 兼容
```

如要执行其他操作，**取消对应函数调用的注释**（如 `convert_merge_base_lora(...)`）。

### 12.7.2 兼容 Qwen3 生态的关键

`convert_torch2transformers` 把权重映射到 `Qwen3ForCausalLM` / `Qwen3MoeForCausalLM`，由于 MiniMind3 架构本就对齐 Qwen3，**无需算子改写**，只需做：
- 字段映射（`vocab_size / hidden_size / num_kv_heads / rope_theta` 等）
- MoE 时把 per-expert 的 `gate_proj/up_proj/down_proj` 堆叠成 Qwen3MoE 的 `gate_up_proj` / `down_proj` 张量
- 适配 transformers ≥ 5.0 的 `tokenizer_class` / `extra_special_tokens` / `rope_parameters` 字段差异

转换后即可直接被 vLLM / TGI / SGLang 加载：

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("./minimind-3")  # 不需要 trust_remote_code
```

## 12.8 长上下文推理：YaRN

`eval_llm.py` 通过 `--inference_rope_scaling` 开关在加载时启用 YaRN 外推，把有效上下文从训练时的 2048 扩展到 32768：

```python
MiniMindConfig(
    inference_rope_scaling=True,
    # 默认 rope_scaling = {"type": "yarn", "factor": 16,
    #                      "original_max_position_embeddings": 2048, ...}
)
```

YaRN 的实现细节（线性斜坡、attention_factor 等）见 [03 - 模型架构 §3.4.2](./03-model-architecture.md)。

注意：YaRN 是**推理时启用**的，训练时一般保持 2048 短上下文以加速。

## 12.9 设备适配

| 设备 | 推理性能（256 token） | 备注 |
|------|---------------------|------|
| CUDA (A100) | ~1500 tok/s | TF32 + Flash Attn |
| CUDA (3090) | ~800 tok/s | 同上 |
| MPS (M1 Pro) | ~80 tok/s | 关闭 SDPA，手动 attention |
| CPU | ~10 tok/s | 仅适合调试 |

设备类型由 `--device` 参数指定，默认 `cuda`（CPU/MPS 需手动设为 `cpu`/`mps`）。模型加载后 `.half().eval().to(device)`，推理走 fp16。

## 12.10 与第三方推理框架集成：派遣到其他平台工作

通过 `convert_torch2transformers`（**Qwen3 兼容格式**）转换后的权重，就像拿到了「国际通用护照」，可以被以下框架直接加载并「派遣」到各种平台上工作：

- **vLLM**：`from vllm import LLM; llm = LLM(model="./minimind-3")`
- **llama.cpp**：用 llama.cpp 自带 `convert_hf_to_gguf.py` 转 GGUF 后用 `llama-server` 启动
- **Ollama**：编写 Modelfile 引用 GGUF 后 `ollama create minimind -f Modelfile`
- **TGI / SGLang**：直接 `--model-id ./minimind-3` 启动

由于 `convert_torch2transformers` 输出的就是标准 `Qwen3ForCausalLM` / `Qwen3MoeForCausalLM` 结构，**几乎不需要任何适配代码**。

> 如果使用 `convert_torch2transformers_minimind`（保留 MiniMind 自定义类），第三方框架则需要 `trust_remote_code=True` 并能访问到 `model_minimind.py`，兼容性较差，**不推荐用于第三方部署**。
