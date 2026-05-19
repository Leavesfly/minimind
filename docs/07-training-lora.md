# 07 - LoRA 微调与权重合并

> **核心类比**：LoRA 就像「给专家戴上不同的眼镜」——不改变专家本人（基础模型），只通过眼镜（低秩矩阵）改变看世界的方式。为什么只需要 2% 的参数就能学会新技能？因为真正的知识早已存在，我们只是调整了观察角度。

> 对应代码：`model/model_lora.py`（189 行）+ `trainer/train_lora.py`（217 行）

## 7.1 LoRA 原理：给专家戴上不同的眼镜

想象你面前有一位精通万物的专家（预训练模型），他的知识已经非常渊博。现在你想让他学会医疗领域的专业知识，有两种做法：

1. **传统做法**：让专家重新学习所有知识，把原来的记忆覆盖掉一部分——风险大、成本高、容易遗忘旧知识。
2. **LoRA 做法**：给专家戴上一副"医疗眼镜"。专家本人不变，只是通过眼镜的镜片调整他看问题的角度。摘下眼镜，他还是原来的专家；戴上眼镜，他就成了医疗专家。

**为什么只需要 2% 的参数？** 因为真正的知识早已存在，我们不需要重新学习，只需要学会"如何调整视角"。

### 技术实现：低秩分解 = 用两张小纸条代替一整页笔记

LoRA（Low-Rank Adaptation）的核心思想是：**冻结预训练权重 `W ∈ R^(d×k)`，并在旁路注入两个低秩矩阵 `A ∈ R^(r×k)` 和 `B ∈ R^(d×r)` 来学习增量更新**。

```
ΔW = B @ A              # rank-r 分解：用两个小矩阵的乘积代替大矩阵
W' = W + ΔW             # 新权重 = 原权重 + 增量
y  = W'x = Wx + BAx     # 输出 = 原输出 + 调整量
```

**类比解释**：
- **冻结权重 W**：把教科书锁在柜子里，保护原有知识不被覆盖
- **低秩矩阵 A 和 B**：用两张小纸条（维度 r 很小）记录关键变化，代替重写整页笔记
- **参数量从 `d×k` 降到 `(d+k)×r`**：当 `r << min(d,k)` 时，存储成本大幅下降

### 初始化策略：保证初始状态等价于原模型

| 矩阵 | 初始化 | 原因 |
|------|--------|------|
| A | `N(0, 0.02²)` | 高斯小值，提供训练信号 |
| B | 全 0 | 保证 `ΔW = 0`，初始等价于原模型（刚戴上眼镜时，视野不变） |

## 7.2 MiniMind 的 LoRA 实现：把眼镜装到镜框上

`model/model_lora.py:LoRA` 是一个简洁的 `nn.Module`，它定义了"眼镜"的结构：

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(0.0, 0.02)
        self.B.weight.data.zero_()
    def forward(self, x):
        return self.B(self.A(x))
```

### 7.2.1 注入策略：`apply_lora(model, rank=16)` —— 给合适的部位戴眼镜

**只对方阵 Linear（`in == out`）注入**，避免破坏 attention 结构中的非方阵投影。就像只给需要调整视力的眼睛配眼镜，而不是给全身都戴上。

```python
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and \
       module.weight.shape[0] == module.weight.shape[1]:
        lora = LoRA(...).to(model.device)
        setattr(module, "lora", lora)
        original_forward = module.forward
        def forward_with_lora(x, layer1=original_forward, layer2=lora):
            return layer1(x) + layer2(x)
        module.forward = forward_with_lora
```

**关键点**：通过 monkey-patch `module.forward` 实现 `y = Wx + BAx`，使用 **闭包默认参数绑定**（`layer1=original_forward`）避免 Python 闭包的常见 bug。

### 7.2.2 保存与加载：把眼镜度数焊死到镜框上

| 函数 | 行为 |
|------|------|
| `save_lora(model, path)` | 仅保存所有 `module.lora` 的权重，转 fp16 节省空间（只存眼镜，不存专家） |
| `load_lora(model, path)` | 把权重按层级 key 还原到对应 LoRA 模块（给专家换一副新眼镜） |
| `merge_lora(model, lora_path, save_path)` | 把 `BA` 累加到原始 `W`，输出无 LoRA 的完整权重（把眼镜度数焊死到镜框上，永久生效） |

合并时的关键代码：

```python
state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
if hasattr(module, 'lora'):
    state_dict[f'{name}.weight'] += (module.lora.B.weight.data
                                     @ module.lora.A.weight.data).cpu().half()
```

合并后模型与普通模型完全等价，可直接被 `vllm`、`llama.cpp`、`ollama` 等推理框架加载。**此时不再需要单独带眼镜，因为度数已经永久融合。**

## 7.3 LoRA 训练流程：只训练眼镜，不训练专家

`trainer/train_lora.py` 与 `train_full_sft.py` 高度同构，只在 4 处有特殊处理。核心思想：**冻结专家（基础模型），只优化眼镜（LoRA 参数）**。

### 7.3.1 注入与冻结：把教科书锁进柜子

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
apply_lora(model)

lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True   # 只有眼镜可以调整
        lora_params.append(param)
    else:
        param.requires_grad = False  # 专家的知识被锁定，不可修改
```

### 7.3.2 优化器只优化 LoRA 参数：只给眼镜调焦

```python
optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
```

### 7.3.3 梯度裁剪只针对 LoRA 参数：防止眼镜度数调过头

```python
torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
```

### 7.3.4 保存只保存 LoRA 权重：只存眼镜，不存专家

```python
save_lora(model, f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth')
```

文件名约定：`lora_{name}_{hidden_size}.pth`（例如 `lora_medical_512.pth`），通常 < 10 MB。**因为只存了眼镜的度数，没存整个专家的大脑。**

## 7.4 启动命令：开始训练你的第一副眼镜

```bash
# 基础训练（在 full_sft 上做医疗领域 LoRA）
python trainer/train_lora.py \
    --from_weight full_sft \
    --lora_name lora_medical \
    --data_path .dataset/lora_medical.jsonl \
    --learning_rate 1e-4 --epochs 10

# 训练自我认知 LoRA
python trainer/train_lora.py \
    --lora_name lora_identity \
    --data_path .dataset/lora_identity.jsonl
```

## 7.5 推理时加载 LoRA：给专家换上一副新眼镜

`eval_llm.py` 与 `scripts/serve_openai_api.py` 都支持 LoRA：

```bash
python eval_llm.py \
    --weight full_sft \
    --lora_weight lora_medical \
    --hidden_size 512
```

底层逻辑：先加载 base 权重（请出专家），再 `apply_lora(model)` 注入空 LoRA（准备好镜框），最后 `load_lora` 装载训练好的权重（戴上医疗眼镜）。**此时，专家就变成了医疗专家。**

## 7.6 权重合并：脱离 LoRA 代码部署 —— 把眼镜度数焊死到镜框上

```bash
python scripts/convert_model.py --merge_lora \
    --base_weight full_sft --lora_weight lora_medical \
    --output_path ./out/full_sft_medical_merged.pth
```

合并后部署到第三方推理框架（vllm / llama.cpp / ollama）时**无需带上 LoRA 代码**，与普通模型完全一致。**此时，眼镜已经不再是可拆卸的配件，而是永久成为了镜框的一部分。专家不再需要"戴眼镜"这个动作，因为他本身就已经是医疗专家了。**

## 7.7 参数效率分析：为什么 2% 的参数就够用？

```
hidden_size = 512, rank = 16
方阵 Linear（仅 attention 内部 Q/K/V/O 中可能为 hidden×hidden 的投影）

总参数量      ≈ 26M
LoRA 参数量   ≈ (512+512) × 16 × N_lora_layers ≈ 0.5M
LoRA 占比     ≈ 1.92%
```

**第一性原理解释**：预训练模型已经学会了"如何思考"，它拥有通用的推理能力、语言理解和世界知识。领域微调不是让模型重新学习这些基础能力，而是让它学会"用专业的视角看问题"。

就像一位通才医生转专科医生，他不需要重新学习解剖学、生理学，只需要掌握专科特有的诊断思路和治疗方案。LoRA 的低秩矩阵 `A` 和 `B` 就是这副"专业眼镜"，它们记录了从通用视角到专业视角的映射关系。

**为什么低秩就够了？** 因为领域知识的增量变化通常是结构化的、有规律的，而不是随机的。用数学语言说，领域适配的权重更新 `ΔW` 往往位于一个低维子空间中。用两张小纸条（低秩分解）就能捕捉到这种结构化变化，无需重写整本教科书。

实际占比 < 2%，但能在领域微调上取得接近全量 SFT 的效果。

## 7.8 与 peft 的对比：极简主义 vs 工业级工具

| 特性 | MiniMind LoRA | HuggingFace peft |
|------|--------------|------------------|
| 实现复杂度 | < 200 行 | 数千行 |
| 注入粒度 | 方阵 Linear（自动检测） | 可配置 target_modules |
| α 缩放因子 | 无（固定 1.0） | 可配置 α/r |
| Dropout | 无 | 可配 |
| QLoRA / DoRA | 不支持 | 支持 |

**设计取舍**：MiniMind 的 LoRA 极简，适合学习与小模型微调，工业级场景仍推荐 peft。**就像手工打造的定制眼镜 vs 工业化生产的眼镜连锁店——前者让你理解原理，后者提供更多选择和优化。**
