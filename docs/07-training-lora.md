# 07 - LoRA 微调与权重合并

> 对应代码：`model/model_lora.py`（189 行）+ `trainer/train_lora.py`（217 行）

## 7.1 LoRA 原理

LoRA（Low-Rank Adaptation）通过冻结预训练权重 `W ∈ R^(d×k)`，并在旁路注入两个低秩矩阵 `A ∈ R^(r×k)` 和 `B ∈ R^(d×r)` 来学习增量更新：

```
ΔW = B @ A              # rank-r 分解
W' = W + ΔW
y  = W'x = Wx + BAx
```

参数量从 `d×k` 降到 `(d+k)×r`，当 `r << min(d,k)` 时大幅减少。

### 初始化策略

| 矩阵 | 初始化 | 原因 |
|------|--------|------|
| A | `N(0, 0.02²)` | 高斯小值，提供训练信号 |
| B | 全 0 | 保证 `ΔW = 0`，初始等价于原模型 |

## 7.2 MiniMind 的 LoRA 实现

`model/model_lora.py:LoRA` 是一个简洁的 `nn.Module`：

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

### 7.2.1 注入策略：`apply_lora(model, rank=16)`

**只对方阵 Linear（`in == out`）注入**，避免破坏 attention 结构中的非方阵投影：

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

### 7.2.2 保存与加载

| 函数 | 行为 |
|------|------|
| `save_lora(model, path)` | 仅保存所有 `module.lora` 的权重，转 fp16 节省空间 |
| `load_lora(model, path)` | 把权重按层级 key 还原到对应 LoRA 模块 |
| `merge_lora(model, lora_path, save_path)` | 把 `BA` 累加到原始 `W`，输出无 LoRA 的完整权重 |

合并时的关键代码：

```python
state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
if hasattr(module, 'lora'):
    state_dict[f'{name}.weight'] += (module.lora.B.weight.data
                                     @ module.lora.A.weight.data).cpu().half()
```

合并后模型与普通模型完全等价，可直接被 `vllm`、`llama.cpp`、`ollama` 等推理框架加载。

## 7.3 LoRA 训练流程

`trainer/train_lora.py` 与 `train_full_sft.py` 高度同构，只在 4 处有特殊处理：

### 7.3.1 注入与冻结

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
apply_lora(model)

lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True
        lora_params.append(param)
    else:
        param.requires_grad = False
```

### 7.3.2 优化器只优化 LoRA 参数

```python
optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
```

### 7.3.3 梯度裁剪只针对 LoRA 参数

```python
torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
```

### 7.3.4 保存只保存 LoRA 权重

```python
save_lora(model, f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth')
```

文件名约定：`lora_{name}_{hidden_size}.pth`（例如 `lora_medical_512.pth`），通常 < 10 MB。

## 7.4 启动命令

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

## 7.5 推理时加载 LoRA

`eval_llm.py` 与 `scripts/serve_openai_api.py` 都支持 LoRA：

```bash
python eval_llm.py \
    --weight full_sft \
    --lora_weight lora_medical \
    --hidden_size 512
```

底层逻辑：先加载 base 权重，再 `apply_lora(model)` 注入空 LoRA，最后 `load_lora` 装载训练好的权重。

## 7.6 权重合并：脱离 LoRA 代码部署

```bash
python scripts/convert_model.py --merge_lora \
    --base_weight full_sft --lora_weight lora_medical \
    --output_path ./out/full_sft_medical_merged.pth
```

合并后部署到第三方推理框架（vllm / llama.cpp / ollama）时**无需带上 LoRA 代码**，与普通模型完全一致。

## 7.7 参数效率分析

```
hidden_size = 512, rank = 16
方阵 Linear（仅 attention 内部 Q/K/V/O 中可能为 hidden×hidden 的投影）

总参数量      ≈ 26M
LoRA 参数量   ≈ (512+512) × 16 × N_lora_layers ≈ 0.5M
LoRA 占比     ≈ 1.92%
```

实际占比 < 2%，但能在领域微调上取得接近全量 SFT 的效果。

## 7.8 与 peft 的对比

| 特性 | MiniMind LoRA | HuggingFace peft |
|------|--------------|------------------|
| 实现复杂度 | < 200 行 | 数千行 |
| 注入粒度 | 方阵 Linear（自动检测） | 可配置 target_modules |
| α 缩放因子 | 无（固定 1.0） | 可配置 α/r |
| Dropout | 无 | 可配 |
| QLoRA / DoRA | 不支持 | 支持 |

**设计取舍**：MiniMind 的 LoRA 极简，适合学习与小模型微调，工业级场景仍推荐 peft。
