# 14 - 训练工具链：所有课程共享的「教室设备」

> 对应代码：`trainer/trainer_utils.py`

## 14.1 概览：为什么需要统一的工具链？

想象一下，如果每堂课都要重新搬黑板、架投影仪、调时钟，那教学效率会有多低？`trainer_utils.py` 就是 MiniMind3 所有训练脚本（`train_pretrain / full_sft / lora / dpo / distillation / ppo / grpo / agent`）的**共享教室设备**，统一提供：

- **分布式协调**（DDP）：让多个教室同步上课
- **混合精度支持**（AMP）：自动调节计算精度
- **学习率调度**：控制教学节奏
- **状态保存与恢复**：自动存档的游戏机制
- **断点续训**：从书签处继续阅读
- **训练日志格式化**：显示 ETA、tok/s、进度条
- **设备能力探测**：自动识别 CUDA / MPS / CPU

有了这套基础设施，8 个不同的训练脚本就能专注于各自的"课程内容"（loss 计算和数据处理），而不必重复搭建基础环境。

## 14.2 分布式初始化：多教室同步上课

### 14.2.1 `is_ddp() / get_ddp_info()`

就像判断今天是否有多个教室同时开课一样，这两个函数检测当前是否处于分布式训练模式：

```python
def is_ddp():
    return int(os.environ.get("RANK", -1)) != -1

def get_ddp_info():
    if not is_ddp():
        return 0, 0, 1   # rank, local_rank, world_size
    return (int(os.environ["RANK"]),
            int(os.environ["LOCAL_RANK"]),
            int(os.environ["WORLD_SIZE"]))
```

- `rank`：当前是第几个教室（从 0 开始编号）
- `local_rank`：在当前机器上是第几块 GPU
- `world_size`：总共有多少个教室在同步上课

### 14.2.2 `init_distributed_mode()`

这个函数负责把所有教室连接起来，确保大家步调一致：

```python
def init_distributed_mode():
    if is_ddp():
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
```

`destroy_distributed_mode()` 在下课（脚本退出）时调用 `dist.destroy_process_group()`，释放资源。

### 14.2.3 启动多教室模式

```bash
torchrun --nproc_per_node=8 trainer/train_pretrain.py [args]
```

只有 `rank == 0`（主教室）才负责打印日志、写 wandb、保存权重，其余 rank（分教室）静默工作，避免输出混乱。这就像只有一个讲台，只有主讲老师能发言。

## 14.3 设备能力探测：`get_device_info()`

就像进教室前先看有哪些设备可用一样，这个函数自动检测当前环境的计算能力：

```python
def get_device_info():
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0), True   # supports_amp
    if torch.backends.mps.is_available():
        return "mps", "Apple Silicon", False                 # MPS 关 AMP
    return "cpu", platform.processor(), False
```

被各训练脚本用于：
- 自动决定 `device_type`（用哪块黑板）
- MPS 时强制 `flash_attn=False, autocast=disabled, num_workers=0`（苹果芯片的特殊限制）
- CUDA 时启用 TF32、cudnn benchmark、`expandable_segments`（NVIDIA GPU 的性能优化）

这确保了同一套代码能在不同硬件上正常运行，就像教案要适配不同教室的设备条件。

## 14.4 学习率调度：控制教学节奏

`get_lr(current_step, total_steps, lr)` 就像课堂上的讲课速度控制器，采用无 warmup 的余弦衰减策略，从初始学习率 `lr` 平滑下降到 `0.1 × lr`：

```python
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

特性：
- **起点**：`lr`（刚开始讲得快一些）
- **终点**：`0.1 × lr`（最后慢慢收尾，最低降到 10%）
- **曲线**：单峰余弦曲线，无 warmup 阶段（没有热身环节）
- **适用范围**：所有训练脚本（pretrain / sft / lora / dpo / distillation / ppo / grpo / agent）共用同一套节奏

> 设计取舍：MiniMind3 数据规模小（百 M~G 级 token），warmup 收益有限，去掉 warmup 让脚本更简洁。就像小班教学不需要太长的热身时间。

## 14.5 模型参数量统计：`get_model_params(model, config)`

就像课前统计教室里有多少学生一样，这个函数支持 Dense 与 MoE 双模式，准确计算模型的"规模"：

```python
total = sum(p.numel() for p in model.parameters()) / 1e6
n_routed = config.num_experts                    # 4
n_active = config.num_experts_per_tok            # 1
expert   = sum(p.numel() ... 'mlp.experts.0.') / 1e6   # 单专家参数量
base     = total - expert * n_routed              # 非 MoE 部分
active   = base + expert * n_active               # 实际激活参数量
```

输出示例：
- **Dense 模式**：`Model Params: 26.30M`（总共 2630 万参数）
- **MoE 模式**：`Model Params: 198.50M-A64.20M`（总参数 1.985 亿，但每次只激活 6420 万）

这让你清楚知道模型的"体重"和"实际工作量"，方便评估硬件需求。

## 14.6 模型加载：`init_model(lm_config, from_weight, ...)`

就像上课前准备好教材和教具一样，这是统一的模型初始化入口，被所有训练脚本与推理脚本调用：

```python
def init_model(lm_config, from_weight='pretrain',
               tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)
    if from_weight != 'none':
        weight_path = f'{save_dir}/{from_weight}_{hidden}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
    return model.to(device), tokenizer
```

要点：
- `from_weight='none'`：随机初始化（仅 pretrain 用，相当于新课从零开始）
- `from_weight='pretrain' / 'full_sft' / 'rlhf' / ...`：自动拼接路径加载已有权重（相当于接着上次的课继续讲）
- 路径相对于 `trainer/` 目录，自动转绝对路径
- `strict=False`：允许 LoRA 等额外字段不在 base 权重里（灵活适配不同配置）

## 14.7 检查点管理：自动存档的游戏机制

`lm_checkpoint(...)` 采用**双文件设计**，就像游戏里的"快速存档"和"完整存档"：

| 文件 | 用途 | 内容 |
|------|------|------|
| `{weight}_{hidden}{_moe}.pth` | **部署/推理**（快速存档） | 仅模型权重（fp16，CPU），可直接用于推理 |
| `{weight}_{hidden}{_moe}_resume.pth` | **断点续训**（完整存档） | 模型 + optimizer + epoch + step + world_size + wandb_id + extra |

### 14.7.1 保存（model 不为 None）

```python
state_dict = {k: v.half().cpu() for k, v in raw_model.state_dict().items()}
# 1. 写部署 ckpt（快速存档，方便随时加载推理）
torch.save(state_dict, ckp_path + '.tmp')
os.replace(ckp_path + '.tmp', ckp_path)        # 原子替换，防止写入中断
# 2. 写续训 ckpt（完整存档，包含所有训练状态）
resume_data = {'model': state_dict, 'optimizer': ..., 'step': ..., ...}
torch.save(resume_data, resume_path + '.tmp')
os.replace(resume_path + '.tmp', resume_path)
```

**安全特性**：
- `tmp + os.replace`：保证写入原子性，避免训练崩溃留下半截文件（就像游戏存档不会存到一半断电）
- `half().cpu()`：fp16 + CPU，节省一半磁盘空间
- `_orig_mod` 处理：兼容 `torch.compile` 包装的模型
- `module` 处理：兼容 DDP 包装

### 14.7.2 加载（model 为 None）：从书签处继续阅读

```python
if os.path.exists(resume_path):
    ckp_data = torch.load(resume_path, map_location='cpu')
    saved_ws  = ckp_data.get('world_size', 1)
    current_ws = dist.get_world_size() if dist.is_initialized() else 1
    if saved_ws != current_ws:
        ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
    return ckp_data
```

**关键能力：跨 GPU 数自适应**。就像你换了台电脑玩游戏，存档依然能读：

例如保存时 8 卡训练，恢复时只剩 4 卡：

```
new_step = old_step × 8 / 4 = 2 × old_step
```

确保等价的训练进度被还原，无需手动换算。这让你可以灵活迁移训练任务到不同硬件环境。

## 14.8 续训跳过：`SkipBatchSampler`——从书签处继续读

`DataLoader` 的 batch sampler 包装类，就像读书时的书签功能，跳过前 `skip_batches` 个 batch：

```python
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches
    def __iter__(self):
        batch, skipped = [], 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1; batch = []; continue
                yield batch; batch = []
```

用法：

```python
sampler = SkipBatchSampler(RandomSampler(ds), batch_size, skip_batches=resumed_step)
loader  = DataLoader(ds, batch_sampler=sampler, ...)
```

只在断点续训当 epoch 内生效，从下一个 epoch 开始恢复正常顺序。就像你读到第 50 页停电了，来电后从第 50 页继续，而不是重新从第 1 页开始。

## 14.9 设备探测与适配

虽然 `trainer_utils.py` 没有专门函数，但提供两个辅助工具来识别当前"教室"的硬件条件：

```python
def get_default_device():
    if torch.cuda.is_available(): return "cuda:0"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def get_device_type(device):
    s = str(device)
    return "cuda" if "cuda" in s else ("mps" if "mps" in s else "cpu")
```

各训练脚本据此分流，就像根据教室设备调整教学方法：

| 设备 | 调整策略 |
|------|------|
| CUDA | TF32 + cudnn benchmark + AMP（bf16/fp16）+ Flash Attn + `expandable_segments`（充分利用 NVIDIA GPU 性能） |
| MPS  | 关 AMP、关 Flash Attn、`num_workers=0`、数据 prefetch 到 mps device（苹果芯片的特殊限制） |
| CPU  | 仅调试用，关 AMP，使用纯 fp32（速度最慢，仅用于验证逻辑） |

## 14.10 分布式训练（DDP）：多教室同步上课详解

### 启动方式

```bash
torchrun --nproc_per_node=8 trainer/train_pretrain.py [args]
```

### 内部行为

1. `init_distributed_mode()`：检测 `RANK` 环境变量，调用 `dist.init_process_group("nccl")`（建立教室间的通信通道）
2. `model = DistributedDataParallel(model, device_ids=[local_rank])`（给每个教室发一份相同的教材）
3. `is_main_process()`：仅 rank 0 打印日志、写 wandb、保存权重（只有主讲老师能发言）
4. `Logger(content)`：包装的 print，自动只在 rank 0 输出（避免多个教室同时说话造成混乱）
5. 每个 rank 独立 forward/backward，DDP 自动 all-reduce 梯度（各教室做完题后统一对答案）
6. 训练结束 `dist.destroy_process_group()`（下课解散）

### 与 grad accumulation 的协作

每 `accumulation_steps` 才 `optimizer.step()`，DDP 在 `backward()` 时会触发 all-reduce。如需进一步优化可用 `model.no_sync()` 上下文跳过中间 step 的同步，但 MiniMind3 默认不开启（小模型同步开销可忽略）。这就像每次小测后都对答案，而不是等所有小测做完再对。

## 14.11 奖励模型：`LMForRewardModel`

为 RLHF/RLAIF 提供的奖励模型封装，就像请了一位专业评委来给学生的作业打分：

| 字段 | 说明 |
|------|------|
| `model_path` | 预训练奖励模型路径（评委的评分标准） |
| `device` | 部署设备 |
| `dtype` | 默认 `torch.float16`，节省显存 |

调用方式：

```python
rm = LMForRewardModel(model_path='./out/reward.pth', device='cuda')
score = rm.score(prompt, response)   # 返回 [-3.0, 3.0] 的标量分数
```

被 `train_ppo.py / train_grpo.py / train_agent.py` 在 rollout 阶段调用，对生成的 response 打分，与规则 reward（format / length / 答案匹配）加权融合。这就像既有老师打分，又有自动批改系统，综合得出最终成绩。

## 14.12 关键设计取舍

| 取舍 | 原因 |
|------|------|
| 只用 `torchrun` 而非 `accelerate` | 减少依赖，让训练循环可读（就像用黑板粉笔而不是复杂的多媒体设备） |
| 不用 `transformers.Trainer` | 教学目的，让每行循环都显式可见（让学生看清每一步操作） |
| 自实现 LR 调度而非 `torch.optim.lr_scheduler` | 简单直观，无 warmup（小班教学不需要复杂节奏控制） |
| 双文件 ckpt（部署 + resume 分离） | 部署 ckpt 可直接用于推理，无 optimizer 噪音（快速存档只存必要内容） |
| `half().cpu()` 保存 | 节省磁盘，加载时再 cast（压缩存档节省空间） |
| 跨 GPU 数自适应 step | 训练任务可灵活迁移到不同硬件（换教室也能接着上课） |

## 14.13 调用关系总览

```mermaid
flowchart TB
    Train[各 train_*.py 训练脚本] --> Init[init_model 准备教材]
    Train --> DDP[init_distributed_mode 多教室同步]
    Train --> Seed[setup_seed 随机种子]
    Train --> LR[get_lr 控制节奏]
    Train --> Stat[get_model_params 统计规模]
    Train --> Ckpt[lm_checkpoint 自动存档]
    Train --> Skip[SkipBatchSampler 书签续读]
    Train --> Log[Logger 日志输出]
    Ckpt --> Atom[原子写 + os.replace 安全保存]
    Ckpt --> WS[跨 GPU 数自适应 换教室继续]
    DDP --> Main[is_main_process 主讲老师]
    Log --> Main
```

理解 `trainer_utils.py` 后，再看 8 个 `train_*.py` 脚本会发现它们高度同构，差异只在 loss 与数据处理。这正是 MiniMind3 工程化的精髓：**抽象底座，业务清晰**——就像所有课程共用同一套教室设备，老师只需专注于教学内容。