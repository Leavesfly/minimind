"""
训练工具函数集合

本模块提供了模型训练所需的工具函数和类：
- get_model_params: 计算模型参数量，支持 MoE 模型的激活参数量统计
- get_lr: 余弦退火学习率调度
- init_distributed_mode: 初始化 DDP 分布式训练环境
- lm_checkpoint: 模型检查点的保存和加载
- init_model: 初始化模型和分词器
- SkipBatchSampler: 跳过已训练 batch 的采样器
- LMForRewardModel: 奖励模型，用于 RLHF 训练

主要功能：
- 分布式训练支持（DDP）
- 模型检查点管理（支持断点续训）
- 学习率调度（余弦退火）
- MoE 模型参数量统计
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import random
import re
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from transformers import AutoModel, AutoTokenizer

from model.model_minimind import MiniMindForCausalLM


def get_model_params(model, config):
    """
    计算模型参数量，支持 MoE (Mixture of Experts) 模型的激活参数量统计
    
    对于 MoE 模型，会分别统计：
    - 总参数量：所有专家的参数总和
    - 激活参数量：实际参与计算的参数（共享参数 + 激活的专家参数）
    
    Args:
        model: 模型实例
        config: 模型配置对象，包含 MoE 相关配置
    
    Returns:
        无返回值，直接打印参数量信息
    """
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total:
        Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else:
        Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    余弦退火学习率调度
    
    学习率从初始值逐渐衰减到 10% 的初始值，使用余弦函数实现平滑衰减。
    公式：lr = initial_lr * (0.1 + 0.45 * (1 + cos(π * step / total_steps)))
    
    Args:
        current_step: 当前训练步数
        total_steps: 总训练步数
        lr: 初始学习率
    
    Returns:
        当前步数的学习率
    """
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def get_default_device():
    """自动检测最佳可用设备：cuda > mps > cpu"""
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_type(device):
    """从设备字符串中提取设备类型"""
    device_str = str(device)
    if "cuda" in device_str:
        return "cuda"
    elif "mps" in device_str:
        return "mps"
    return "cpu"


def init_distributed_mode():
    """
    初始化 DDP (DistributedDataParallel) 分布式训练环境
    
    通过环境变量检测是否启用分布式训练：
    - RANK=-1: 单机单卡模式
    - RANK>=0: 分布式模式，使用 NCCL 后端
    
    Returns:
        local_rank: 当前进程的本地 GPU rank，单卡模式返回 0
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None,
                  save_dir='../checkpoints', **kwargs):
    """
    模型检查点的保存和加载
    
    保存模式（model 不为 None）：
    1. 保存模型权重到 {weight}_{hidden_size}{_moe}.pth（仅模型）
    2. 保存训练状态到 {weight}_{hidden_size}{_moe}_resume.pth（包含优化器、epoch、step 等）
    3. 使用临时文件 + 原子替换确保保存安全
    4. 自动处理 DDP 包装和 FSDP 原始模块
    
    加载模式（model 为 None）：
    1. 加载 resume 检查点
    2. 自动处理 GPU 数量变化，调整 step 数量
    
    Args:
        lm_config: 模型配置
        weight: 权重名称（如 'pretrain', 'full_sft'）
        model: 模型实例，None 表示加载模式
        optimizer: 优化器实例
        epoch: 当前 epoch
        step: 当前 step
        wandb: wandb 实例
        save_dir: 保存目录
        **kwargs: 其他需要保存的状态（如 scheduler、ema 等）
    
    Returns:
        加载模式时返回检查点数据，保存模式时返回 None
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    """
    初始化模型和分词器
    
    Args:
        lm_config: 模型配置对象
        from_weight: 要加载的权重名称（如 'pretrain', 'full_sft'），'none' 表示不加载
        tokenizer_path: 分词器路径
        save_dir: 权重保存目录
        device: 设备类型
    
    Returns:
        (model, tokenizer) 元组
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(tokenizer_path):
        tokenizer_path = os.path.normpath(os.path.join(script_dir, tokenizer_path))
    if not os.path.isabs(save_dir):
        save_dir = os.path.normpath(os.path.join(script_dir, save_dir))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """
    跳过已训练 batch 的采样器
    
    用于断点续训时跳过已经训练过的 batch，避免重复训练。
    例如：已经训练了 100 个 batch，设置 skip_batches=100，则从第 101 个 batch 开始训练。
    
    Args:
        sampler: 基础采样器（如 RandomSampler）
        batch_size: batch 大小
        skip_batches: 要跳过的 batch 数量
    """

    def __init__(self, sampler, batch_size, skip_batches=0):
        """
        初始化 SkipBatchSampler
        
        Args:
            sampler: 基础采样器
            batch_size: batch 大小
            skip_batches: 要跳过的 batch 数量
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        """
        生成批次索引，跳过前 skip_batches 个 batch
        
        Yields:
            batch: 包含样本索引的列表
        """
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        """
        返回实际训练的 batch 数量
        
        Returns:
            总 batch 数量减去跳过的 batch 数量
        """
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    """
    奖励模型，用于 RLHF (Reinforcement Learning from Human Feedback) 训练
    
    功能：
    - 对模型生成的回复进行评分
    - 评分范围：[-3.0, 3.0]，分数越高表示回复质量越好
    - 支持多轮对话上下文
    
    使用场景：
    - PPO 训练：计算奖励信号
    - DPO 训练：生成偏好对
    - 模型评估：评估回复质量
    """

    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        """
        初始化奖励模型
        
        Args:
            model_path: 预训练奖励模型路径
            device: 设备类型
            dtype: 数据类型
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        """
        对回复进行评分
        
        Args:
            messages: 对话历史消息列表
            response: 待评分的回复内容
        
        Returns:
            score: 评分，范围 [-3.0, 3.0]
        """
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)


# =====================================================================================
# 通用训练样板函数：抽取自各 train_*.py 中的重复模式
# 这些函数封装了所有训练脚本共享的初始化、配置、保存、恢复逻辑，
# 让具体的训练脚本只需关注算法本身（损失计算、训练循环）。
# =====================================================================================

def setup_precision_context(device_type, dtype_str, lm_config=None, *, disable_amp_on_mps=False):
    """构造混合精度训练所需的 autocast 上下文与 GradScaler。

    本函数统一处理三类设备的差异：
    - CUDA：使用 ``torch.cuda.amp.autocast``，仅当 dtype 为 float16 时启用 GradScaler。
    - MPS（Apple Silicon）：可选关闭 amp（实测 autocast+scaler 比 fp32 慢 3 倍以上）。
    - CPU：不启用混合精度。

    同时，由于 MPS 上 ``F.scaled_dot_product_attention`` 性能极差
    （forward 慢约 15x，backward 慢 100x+），会自动关闭 ``lm_config.flash_attn``。

    Args:
        device_type: 设备类型字符串，"cuda" / "mps" / "cpu"。
        dtype_str: 混合精度数据类型字符串，"bfloat16" 或 "float16"。
        lm_config: 模型配置对象，若提供且为 MPS 设备会强制关闭 ``flash_attn``。
        disable_amp_on_mps: 是否在 MPS 上完全关闭 amp（推荐设为 True）。

    Returns:
        (autocast_ctx, scaler, use_scaler) 三元组：
            - autocast_ctx: 用于 ``with`` 语句的混合精度上下文管理器。
            - scaler: ``torch.amp.GradScaler`` 实例（始终返回，未启用时 ``enabled=False``）。
            - use_scaler: 布尔值，标识 scaler 是否真正启用（仅 CUDA + float16 为 True）。
    """
    from contextlib import nullcontext

    # MPS 上禁用 flash_attn，避免 SDPA 性能崩塌
    if device_type == "mps" and lm_config is not None and getattr(lm_config, "flash_attn", False):
        lm_config.flash_attn = False
        Logger("⚡ MPS: flash_attn disabled (SDPA is extremely slow on MPS, using manual attention)")

    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    if device_type == "cuda":
        autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
        use_scaler = (dtype_str == "float16")
        scaler = torch.amp.GradScaler(device="cuda", enabled=use_scaler)
    elif device_type == "mps":
        if disable_amp_on_mps:
            Logger("⚡ MPS: autocast/scaler disabled (native fp32 is fastest on Apple Silicon)")
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(device_type="mps", dtype=dtype)
        scaler = torch.amp.GradScaler(enabled=False)
        use_scaler = False
    else:  # cpu
        autocast_ctx = nullcontext()
        scaler = torch.amp.GradScaler(enabled=False)
        use_scaler = False

    return autocast_ctx, scaler, use_scaler


def setup_cuda_perf_options():
    """为 CUDA 设备开启常用性能优化（TF32、cuDNN benchmark、可扩展显存段）。

    对 Ampere 及以上架构 GPU 启用 TF32 替代 FP32 做矩阵乘法和卷积，
    精度损失极小但速度提升显著；同时启用 cuDNN benchmark 自动选择
    最快卷积算法（输入尺寸固定时效果最佳）。

    建议仅在预训练等长时间任务中调用一次。
    """
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    Logger("⚡ CUDA: TF32 + cuDNN benchmark + expandable_segments enabled")


def setup_wandb(args, ckp_data=None, run_name_prefix=None):
    """初始化 wandb / swanlab 日志记录器，自动支持断点续训。

    若 ``ckp_data`` 中存有先前 run 的 ``wandb_id``，会自动以 ``resume='must'`` 接续。

    Args:
        args: argparse 解析得到的命令行参数对象，需包含 ``use_wandb``、``wandb_project``，
            可选包含 ``epochs``、``batch_size``、``learning_rate``。
        ckp_data: 检查点数据字典（可为 None），用于读取历史 wandb_id。
        run_name_prefix: 自定义 run 名前缀，默认从 ``wandb_project`` 推导。

    Returns:
        wandb 模块对象（实际为 swanlab）；若未启用或非主进程则返回 None。
    """
    if not getattr(args, "use_wandb", False) or not is_main_process():
        return None

    import swanlab as wandb

    wandb_id = ckp_data.get("wandb_id") if ckp_data else None
    resume = "must" if wandb_id else None

    prefix = run_name_prefix or getattr(args, "wandb_project", "MiniMind")
    epochs = getattr(args, "epochs", "?")
    batch_size = getattr(args, "batch_size", "?")
    lr = getattr(args, "learning_rate", "?")
    run_name = f"{prefix}-Epoch-{epochs}-BatchSize-{batch_size}-LearningRate-{lr}"

    wandb.init(project=getattr(args, "wandb_project", "MiniMind"),
               name=run_name, id=wandb_id, resume=resume)
    return wandb


def save_checkpoint(model, lm_config, save_dir, save_weight, *, optimizer=None,
                    scaler=None, scheduler=None, epoch=0, step=0, wandb=None,
                    resume_dir="../checkpoints"):
    """统一的模型权重 + 训练状态保存函数。

    同时完成两件事：
    1. 将模型权重以 fp16 保存到 ``{save_dir}/{save_weight}_{hidden_size}{_moe}.pth``，
       供推理直接加载。
    2. 调用 ``lm_checkpoint`` 把 optimizer / scaler / scheduler / epoch / step 等
       完整训练状态保存到 ``{resume_dir}/..._resume.pth``，供断点续训使用。

    仅在主进程执行实际保存动作（多进程下其他进程会直接返回）。

    Args:
        model: 当前训练模型（可被 DDP 或 ``torch.compile`` 包装）。
        lm_config: 模型配置对象。
        save_dir: 推理权重保存目录。
        save_weight: 权重文件前缀名，例如 ``"pretrain"``、``"full_sft"``。
        optimizer: 优化器实例。
        scaler: ``torch.amp.GradScaler`` 实例（可为 None）。
        scheduler: 学习率调度器（可为 None）。
        epoch: 当前 epoch。
        step: 当前 step。
        wandb: wandb / swanlab 实例（用于保存 wandb_id）。
        resume_dir: 训练状态保存目录。
    """
    if not is_main_process():
        return
    moe_suffix = "_moe" if lm_config.use_moe else ""
    weight_path = f"{save_dir}/{save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    # NOTE: 一次性 half().cpu() 拷贝整个 state_dict 在 ≥1B 模型上会带来短暂的内存峰值；
    # 当前 MiniMind (~100M-400M) 范围下完全可控，未来若训练更大模型可改为按张量分片落盘。
    state_dict = {k: v.half().cpu() for k, v in raw_model.state_dict().items()}
    torch.save(state_dict, weight_path)
    lm_checkpoint(lm_config, weight=save_weight, model=model, optimizer=optimizer,
                  scaler=scaler, scheduler=scheduler, epoch=epoch, step=step,
                  wandb=wandb, save_dir=resume_dir)
    del state_dict


def restore_training_state(ckp_data, model, optimizer=None, scaler=None,
                           scheduler=None, *, strict=False):
    """从检查点字典恢复模型与训练状态，返回 ``(start_epoch, start_step)``。

    Args:
        ckp_data: ``lm_checkpoint`` 加载得到的字典；为 None 时直接返回 ``(0, 0)``。
        model: 待恢复的模型。
        optimizer: 待恢复的优化器（可为 None）。
        scaler: 待恢复的 GradScaler（仅当其本身已启用时才会恢复）。
        scheduler: 待恢复的学习率调度器（可为 None）。
        strict: ``model.load_state_dict`` 的 strict 参数。默认 ``False`` 以兼容
            LoRA / 部分加载场景；纯预训练等需要严格匹配权重时可显式传 ``True``。

    Returns:
        ``(start_epoch, start_step)``：从该位置继续训练。
    """
    if not ckp_data:
        return 0, 0
    model.load_state_dict(ckp_data["model"], strict=strict)
    if optimizer is not None and "optimizer" in ckp_data:
        optimizer.load_state_dict(ckp_data["optimizer"])
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)() and "scaler" in ckp_data:
        scaler.load_state_dict(ckp_data["scaler"])
    if scheduler is not None and "scheduler" in ckp_data:
        scheduler.load_state_dict(ckp_data["scheduler"])
    start_epoch = ckp_data.get("epoch", 0)
    start_step = ckp_data.get("step", 0)
    return start_epoch, start_step


def wrap_model_for_training(model, *, use_compile=False, local_rank=0):
    """对模型按需应用 ``torch.compile`` 与 ``DistributedDataParallel`` 包装。

    Args:
        model: 待包装的原始模型。
        use_compile: 是否启用 ``torch.compile``（PyTorch 2.0+）。
        local_rank: 本地 GPU rank（DDP 使用，单卡时忽略）。

    Returns:
        包装后的模型。未启用 DDP 时与未启用 compile 时均按原状返回。
    """
    if use_compile:
        model = torch.compile(model)
        Logger("torch.compile enabled")
    if dist.is_initialized():
        # MiniMind 的旋转位置编码 buffer 不需要在 DDP 间同步
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    return model


def build_train_dataloader(dataset, batch_size, *, num_workers=0, device_type="cuda",
                           start_step=0, train_sampler=None,
                           persistent_workers=False, prefetch_factor=None):
    """构造支持断点续训的训练 ``DataLoader``。

    通过 ``SkipBatchSampler`` 自动跳过 ``start_step`` 之前的 batch，
    实现严格按 step 粒度恢复训练。

    Args:
        dataset: 训练数据集。
        batch_size: 每个 batch 的样本数。
        num_workers: ``DataLoader`` 加载工作进程数。
        device_type: 设备类型，决定是否启用 ``pin_memory``。
        start_step: 从该 step 开始训练，前面的 batch 会被跳过。
        train_sampler: 已存在的 sampler（如 ``DistributedSampler``）；为 None 时使用
            随机打乱的全量索引。
        persistent_workers: 是否启用 persistent workers（多 epoch 训练推荐开启）。
        prefetch_factor: ``DataLoader`` prefetch_factor，仅当 ``num_workers > 0`` 有效。

    Returns:
        DataLoader 实例。
    """
    # 兼容三种 train_sampler 输入：
    # 1) None       -> 用随机打乱的全量索引；
    # 2) 索引列表    -> 直接使用；
    # 3) Sampler 对象（如 DistributedSampler） -> 物化成索引列表后再封装。
    if train_sampler is None:
        indices = torch.randperm(len(dataset)).tolist()
    elif isinstance(train_sampler, list):
        indices = train_sampler
    else:
        indices = list(train_sampler)
    batch_sampler = SkipBatchSampler(indices, batch_size, skip_batches=start_step)
    loader_kwargs = {
        "batch_sampler": batch_sampler,
        "num_workers": num_workers,
        "pin_memory": (device_type == "cuda"),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def optimizer_step(optimizer, model_or_params, *, scaler=None, grad_clip=1.0):
    """统一的优化器更新一步：自动处理 GradScaler 与梯度裁剪。

    Args:
        optimizer: 优化器实例。
        model_or_params: 模型对象或参数列表（用于梯度裁剪）。LoRA 训练应仅传 LoRA 参数。
        scaler: GradScaler 实例。若启用则会先调用 ``unscale_`` 再裁剪。
        grad_clip: 梯度裁剪阈值；<=0 时跳过裁剪。
    """
    params = (model_or_params.parameters()
              if hasattr(model_or_params, "parameters") else model_or_params)
    scaler_enabled = scaler is not None and getattr(scaler, "is_enabled", lambda: False)()
    if scaler_enabled:
        scaler.unscale_(optimizer)
    if grad_clip and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
    if scaler_enabled:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def update_lr(optimizer, lr):
    """将所有参数组的学习率统一设置为 ``lr``。"""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def parse_messages_from_chat_prompt(prompt):
    """从带 ``<|im_start|>`` 标签的 chat 模板字符串中解析 messages 列表。

    用于 RL 训练中从 prompt 反推 ``[{role, content}, ...]`` 结构，
    以便传给 reward model 评分。

    Args:
        prompt: 包含 ChatML 格式的字符串。

    Returns:
        ``[{"role": ..., "content": ...}, ...]`` 列表。
    """
    pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
    matches = re.findall(pattern, prompt, re.DOTALL)
    return [{"role": role, "content": content.strip()} for role, content in matches]


def format_duration(seconds):
    """将秒数格式化为人类可读的时间字符串。

    Args:
        seconds: 秒数（可为浮点）。

    Returns:
        如 ``"30s"`` / ``"5.5min"`` / ``"2h30m"`` 的字符串。
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h{minutes:02d}m"


def get_memory_usage(device_type):
    """返回当前设备的内存使用摘要字符串，用于日志展示。

    Args:
        device_type: ``"cuda"`` / ``"mps"`` / 其他。

    Returns:
        如 ``"mem: 12.3/15.0GB"``；CPU 等不支持的设备返回空字符串。
    """
    if device_type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        return f"mem: {allocated:.1f}/{reserved:.1f}GB"
    if device_type == "mps":
        allocated = torch.mps.current_allocated_memory() / 1024 ** 3
        return f"mem: {allocated:.2f}GB"
    return ""
