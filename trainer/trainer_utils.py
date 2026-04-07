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
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel
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
