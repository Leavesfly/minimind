# -*- coding: utf-8 -*-
"""
LoRA（Low-Rank Adaptation）低秩适配模块

LoRA 是一种参数高效微调方法，通过冻结预训练模型权重，并在每个 Linear 层旁边注入可训练的
低秩分解矩阵来实现模型适配。相比全参数微调，LoRA 大幅减少了可训练参数量，同时保持了
与全参数微调相当的性能。

核心原理：
对于原始权重矩阵 W ∈ R^(d×k)，LoRA 通过两个低秩矩阵 A ∈ R^(r×k) 和 B ∈ R^(d×r) 来学习
增量更新 ΔW = BA，其中 r << min(d, k)。训练时，前向计算为：
    W' = W + BA
    output = W'x = Wx + BAx

初始化策略：
- 矩阵 A 使用高斯分布初始化（mean=0, std=0.02）
- 矩阵 B 初始化为全零，确保初始训练时 ΔW = 0，模型行为与预训练模型一致

使用场景：
- 大规模语言模型的高效微调
- 多任务适配（每个任务训练独立的 LoRA 权重）
- 减少显存占用和存储成本
"""

import torch
from torch import nn


# 定义Lora网络结构
class LoRA(nn.Module):
    """
    LoRA（Low-Rank Adaptation）低秩适配层
    
    通过两个低秩矩阵 A 和 B 来模拟原始权重的增量更新，实现参数高效的模型微调。
    
    数学原理：
    原始线性变换：y = Wx，其中 W ∈ R^(out_features × in_features)
    LoRA 增量更新：y = Wx + BAx = (W + BA)x
    其中：
        - A ∈ R^(rank × in_features)：降维投影矩阵
        - B ∈ R^(out_features × rank)：升维投影矩阵
        - rank：低秩分解的秩，通常远小于 in_features 和 out_features
        - BA ∈ R^(out_features × in_features)：学习到的增量权重矩阵
    
    参数量对比：
    原始权重参数量：out_features × in_features
    LoRA 可训练参数量：(out_features + in_features) × rank
    当 rank << min(in_features, out_features) 时，参数量大幅减少
    
    Args:
        in_features (int): 输入特征维度
        out_features (int): 输出特征维度
        rank (int): 低秩分解的秩，控制可训练参数量和表达能力
    """

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    """
    将 LoRA 注入到模型的所有方阵 Linear 层
    
    遍历模型的所有模块，找到权重为方阵的 Linear 层（即输入维度等于输出维度），
    为这些层创建并附加 LoRA 适配器。注入后，该层的前向计算变为：
        output = original_forward(x) + lora_forward(x)
    
    注入机制：
    1. 为每个符合条件的 Linear 层创建 LoRA 实例
    2. 将 LoRA 实例作为属性 'lora' 附加到该层
    3. 动态替换该层的 forward 方法，在原始计算基础上叠加 LoRA 的输出
    
    适用场景：
    - 仅对方阵层注入 LoRA，避免破坏模型结构
    - 常用于 Transformer 的注意力机制中的 Q、K、V 投影层
    
    Args:
        model (nn.Module): 待注入 LoRA 的 PyTorch 模型
        rank (int): LoRA 的秩，默认为 16。秩越大，表达能力越强，但可训练参数量也越大
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


def load_lora(model, path):
    """
    从指定路径加载 LoRA 权重到模型
    
    读取保存的 LoRA 权重文件，并将权重加载到模型中对应的 LoRA 层。
    该函数支持处理由 DataParallel 包装的模型（权重键名带 'module.' 前缀）。
    
    加载流程：
    1. 加载权重文件，处理可能存在的 'module.' 前缀
    2. 遍历模型的所有模块，找到带有 'lora' 属性的层
    3. 根据模块名称匹配权重键，提取对应的 LoRA 权重
    4. 将权重加载到各层的 LoRA 实例中
    
    Args:
        model (nn.Module): 已注入 LoRA 的模型
        path (str): LoRA 权重文件路径
    """
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """
    保存模型中所有 LoRA 层的权重到指定路径
    
    遍历模型的所有模块，提取各 LoRA 层的权重参数，并保存到文件中。
    仅保存 LoRA 相关的权重，不保存原始模型权重，从而大幅减少存储空间。
    
    保存流程：
    1. 获取原始模型（处理可能的 DDP 包装）
    2. 遍历所有模块，找到带有 'lora' 属性的层
    3. 提取 LoRA 权重，转换为半精度（FP16）以节省空间
    4. 按层级结构组织权重键名，保存到文件
    
    Args:
        model (nn.Module): 已注入 LoRA 的模型
        path (str): 保存 LoRA 权重的文件路径
    """
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    """
    将 LoRA 权重合并到原始模型权重中，并保存合并后的完整模型
    
    合并过程将 LoRA 学习到的增量更新（BA）直接加到原始权重 W 上，得到新的权重 W' = W + BA。
    合并后的模型不再需要 LoRA 层，可以直接作为普通模型使用，推理时无需额外的计算开销。
    
    合并流程：
    1. 加载 LoRA 权重到模型
    2. 提取原始模型权重（排除 LoRA 相关权重）
    3. 对于每个 Linear 层，计算 W' = W + BA，其中 BA = B @ A
    4. 保存合并后的完整模型权重
    
    应用场景：
    - 部署时减少推理计算量（无需计算 LoRA 分支）
    - 与其他推理框架兼容（某些框架不支持动态注入）
    - 权重分发给其他用户时无需传递 LoRA 代码
    
    Args:
        model (nn.Module): 已注入 LoRA 的模型
        lora_path (str): LoRA 权重文件路径
        save_path (str): 合并后模型权重的保存路径
    """
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
