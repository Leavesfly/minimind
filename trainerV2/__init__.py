# -*- coding: utf-8 -*-
"""
MiniMind MLX Trainer - Apple Silicon 原生训练框架
================================================
基于 Apple MLX 框架重新实现的 MiniMind 训练器，
充分利用 Apple Silicon 统一内存架构的优势。

核心特性：
- 零拷贝统一内存：无需 CPU↔GPU 数据搬运
- 惰性求值（Lazy Evaluation）：自动优化计算图
- 原生 float16/bfloat16 支持，无需 GradScaler
- 简洁优雅的函数式梯度计算 API

模块清单：
- model_mlx: MiniMind 模型（RoPE / GQA / RMSNorm / MoE）
- trainer_utils: 训练工具函数（学习率/梯度裁剪/检查点/日志）
- reward_utils: RL 奖励计算工具
- rollout_engine: 推理采样引擎
- train_pretrain: 预训练
- train_full_sft: 全量 SFT 微调
- train_lora: LoRA 参数高效微调
- train_dpo: DPO 偏好对齐
- train_distillation: 知识蒸馏
- train_grpo: GRPO 强化学习
- train_ppo: PPO 强化学习
- train_agent: Agent 工具调用 RL
"""
