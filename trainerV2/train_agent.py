# -*- coding: utf-8 -*-
"""
MiniMind MLX Agent RL 训练脚本 - 工具调用强化学习
=============================================
基于 MLX 框架实现的 Agent 训练，专注于工具调用能力的优化。

主要功能：
- 多轮工具调用 rollout 生成
- Agent 特有奖励计算（工具调用格式、参数校验、执行结果等）
- 支持 GRPO 和 CISPo 两种 RL 算法
- 思考模式（Thinking）训练

训练流程：
1. 使用 rollout_engine 生成多个候选响应
2. 计算奖励（工具调用正确性、格式规范性、GT 命中等）
3. GRPO/CISPo 算法优化策略
4. 定期保存 checkpoint

使用方式：
    python trainerV2/train_agent.py --data_path ../.dataset/agent_rl.jsonl
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import json
import math
import random
import signal
import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from trainerV2.model_mlx import MiniMindConfig, MiniMindForCausalLM
from trainerV2.trainer_utils import (
    format_duration, get_lr, logger, save_checkpoint,
    setup_seed, setup_wandb, clip_grad_norm, log_model_params,
)
from trainerV2.reward_utils import compute_repetition_penalty
from trainerV2.rollout_engine import (
    MLXRolloutEngine, compute_per_token_logps, create_rollout_engine,
)


# =====================================================================================
# 工具定义与模拟数据
# =====================================================================================

TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式",
                                      "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                                                     "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位换算",
                                      "parameters": {"type": "object", "properties": {"value": {"type": "number"},
                                                                                      "from_unit": {"type": "string"},
                                                                                      "to_unit": {"type": "string"}},
                                                     "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气",
                                      "parameters": {"type": "object", "properties": {"location": {"type": "string"}},
                                                     "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取时间",
                                      "parameters": {"type": "object", "properties": {
                                          "timezone": {"type": "string", "default": "Asia/Shanghai"}},
                                                     "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询汇率",
                                      "parameters": {"type": "object",
                                                     "properties": {"from_currency": {"type": "string"},
                                                                    "to_currency": {"type": "string"}},
                                                     "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本",
                                      "parameters": {"type": "object", "properties": {"text": {"type": "string"},
                                                                                      "target_language": {
                                                                                          "type": "string"}},
                                                     "required": ["text", "target_language"]}}},
]

# 模拟数据
WEATHER_DATA = {"北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"),
                "深圳": ("30°C", "晴"), "杭州": ("22°C", "阴"), "成都": ("18°C", "小雨"),
                "武汉": ("25°C", "多云"), "南京": ("20°C", "晴"), "西安": ("16°C", "大风"),
                "重庆": ("26°C", "阴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"),
                "London": ("5°C", "小雨"), "Paris": ("10°C", "阴"), "Sydney": ("25°C", "晴朗")}
TIME_DATA = {"Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00",
             "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00"}
EXCHANGE_DATA = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12,
                 ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("CNY", "JPY"): 20.83}
TRANSLATE_DATA = {("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好",
                  ("今天天气真好", "english"): "The weather is nice today",
                  ("I love programming", "chinese"): "我喜欢编程"}
UNIT_DATA = {"km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462,
             "pounds_kg": 0.453592, "meters_feet": 3.28084, "feet_meters": 0.3048}

# 模拟执行
MOCK_RESULTS = {
    "calculate_math": lambda a: {"result": str(eval(
        str(a.get("expression", "0")).replace("^", "**").replace("×", "*").replace("÷", "/"),
        {"__builtins__": {}, "math": math}))},
    "unit_converter": lambda a: {"result": round(float(a.get("value", 0)) * UNIT_DATA.get(
        f"{a.get('from_unit', '').lower()}_{a.get('to_unit', '').lower()}", 1), 4)},
    "get_current_weather": lambda a: (
        lambda w: {"city": a.get("location"), "temperature": w[0], "condition": w[1]})(
        WEATHER_DATA.get(a.get("location"), ("22°C", "晴"))),
    "get_current_time": lambda a: {
        "datetime": TIME_DATA.get(a.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00")},
    "get_exchange_rate": lambda a: {"rate": EXCHANGE_DATA.get(
        (a.get("from_currency"), a.get("to_currency")), 1.0)},
    "translate_text": lambda a: {"translated_text": TRANSLATE_DATA.get(
        (a.get("text"), a.get("target_language")), a.get("text", ""))},
}

# 参数校验
CHECK_ARGS = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "unit_converter": lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: bool(a.get("location")),
    "get_current_time": lambda a: True,
    "get_exchange_rate": lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),
    "translate_text": lambda a: bool(a.get("text")) and bool(a.get("target_language")),
}


# =====================================================================================
# 工具调用解析与执行
# =====================================================================================

def parse_tool_calls(text: str) -> list:
    """从模型生成文本中解析 <tool_call>...</tool_call> 标签内的工具调用"""
    calls = []
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try:
            calls.append(json.loads(m.strip()))
        except Exception:
            pass
    return calls


def execute_tool(name: str, args: dict):
    """执行指定的 mock 工具，带超时保护"""
    fn = MOCK_RESULTS.get(name)
    if not fn:
        return None
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(1)
        return fn(args)
    except Exception:
        return None
    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass


# =====================================================================================
# 多轮 Rollout
# =====================================================================================

def rollout_single(rollout_engine: MLXRolloutEngine, tokenizer, messages: list,
                   tools: list, max_turns: int = 3, max_new_tokens: int = 256,
                   thinking_ratio: float = 0.5) -> tuple:
    """单样本多轮工具调用 rollout

    模拟 Agent 与工具环境的多轮交互：
    - 第一轮：生成初始响应（可能含 tool_call）
    - 中间轮：执行工具 → 追加 observation → 继续生成
    - 最后轮：无工具调用或达到 max_turns 时结束

    Returns:
        (final_output, final_context, prompt_ids, response_ids,
         response_mask, response_old_logps, turn_outputs, unfinished)
    """
    all_outputs = []
    prompt_ids = None
    response_ids = []
    response_mask = []
    response_old_logps = []
    final_context = ""
    unfinished = False
    open_thinking = random.random() < thinking_ratio

    for turn in range(max_turns):
        # Step 1: 构造当前 prompt
        context = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            tools=tools, open_thinking=open_thinking,
        )
        inputs = tokenizer(context, return_tensors="np", add_special_tokens=False)
        context_ids = inputs["input_ids"][0].tolist()
        if prompt_ids is None:
            prompt_ids = context_ids

        # Step 2: Rollout 生成
        input_mx = mx.array(inputs["input_ids"])
        rollout_result = rollout_engine.rollout(
            input_mx, num_generations=1,
            max_new_tokens=max_new_tokens, temperature=0.8,
        )

        # Step 3: 提取生成结果
        new_ids = rollout_result.completion_ids[0].tolist()
        new_logps = rollout_result.per_token_logps[0].tolist()

        # 过滤 pad/eos
        eos_id = tokenizer.eos_token_id or 2
        pad_id = tokenizer.pad_token_id or 0
        pairs = [(t, lp) for t, lp in zip(new_ids, new_logps)
                 if t != pad_id and t != eos_id]
        new_ids = [t for t, _ in pairs]
        new_logps = [lp for _, lp in pairs]
        new_text = rollout_result.completions[0]

        # Step 4: 累积
        all_outputs.append(new_text)
        response_ids.extend(new_ids)
        response_mask.extend([1] * len(new_ids))
        response_old_logps.extend(new_logps)
        final_context = context + new_text

        # Step 5: 解析工具调用
        calls = parse_tool_calls(new_text)
        if not calls:
            break

        unfinished = (turn == max_turns - 1)

        # Step 6: 执行工具，追加结果
        messages.append({"role": "assistant", "content": new_text})
        for call in calls:
            name, raw = call.get("name", ""), call.get("arguments", {})
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = {}
            result = execute_tool(name, raw)
            result_str = (json.dumps(result, ensure_ascii=False) if result
                          else '{"error": "tool not found"}')[:2048]
            messages.append({"role": "tool", "content": result_str})

        # Step 7: 观察 token（mask=0，不参与 policy loss）
        observe_context = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=not unfinished,
            tools=tools, open_thinking=open_thinking,
        )
        observe_ids = tokenizer(observe_context, return_tensors="np",
                                add_special_tokens=False)["input_ids"][0].tolist()
        current_len = len(prompt_ids) + len(response_ids)
        obs_delta = observe_ids[current_len:]
        response_ids.extend(obs_delta)
        response_mask.extend([0] * len(obs_delta))
        response_old_logps.extend([0.0] * len(obs_delta))
        final_context = observe_context

    final_output = all_outputs[-1] if all_outputs else ""
    prompt_ids = prompt_ids or []
    return (final_output, final_context, prompt_ids, response_ids,
            response_mask, response_old_logps, all_outputs, unfinished)


def rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch,
                  num_gen, max_turns=3, max_new_tokens=256,
                  thinking_ratio=0.5) -> tuple:
    """批量多轮工具调用 rollout"""
    all_completions, all_contexts = [], []
    all_prompt_ids, all_response_ids = [], []
    all_response_masks, all_response_old_logps = [], []
    all_turn_outputs, all_unfinished = [], []

    for messages, tools in zip(messages_batch, tools_batch):
        for _ in range(num_gen):
            msgs_copy = [dict(m) for m in messages]
            result = rollout_single(
                rollout_engine, tokenizer, msgs_copy, tools,
                max_turns, max_new_tokens, thinking_ratio,
            )
            completion, context, p_ids, r_ids, r_mask, r_logps, turns, unf = result
            all_completions.append(completion)
            all_contexts.append(context)
            all_prompt_ids.append(p_ids)
            all_response_ids.append(r_ids)
            all_response_masks.append(r_mask)
            all_response_old_logps.append(r_logps)
            all_turn_outputs.append(turns)
            all_unfinished.append(unf)

    return (all_completions, all_contexts, all_prompt_ids, all_response_ids,
            all_response_masks, all_response_old_logps, all_turn_outputs, all_unfinished)


# =====================================================================================
# Agent 奖励计算
# =====================================================================================

def validate_gt_in_text(text: str, gt_list: list) -> set:
    """验证文本中是否包含 ground truth 的数值"""
    text_lower = str(text).lower()
    text_num = str(text).replace(',', '')
    nums = [float(x) for x in re.findall(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])', text_num)]
    return {g for g in gt_list if (
        (s := str(g).strip()) and s.lower() in text_lower
    ) or (
        re.fullmatch(r'[-+]?\d+(?:\.\d+)?', str(g).strip().replace(',', ''))
        and any(abs(float(str(g).strip().replace(',', '')) - n) < 1e-6 for n in nums)
    )}


def calculate_rewards(prompts, completions, gt_batch, tools_batch, num_gen,
                      turn_outputs_batch=None, unfinished_batch=None) -> mx.array:
    """计算 Agent 响应的综合奖励

    分支 A（无工具调用）：格式规范 + 重复惩罚
    分支 B（有工具调用）：工具正确性 + GT 命中 + 未完成惩罚
    """
    rewards = [0.0] * len(completions)

    for idx, response in enumerate(completions):
        reward, answer = 0.0, response
        sample_idx = idx // num_gen
        tools = tools_batch[sample_idx]
        turn_outputs = turn_outputs_batch[idx] if turn_outputs_batch else [response]
        unfinished = unfinished_batch[idx] if unfinished_batch else False

        # 提取每轮回答（去掉思考部分）
        turn_answers = [
            turn.split('</think>', 1)[-1].strip() if '</think>' in turn else turn.strip()
            for turn in turn_outputs
        ]
        answer = turn_answers[-1] if turn_answers else response.strip()
        valid_names = {t['function']['name'] for t in tools} if tools else set()

        # 聚合所有轮次的工具调用
        tool_calls = []
        for ta in turn_answers:
            tool_calls.extend(parse_tool_calls(ta))

        # 标签不成对扣分
        reward -= 0.5 * sum(
            abs(turn.count('<tool_call>') - turn.count('</tool_call>'))
            for turn in turn_answers
        )

        if not tool_calls:
            # 分支 A: 无工具调用
            reward += 0.5 if 5 <= len(response.strip()) <= 800 else -0.5
            if '</think>' in response:
                think, answer = response.split('</think>', 1)
                reward += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5
                reward += 0.25 if response.count('</think>') == 1 else -0.25
                answer = answer.strip()
            reward -= compute_repetition_penalty(answer)
        else:
            # 分支 B: 有工具调用
            gt = gt_batch[sample_idx]

            # 统计合法调用数
            valid_count = 0
            for tc in tool_calls:
                name = tc.get("name", "")
                raw = tc.get("arguments", {})
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except Exception:
                        raw = {}
                check = CHECK_ARGS.get(name)
                valid_count += int(bool(name in valid_names and check and check(raw)))

            # 工具对齐分
            tool_gap = abs(valid_count - len(gt)) + max(0, len(tool_calls) - valid_count)
            reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap

            # GT 命中
            final_text = "" if unfinished else (
                answer.split('</tool_call>')[-1] if '</tool_call>' in answer else answer)
            verified = validate_gt_in_text(final_text, gt) if gt else set()
            if gt:
                reward += 2.5 * len(verified) / len(gt)
            if unfinished:
                reward -= 0.5
            reward -= compute_repetition_penalty(final_text if final_text else answer)

        rewards[idx] = max(min(reward, 3.0), -3.0)

    return mx.array(rewards)


# =====================================================================================
# RL 训练步
# =====================================================================================

def rl_train_step(model, ref_model, optimizer, rollout_engine,
                  messages_batch, tools_batch, gt_batch,
                  tokenizer, args, global_step) -> dict:
    """Agent RL 单步训练

    流程：
    1. 多轮 rollout 生成候选
    2. 构造训练张量
    3. 计算奖励和组内优势
    4. GRPO/CISPo 策略更新
    """
    # Phase 1: Rollout
    completions, contexts, prompt_ids_batch, response_ids_batch, \
        response_masks_batch, response_old_logps_batch, \
        turn_outputs_batch, unfinished_batch = rollout_batch(
            rollout_engine, tokenizer, messages_batch, tools_batch,
            args.num_generations, max_turns=3,
            max_new_tokens=args.max_gen_len, thinking_ratio=args.thinking_ratio,
        )

    if not completions:
        return {"reward": 0.0, "loss": 0.0, "kl": 0.0}

    # Phase 2: 构造训练张量
    prompts = [tokenizer.apply_chat_template(m, tokenize=False,
               add_generation_prompt=True, tools=t)
               for m, t in zip(messages_batch, tools_batch)]

    packed_samples = []
    for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch,
                                response_masks_batch, response_old_logps_batch):
        ids = p + r
        mask = [0] * len(p) + m
        old_logps = [0.0] * max(len(p) - 1, 0) + old_lp
        if len(ids) > args.max_total_len:
            ids = ids[-args.max_total_len:]
            mask = mask[-args.max_total_len:]
            old_logps = old_logps[-(len(ids) - 1):]
        prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask))
        packed_samples.append((ids, mask, prompt_len, old_logps))

    if not packed_samples:
        return {"reward": 0.0, "loss": 0.0, "kl": 0.0}

    # Padding 到 batch 内最大长度
    max_len = max(len(ids) for ids, _, _, _ in packed_samples)
    pad_id = tokenizer.pad_token_id or 0
    eos_id = tokenizer.eos_token_id or 2

    input_ids = mx.array([ids + [pad_id] * (max_len - len(ids))
                          for ids, _, _, _ in packed_samples])
    full_response_masks = mx.array(
        [mask + [0] * (max_len - len(mask)) for _, mask, _, _ in packed_samples],
        dtype=mx.float32,
    )
    old_per_token_logps = mx.array(
        [lp + [0.0] * ((max_len - 1) - len(lp)) for _, _, _, lp in packed_samples],
        dtype=mx.float32,
    )

    # Phase 3: 计算奖励
    rewards = calculate_rewards(
        prompts, completions, gt_batch, tools_batch,
        args.num_generations, turn_outputs_batch, unfinished_batch,
    )

    # Phase 4: GRPO 组内优势
    G = args.num_generations
    rewards_grouped = rewards.reshape(-1, G)
    mean_r = mx.repeat(mx.mean(rewards_grouped, axis=1), G)
    std_r = mx.repeat(mx.maximum(mx.std(rewards_grouped, axis=1), mx.array(1e-4)), G)
    advantages = (rewards - mean_r) / std_r

    # Phase 5: 策略更新
    def policy_loss_fn(model_):
        output = model_(input_ids)
        logits = output.logits[:, :-1, :]
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        # Gather per-token logps
        labels = input_ids[:, 1:]
        B, L, V = log_probs.shape
        batch_idx = mx.arange(B)[:, None]
        seq_idx = mx.arange(L)[None, :]
        per_token_logps = log_probs[batch_idx, seq_idx, labels]

        # Completion mask（EOS 之后不参与）
        completion_mask = full_response_masks[:, 1:]
        is_eos = (labels == eos_id) & (completion_mask > 0)
        # 简化 EOS 处理
        mask_for_loss = completion_mask

        # 参考模型 logps
        ref_logps = mx.stop_gradient(compute_per_token_logps(ref_model, input_ids, max_len - 1))

        # KL 散度
        kl_div = ref_logps - per_token_logps
        per_token_kl = mx.exp(kl_div) - kl_div - 1

        # 重要性采样比
        ratio = mx.exp(per_token_logps - old_per_token_logps)

        if args.loss_type == "cispo":
            clamped_ratio = mx.stop_gradient(mx.clip(ratio, a_min=None, a_max=args.epsilon_high))
            per_token_loss = -(clamped_ratio * advantages[:, None] * per_token_logps
                               - args.beta * per_token_kl)
        else:
            clipped_ratio = mx.clip(ratio, 1 - args.epsilon, 1 + args.epsilon)
            obj1 = ratio * advantages[:, None]
            obj2 = clipped_ratio * advantages[:, None]
            per_token_loss = -(mx.minimum(obj1, obj2) - args.beta * per_token_kl)

        # 平均 loss
        token_counts = mx.maximum(mx.sum(mask_for_loss, axis=1), mx.array(1.0))
        sample_loss = mx.sum(per_token_loss * mask_for_loss, axis=1) / token_counts
        valid_mask = token_counts > 0
        loss = mx.mean(sample_loss * valid_mask.astype(mx.float32))

        # MoE 辅助损失
        aux_loss = output.aux_loss if hasattr(output, 'aux_loss') and output.aux_loss is not None else mx.array(0.0)
        return loss + aux_loss

    # 梯度计算与更新
    loss_and_grad_fn = nn.value_and_grad(model, policy_loss_fn)
    loss_val, grads = loss_and_grad_fn(model)
    grads = clip_grad_norm(grads, args.grad_clip)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    return {
        "reward": float(mx.mean(rewards).item()),
        "loss": float(loss_val.item()),
        "kl": 0.0,  # 简化，避免额外前向
        "group_std": float(mx.mean(mx.std(rewards_grouped, axis=1)).item()),
        "avg_len": float(mx.sum(full_response_masks).item() / max(len(completions), 1)),
    }


# =====================================================================================
# 主训练流程
# =====================================================================================

def train(args):
    """Agent RL 完整训练流程"""
    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    # 模型配置
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        num_key_value_heads=2,
    )

    # 加载 tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("../model", trust_remote_code=True)

    # 策略模型
    model = MiniMindForCausalLM(config)
    weight_path = os.path.join(args.save_dir, f"{args.from_weight}.safetensors")
    if os.path.exists(weight_path):
        weights = mx.load(weight_path)
        model.load_weights(list(weights.items()))
        logger(f"Model loaded from {weight_path}")

    # 参考模型（冻结）
    ref_model = MiniMindForCausalLM(config)
    if os.path.exists(weight_path):
        ref_model.load_weights(list(weights.items()))
    ref_model.freeze()
    logger("Reference model frozen")

    # 优化器
    optimizer = optim.AdamW(learning_rate=args.learning_rate)

    # Rollout 引擎
    eos_id = tokenizer.eos_token_id or 2
    pad_id = tokenizer.pad_token_id or 0
    rollout_engine = create_rollout_engine(model, tokenizer, eos_id, pad_id)

    # 数据集
    from dataset.lm_dataset import AgentRLDataset
    train_ds = AgentRLDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    logger(f"Dataset loaded: {len(train_ds)} samples")
    log_model_params(model)

    # wandb
    wandb = setup_wandb(args) if args.use_wandb else None

    def collate_fn(batch):
        return {
            'messages': [b['messages'] for b in batch],
            'tools': [b['tools'] for b in batch],
            'gt': [b['gt'] for b in batch],
        }

    # 训练循环
    total_steps = math.ceil(len(train_ds) / args.batch_size) * args.epochs
    global_step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        setup_seed(42 + epoch)
        indices = np.random.permutation(len(train_ds)).tolist()

        for batch_start in range(0, len(train_ds), args.batch_size):
            global_step += 1
            batch_indices = indices[batch_start:batch_start + args.batch_size]
            batch_items = [train_ds[i] for i in batch_indices]
            batch = collate_fn(batch_items)

            # 动态学习率
            lr = get_lr(global_step, total_steps, args.learning_rate)
            optimizer.learning_rate = lr

            # RL 训练步
            metrics = rl_train_step(
                model, ref_model, optimizer, rollout_engine,
                batch['messages'], batch['tools'], batch['gt'],
                tokenizer, args, global_step,
            )

            # 同步策略
            rollout_engine.update_policy(model)

            # 日志
            if global_step % args.log_interval == 0:
                elapsed = time.time() - start_time
                logger(
                    f"Epoch:[{epoch + 1}/{args.epochs}]({global_step}/{total_steps}), "
                    f"Reward: {metrics['reward']:.4f}, "
                    f"Loss: {metrics['loss']:.4f}, "
                    f"GrpStd: {metrics.get('group_std', 0):.4f}, "
                    f"AvgLen: {metrics.get('avg_len', 0):.1f}, "
                    f"LR: {lr:.2e}, "
                    f"Time: {format_duration(elapsed)}"
                )
                if wandb:
                    wandb.log(metrics | {"learning_rate": lr})

            # 保存
            if global_step % args.save_interval == 0:
                save_checkpoint(model, optimizer, args.save_dir,
                                args.save_weight, epoch=epoch, step=global_step)

    # 最终保存
    save_checkpoint(model, optimizer, args.save_dir, args.save_weight,
                    epoch=args.epochs, step=global_step)
    logger(f"Training complete! Total time: {format_duration(time.time() - start_time)}")


# =====================================================================================
# 命令行入口
# =====================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind MLX Agent RL Training")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", type=str, default="agent")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_gen_len", type=int, default=768)
    parser.add_argument("--max_total_len", type=int, default=2500)
    parser.add_argument("--data_path", type=str, default="../.dataset/agent_rl.jsonl")
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"])
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=5.0)
    parser.add_argument("--from_weight", type=str, default="full_sft")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Agent-RL-MLX")
    parser.add_argument("--thinking_ratio", type=float, default=0.1)
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--debug_interval", type=int, default=20)
    args = parser.parse_args()

    train(args)
