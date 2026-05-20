# -*- coding: utf-8 -*-
"""
MiniMind Agent RL 训练脚本 - 工具调用强化学习
==============================================
本脚本实现了基于强化学习的 Agent 训练，专注于工具调用能力的优化。

主要功能：
- 支持多轮工具调用的 rollout 生成
- 实现 Agent 特有的奖励计算（工具调用格式、参数校验、执行结果等）
- 支持 GRPO 和 CISPo 两种强化学习算法
- 集成 Reward Model 进行质量评估
- 支持思考模式（Thinking）的训练

训练流程：
1. 使用 rollout_engine 生成多个候选响应
2. 计算每个响应的奖励（基于工具调用正确性、格式规范性、执行结果等）
3. 使用 PPO/GRPO 算法优化策略
4. 定期保存模型 checkpoint

作者：MiniMind Team
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import json
import math
import random
import signal
import argparse
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import AgentRLDataset
from trainer.reward_utils import compute_repetition_penalty
from trainer.rollout_engine import compute_per_token_logps, create_rollout_engine
from trainer.trainer_utils import (
    LMForRewardModel, Logger, SkipBatchSampler, build_train_dataloader,
    get_default_device, get_device_type, init_distributed_mode, init_model,
    is_main_process, lm_checkpoint, restore_training_state, save_checkpoint,
    setup_precision_context, setup_seed, setup_wandb, wrap_model_for_training,
)

warnings.filterwarnings('ignore')


# ================================ 工具与 Reward = Start ================================

# n-gram 重复惩罚统一抽取到 trainer.reward_utils.compute_repetition_penalty，
# 这里保留 rep_penalty 同名别名以维持本文件其他位置的 API 兼容
rep_penalty = compute_repetition_penalty


# ======== 工具定义 ========
# 定义 Agent 可用的工具列表，每个工具包含名称、描述和参数 schema
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

# ======== 模拟数据 ========
# 模拟各工具的返回数据，用于训练和评测
WEATHER_DATA = {"北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"), "深圳": ("30°C", "晴"),
                "杭州": ("22°C", "阴"), "成都": ("18°C", "小雨"), "武汉": ("25°C", "多云"), "南京": ("20°C", "晴"),
                "西安": ("16°C", "大风"), "重庆": ("26°C", "阴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"),
                "London": ("5°C", "小雨"), "Paris": ("10°C", "阴"), "Sydney": ("25°C", "晴朗")}
TIME_DATA = {"Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00",
             "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00",
             "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00"}
EXCHANGE_DATA = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12, ("JPY", "CNY"): 0.048,
                 ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79, ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72}
TRANSLATE_DATA = {("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好",
                  ("今天天气真好", "english"): "The weather is nice today",
                  ("I love programming", "chinese"): "我喜欢编程",
                  ("机器学习很有趣", "english"): "Machine learning is interesting",
                  ("Happy birthday", "chinese"): "生日快乐"}
UNIT_DATA = {"km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462, "pounds_kg": 0.453592,
             "meters_feet": 3.28084, "feet_meters": 0.3048, "celsius_fahrenheit": 1.8, "fahrenheit_celsius": 0.5556}

# ======== 模拟执行 ========
# 定义各工具的模拟执行函数，用于训练时的工具调用模拟
MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(eval(
        str(args.get("expression", "0")).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−",
                                                                                                        "-").replace(
            "（", "(").replace("）", ")"), {"__builtins__": {}, "math": math}))},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * UNIT_DATA.get(
        f"{args.get('from_unit', '').lower()}_{args.get('to_unit', '').lower()}", 1), 4)},
    "get_current_weather": lambda args: (
        lambda w: {"city": args.get("location"), "temperature": w[0], "humidity": "65%", "condition": w[1]})(
        WEATHER_DATA.get(args.get("location"), ("22°C", "晴"))),
    "get_current_time": lambda args: {
        "datetime": TIME_DATA.get(args.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00"),
        "timezone": args.get("timezone", "Asia/Shanghai")},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency"), "to": args.get("to_currency"),
                                       "rate": EXCHANGE_DATA.get((args.get("from_currency"), args.get("to_currency")),
                                                                 1.0)},
    "translate_text": lambda args: {
        "translated_text": TRANSLATE_DATA.get((args.get("text"), args.get("target_language")), args.get("text", ""))},
}

# ======== 参数校验 ========
# 定义各工具的参数校验函数，用于验证工具调用参数是否合法
CHECK_ARGS = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "unit_converter": lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: bool(a.get("location")),
    "get_current_time": lambda a: True,
    "get_exchange_rate": lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),
    "translate_text": lambda a: bool(a.get("text")) and bool(a.get("target_language")),
}


# ======== 工具调用解析与执行 ========
def parse_tool_calls(text):
    """从模型生成文本中解析工具调用请求
    
    模型通过 <tool_call>...</tool_call> 标签包裹 JSON 格式的工具调用信息，
    本函数提取所有标签内的 JSON 并解析为字典列表。
    
    Args:
        text: 模型生成的文本，可能包含一个或多个 <tool_call> 标签
    
    Returns:
        list[dict]: 解析成功的工具调用列表，每个元素包含 "name" 和 "arguments" 字段。
                    解析失败的条目会被静默跳过。
    """
    calls = []
    # 使用正则匹配所有 <tool_call>...</tool_call> 标签对中的内容（支持跨行）
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try:
            calls.append(json.loads(m.strip()))
        except Exception:
            pass  # JSON 解析失败则跳过该条目（容错处理）
    return calls


def execute_tool(name, args):
    """执行指定名称的 mock 工具，带 1 秒超时保护；失败/超时返回 None。
    
    使用 SIGALRM 信号实现超时保护，防止某些工具函数（如 eval 数学表达式）
    因死循环或极耗时计算阻塞训练流程。
    
    Args:
        name: 工具名称，需在 MOCK_RESULTS 中已注册
        args: 工具参数字典
    
    Returns:
        dict | None: 工具执行结果字典，若工具不存在、执行异常或超时则返回 None
    """
    fn = MOCK_RESULTS.get(name)
    if not fn:
        return None
    try:
        # 设置 SIGALRM 信号处理器，1 秒后触发 TimeoutError
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(1)  # 启动 1 秒计时器
        return fn(args)
    except Exception:
        return None  # 任何异常（含超时）均返回 None
    finally:
        try:
            signal.alarm(0)  # 取消未触发的计时器，避免影响后续逻辑
        except Exception:
            pass


# ======== 多轮 Rollout ========
def rollout_single(rollout_engine, tokenizer, messages, tools, max_turns=3, max_new_tokens=256, thinking_ratio=0.5,
                   device="cuda"):
    """单样本多轮工具调用 rollout
    
    模拟 Agent 与工具环境的多轮交互流程，实现完整的 tool-call 循环：
    
    **多轮交互机制**：
    - **第一轮（初始响应）**：根据用户消息生成初始响应，可能包含 tool_call 标签
    - **中间轮（工具执行）**：解析工具调用 -> 执行工具 -> 将结果作为 observation 追加到上下文 -> 继续生成
    - **最后轮（最终答案）**：当无工具调用或达到 max_turns 时，生成最终答案
    
    **状态追踪**：
    - response_ids: 拼接所有轮次的 token ids（模型生成 + 工具返回）
    - response_mask: 标记每个 token 的来源（1=模型生成参与 loss，0=工具返回不参与 loss）
    - response_old_logps: 记录旧策略的 log probability，用于重要性采样
    - unfinished: 标记是否因达到 max_turns 而仍有未完成的工具调用
    
    **思考模式**：
    - 按 thinking_ratio 概率开启  思考链
    - 思考部分计入 response_ids 但不单独区分 mask
    
    Args:
        rollout_engine: 推理引擎，负责生成候选响应
        tokenizer: 分词器，用于编码/解码文本
        messages: 对话消息列表（含用户问题和历史交互）
        tools: 可用工具列表（OpenAI function calling 格式）
        max_turns: 最大交互轮数，默认 3 轮
        max_new_tokens: 每轮最大生成长度，默认 256
        thinking_ratio: 开启思考模式的概率（0.0~1.0），默认 0.5
        device: 运行设备，默认 "cuda"
    
    Returns:
        tuple: (
            final_output: 最后一轮的模型输出文本
            final_context: 完整上下文字符串（含所有轮次）
            prompt_ids: 首轮 prompt token ids（后续轮不更新）
            response_ids: 所有轮次拼接后的 response token ids
            response_mask: response 掩码（1=模型生成，0=环境反馈）
            response_old_logps: 旧策略 log prob 列表
            turn_outputs: 每轮模型输出文本列表
            unfinished: 是否未完成标记（达到 max_turns 仍有工具调用）
        )
    """
    # ---- 状态变量初始化 ----
    all_outputs = []          # 每轮模型生成的文本列表
    prompt_ids = None         # 首轮的 prompt token ids（后续轮不更新）
    response_ids = []         # 所有轮次拼接后的 response token ids
    response_mask = []        # 与 response_ids 等长的 mask：1=模型生成的 token，0=工具返回/观察部分
    response_old_logps = []   # 与 response_ids 等长的旧策略 log prob（工具返回部分填 0）
    final_context = ""        # 最终完整上下文字符串
    unfinished = False        # 标记是否因达到 max_turns 而未完成（仍有待调用的工具）
    open_thinking = random.random() < thinking_ratio  # 按概率决定本次 rollout 是否开启思考模式

    for turn in range(max_turns):
        # Step 1: 将当前对话历史构造为完整 prompt 文本
        context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools,
                                                open_thinking=open_thinking)
        inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False).to(device)
        context_ids = inputs["input_ids"][0].tolist()
        if prompt_ids is None:
            prompt_ids = context_ids  # 仅第一轮记录 prompt ids

        # Step 2: 使用 rollout engine 生成一个候选响应
        rollout_result = rollout_engine.rollout(
            prompt_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_generations=1,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )

        # Step 3: 提取生成的 token ids 和对应的 log probabilities
        new_ids = rollout_result.completion_ids[0].tolist()
        new_logps = rollout_result.per_token_logps[0].tolist()
        if len(new_ids) != len(new_logps): Logger(
            f"rollout token/logprob length mismatch: {len(new_ids)} vs {len(new_logps)}")

        # 过滤掉 pad 和 eos token，只保留有效生成内容
        pairs = [(t, lp) for t, lp in zip(new_ids, new_logps) if
                 t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
        new_ids = [t for t, _ in pairs]
        new_logps = [lp for _, lp in pairs]
        new_text = rollout_result.completions[0]

        # Step 4: 累积本轮生成结果到全局状态
        all_outputs.append(new_text)
        response_ids.extend(new_ids)
        response_mask.extend([1] * len(new_ids))       # 模型生成的 token，mask=1（参与 loss 计算）
        response_old_logps.extend(new_logps)
        final_context = context + new_text

        # Step 5: 解析本轮生成中的工具调用
        calls = parse_tool_calls(new_text)
        if not calls:
            break  # 无工具调用，本次 rollout 结束

        # 若已到最后一轮但仍有工具调用，标记为 unfinished（会在 reward 中扣分）
        unfinished = turn == max_turns - 1

        # Step 6: 执行工具调用，将 assistant 响应和工具结果追加到消息列表
        messages.append({"role": "assistant", "content": new_text})
        for call in calls:
            name, raw = call.get("name", ""), call.get("arguments", {})
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = {}
            result = execute_tool(name, raw)
            result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[
                :2048]  # 截断防止超长结果撑爆 tokenizer
            messages.append({"role": "tool", "content": result_str})

        # Step 7: 构造包含工具返回结果的观察上下文，提取新增的 observation token ids
        # 这些 token 是工具返回的"环境反馈"，不参与 policy loss 计算（mask=0）
        observe_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished,
                                                        tools=tools, open_thinking=open_thinking)
        observe_ids = tokenizer(observe_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        current_len = len(prompt_ids) + len(response_ids)
        obs_delta = observe_ids[current_len:]           # 只取增量部分（工具返回对应的 token）
        response_ids.extend(obs_delta)
        response_mask.extend([0] * len(obs_delta))      # 工具返回部分 mask=0，不计入 policy gradient
        response_old_logps.extend([0.0] * len(obs_delta))  # 非模型生成，log prob 置 0
        final_context = observe_context

    final_output = all_outputs[-1] if all_outputs else ""
    prompt_ids = prompt_ids or []
    return final_output, final_context, prompt_ids, response_ids, response_mask, response_old_logps, list(
        all_outputs), unfinished


def rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, num_gen, max_turns=3, max_new_tokens=256,
                  thinking_ratio=0.5, device="cuda"):
    """批量多轮工具调用 rollout
    
    对多个样本进行 rollout，每个样本生成 num_gen 个候选响应
    
    Args:
        rollout_engine: 推理引擎
        tokenizer: 分词器
        messages_batch: 批量对话消息列表
        tools_batch: 批量可用工具列表
        num_gen: 每个样本生成的候选数量
        max_turns: 最大交互轮数
        max_new_tokens: 每轮最大生成长度
        thinking_ratio: 开启思考模式的概率
        device: 运行设备
    
    Returns:
        tuple: (all_completions, all_contexts, all_prompt_ids, all_response_ids, all_response_masks, all_response_old_logps, all_turn_outputs, all_unfinished)
    """
    # 输出容器：每个样本 × num_gen 个候选，扁平化存储
    all_completions = []          # 最终响应文本
    all_contexts = []             # 完整上下文（含多轮交互）
    all_prompt_ids = []           # 首轮 prompt token ids
    all_response_ids = []         # 完整 response token ids（含工具返回部分）
    all_response_masks = []       # response mask（1=模型生成，0=环境反馈）
    all_response_old_logps = []   # 旧策略 log prob
    all_turn_outputs = []         # 每轮模型输出文本列表
    all_unfinished = []           # 是否未完成标记

    for messages, tools in zip(messages_batch, tools_batch):
        # 对同一个 prompt 生成 num_gen 个独立候选（GRPO 组内对比需要）
        for _ in range(num_gen):
            msgs_copy = [dict(m) for m in messages]  # 深拷贝，避免多次 rollout 间相互污染
            completion, context, prompt_ids, response_ids, response_mask, response_old_logps, turn_outputs, unfinished = rollout_single(
                rollout_engine, tokenizer, msgs_copy, tools, max_turns, max_new_tokens, thinking_ratio, device)
            all_completions.append(completion)
            all_contexts.append(context)
            all_prompt_ids.append(prompt_ids)
            all_response_ids.append(response_ids)
            all_response_masks.append(response_mask)
            all_response_old_logps.append(response_old_logps)
            all_turn_outputs.append(turn_outputs)
            all_unfinished.append(unfinished)
    return all_completions, all_contexts, all_prompt_ids, all_response_ids, all_response_masks, all_response_old_logps, all_turn_outputs, all_unfinished


# ======== Reward 计算 ========
def validate_gt_in_text(text, gt_list):
    """验证文本中是否包含 ground truth 的数值
    
    采用两种匹配策略：
    1. 字符串子串匹配：gt 值的字符串形式出现在文本中（忽略大小写）
    2. 数值近似匹配：从文本中提取所有数字，判断与 gt 数值的差值是否 < 1e-6
    
    这种双重验证策略可以处理 "7.21" vs "7.2100" 等数值表示差异。
    
    Args:
        text: 待验证的模型最终回答文本
        gt_list: ground truth 列表（可包含字符串或数值）
    
    Returns:
        set: 在文本中成功匹配到的 gt 值集合
    """
    text, text_num = str(text), str(text).replace(',', '')
    # 从文本中提取所有独立的数字（整数或浮点数），用于后续数值比较
    nums = [float(x) for x in re.findall(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])', text_num)]
    return {g for g in gt_list if ((s := str(g).strip()) and s.lower() in text.lower()) or (
            re.fullmatch(r'[-+]?\d+(?:\.\d+)?', str(g).strip().replace(',', '')) and any(
        abs(float(str(g).strip().replace(',', '')) - n) < 1e-6 for n in nums))}


def calculate_rewards(prompts, completions, gt_batch, tools_batch, num_gen, reward_model=None, device="cuda",
                      turn_outputs_batch=None, unfinished_batch=None):
    """计算 Agent 响应的奖励
    
    奖励计算包含多个维度，针对有无工具调用采用不同的评分策略：
    
    **分支 A：无工具调用时**
    - 格式规范性：响应长度适中（5~800字符）、思考链完整且长度合理（20~300字符）
    - 标签闭合：<tool_call> 与 </tool_call> 数量匹配、</think> 只出现一次
    - Reward Model 评分：使用外部预训练模型对回答质量打分
    - n-gram 重复惩罚：防止生成重复内容
    
    **分支 B：有工具调用时**
    - 工具调用正确性：工具名在允许列表内 + 参数校验通过
    - 工具对齐分：合法调用数与 GT 期望数的差距 + 无效调用的额外惩罚
    - GT 命中比例：从最终回答中提取数值，验证是否包含 ground truth
    - 未完成惩罚：达到 max_turns 仍有工具调用则扣分
    - n-gram 重复惩罚
    
    **奖励范围**：所有奖励最终 clip 到 [-3, 3] 区间，防止极端值影响训练稳定性
    
    Args:
        prompts: 提示文本列表，用于提取对话历史供 Reward Model 使用
        completions: 生成响应列表（每个样本有 num_gen 个候选）
        gt_batch: ground truth 批次，期望出现在最终回答中的数值/文本列表
        tools_batch: 工具列表批次，定义每个样本可用的工具集合
        num_gen: 每个样本的生成数量（GRPO 组大小）
        reward_model: 奖励模型（可选），用于无工具调用时的质量评分
        device: 运行设备
        turn_outputs_batch: 多轮输出批次，记录每轮的模型输出
        unfinished_batch: 未完成标记批次，标识是否因达到 max_turns 而中断
    
    Returns:
        torch.Tensor: 奖励张量，形状为 [batch_size * num_gen]，每个元素对应一个候选的奖励值
    """
    rewards = torch.zeros(len(completions), device=device)
    for idx, response in enumerate(completions):
        reward, answer = 0.0, response
        sample_idx = idx // num_gen  # 映射回原始样本索引（每个样本有 num_gen 个候选）
        tools = tools_batch[sample_idx]
        turn_outputs = turn_outputs_batch[idx] if turn_outputs_batch is not None else [response]
        unfinished = unfinished_batch[idx] if unfinished_batch is not None else False

        # 提取每轮的实际回答内容（去掉 <think>...</think> 思考部分）
        turn_answers = [turn.split('</think>', 1)[-1].strip() if '</think>' in turn else turn.strip() for turn in
                        turn_outputs]
        answer = turn_answers[-1] if turn_answers else response.strip()

        # 构建合法工具名称集合，用于后续验证工具调用的合法性
        valid_names = {t['function']['name'] for t in tools} if tools else set()

        # 聚合所有轮次中的工具调用
        tool_calls = []
        for turn_answer in turn_answers: tool_calls.extend(parse_tool_calls(turn_answer))

        # 【惩罚项】标签不成对扣分：<tool_call> 与 </tool_call> 数量不匹配说明格式有误
        reward -= 0.5 * sum(
            abs(turn.count('<tool_call>') - turn.count('</tool_call>')) for turn in turn_answers)

        # -------- 分支 A: 无工具调用 —— 基于格式规范性 + Reward Model 评分 --------
        if not tool_calls:
            # 响应长度奖惩：过短或过长均扣分
            reward += 0.5 if 5 <= len(response.strip()) <= 800 else -0.5
            if '</think>' in response:
                think, answer = response.split('</think>', 1)
                # 思考链长度奖惩：鼓励适中长度的思考过程
                reward += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5
                # 思考标签闭合奖惩：只允许出现一次 </think>
                reward += 0.25 if response.count('</think>') == 1 else -0.25
                answer = answer.strip()
            if reward_model is not None:
                # 使用外部 Reward Model（如 InternLM2-Reward）对回答质量评分
                prompt = prompts[sample_idx]
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                score = reward_model.get_score(messages, answer)
                reward += score
            # n-gram 重复惩罚：防止模型生成重复内容
            reward -= rep_penalty(answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)  # 总奖励 clip 到 [-3, 3] 防止极端值

        # -------- 分支 B: 有工具调用 —— 基于工具调用正确性和执行结果 --------
        else:
            gt = gt_batch[sample_idx]  # 当前样本的 ground truth 预期值列表

            # 统计合法的工具调用数量（工具名在允许列表内 + 参数校验通过）
            valid_call_count = 0
            for tool_call in tool_calls:
                name, raw = tool_call.get("name", ""), tool_call.get("arguments", {})
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except Exception:
                        raw = {}
                check = CHECK_ARGS.get(name)
                valid_call_count += int(bool(name in valid_names and check and check(raw)))

            # 工具对齐分：合法调用数与 GT 期望数的差距 + 无效调用的额外惩罚
            tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)
            reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap

            # 提取最终回答文本（工具调用标签之后的内容），用于验证 GT 是否出现
            final_text = "" if unfinished else (
                answer.split('</tool_call>')[-1] if '</tool_call>' in answer else answer)
            verified = validate_gt_in_text(final_text, gt) if gt else set()
            if gt: reward += 2.5 * len(verified) / len(gt)  # GT 命中比例奖励（满分 2.5）
            if unfinished: reward -= 0.5  # 达到最大轮数仍未完成，扣分
            reward -= rep_penalty(final_text if final_text else answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)  # 总奖励 clip 到 [-3, 3]
    return rewards


# ================================ 工具与 Reward = End ================================
def rl_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model=None, start_step=0, wandb=None,
                   use_sglang=False):
    """执行一个 epoch 的强化学习训练
    
    完整的 on-policy 强化学习训练循环，包含以下 8 个阶段：
    
    **Phase 1: Rollout（候选生成）**
    - 无梯度模式下使用旧策略生成 num_gen 个候选响应
    - 支持多轮工具调用交互，每个候选可能包含多次 tool-call
    
    **Phase 2: 构造训练张量**
    - 将 rollout 结果打包为统一格式：(token_ids, mask, prompt_len, old_logps)
    - prompt 部分 mask=0，response 部分保留原 mask
    - 超长序列从右侧截断，确保 response 尾部完整
    
    **Phase 3: 前向计算当前策略的 log prob**
    - 计算当前策略在每个 token 位置的 log probability
    - 同时计算参考模型（冻结）的 log prob，用于 KL 散度约束
    
    **Phase 4: 构造 completion mask**
    - 处理 EOS 截断：EOS 之后的 token 不参与 loss 计算
    - 找到每个序列中第一个 EOS token 的位置，将其后 mask 置 0
    
    **Phase 5: 计算奖励**
    - 调用 calculate_rewards 计算每个候选的奖励值
    - 支持 debug 模式输出详细的生成内容和奖励信息
    
    **Phase 6: 计算 GRPO 组内优势 (Advantage)**
    - GRPO 核心思想：同一 prompt 的 num_gen 个候选组成一组
    - 组内标准化得到相对优势：advantages = (rewards - mean) / std
    
    **Phase 7: 计算 Policy Loss**
    - KL 散度惩罚：使用 Schulman 近似公式 KL ≈ exp(r) - r - 1
    - 重要性采样比 ratio = π_new / π_old
    - 支持两种 loss 类型：
      - CISPo：只 clamp ratio 上界，保守截断
      - GRPO/PPO：双边截断 [1-ε, 1+ε]，取 min 实现悲观下界
    
    **Phase 8: 梯度累积与参数更新**
    - 梯度累积：每 accumulation_steps 步更新一次参数
    - 梯度裁剪、优化器步进、学习率调度
    - 同步最新策略权重到 rollout 引擎（on-policy 要求）
    
    Args:
        epoch: 当前 epoch 编号
        loader: 数据加载器，提供训练批次
        iters: 总迭代次数（用于日志显示）
        rollout_engine: 推理引擎，负责生成候选响应
        ref_model: 参考模型（冻结），用于 KL 散度约束
        reward_model: 奖励模型（可选），用于无工具调用时的质量评分
        start_step: 起始步数（断点续训时使用）
        wandb: wandb 日志记录器（可选）
        use_sglang: 是否使用 SGLang 引擎进行 rollout
    """
    last_step = start_step
    for step, batch in enumerate(loader, start=start_step + 1):
        messages_batch = batch['messages']
        tools_batch = batch['tools']
        gt_batch = batch['gt']  # ground truth：期望出现在最终回答中的数值/文本
        last_step = step

        # ======== Phase 1: Rollout —— 生成候选响应 ========
        # 无梯度模式下使用旧策略生成多个候选，用于后续 on-policy 优化
        with torch.no_grad():
            completions, contexts, prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch, turn_outputs_batch, unfinished_batch = rollout_batch(
                rollout_engine, tokenizer, messages_batch, tools_batch, args.num_generations, max_turns=3,
                max_new_tokens=args.max_gen_len, thinking_ratio=args.thinking_ratio, device=args.device)

        # ======== Phase 2: 构造训练张量 ========
        # 还原每个样本的 prompt 文本（用于 reward 计算时提取对话历史）
        prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, tools=t) for m, t in
                   zip(messages_batch, tools_batch)]

        # 将 rollout 结果打包为统一格式：(token_ids, mask, prompt_len, old_logps)
        packed_samples = []
        for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch, response_masks_batch,
                                   response_old_logps_batch):
            ids = p + r                                    # 完整序列 = prompt + response
            mask = [0] * len(p) + m                        # prompt 部分 mask=0，response 部分保留原 mask
            old_logps = [0.0] * max(len(p) - 1, 0) + old_lp  # prompt 部分 logp=0（偏移 1 位对齐 next-token 预测）
            if len(ids) > args.max_total_len:
                # 超长截断：从右侧保留，确保 response 尾部完整
                ids = ids[-args.max_total_len:]
                mask = mask[-args.max_total_len:]
                old_logps = old_logps[-(len(ids) - 1):]
            # 找到 mask 中第一个 1 的位置，即 response 起始位置
            prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask))
            packed_samples.append((ids, mask, prompt_len, old_logps))

        # 构造 batch 张量，padding 到 batch 内最大长度
        seq_lens = torch.tensor([len(ids) for ids, _, _, _ in packed_samples], device=args.device)
        max_len = seq_lens.max().item()
        input_ids = torch.tensor(
            [ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _, _, _ in packed_samples],
            device=args.device)
        prompt_lens = torch.tensor([prompt_len for _, _, prompt_len, _ in packed_samples], device=args.device)
        full_response_masks = torch.tensor([mask + [0] * (max_len - len(mask)) for _, mask, _, _ in packed_samples],
                                           device=args.device, dtype=torch.float32)
        old_per_token_logps = torch.tensor(
            [old_logps + [0.0] * ((max_len - 1) - len(old_logps)) for _, _, _, old_logps in packed_samples],
            device=args.device, dtype=torch.float32)

        # ======== Phase 3: 前向计算当前策略的 log prob ========
        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res = model_unwrapped(input_ids)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            logits = res.logits[:, :-1, :]  # 去掉最后一个位置（无对应 target）
            # 计算当前策略在每个 token 位置的 log probability
            per_token_logps = F.log_softmax(logits, dim=-1).gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        # 计算参考模型（冻结）的 log prob，用于 KL 散度约束
        with torch.no_grad():
            ref_per_token_logps = compute_per_token_logps(ref_model, input_ids, input_ids.size(1) - 1)

        # ======== Phase 4: 构造 completion mask（处理 EOS 截断） ========
        # completion_mask 偏移 1 位对齐 next-token 预测视角
        completion_mask = full_response_masks[:, 1:]
        # 找到每个序列中第一个 EOS token 的位置，EOS 之后的 token 不参与 loss
        is_eos = (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool()
        eos_idx = torch.full((completion_mask.size(0),), completion_mask.size(1) - 1, device=args.device,
                             dtype=torch.long)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        # 将 EOS 之后的位置 mask 置 0
        pos = torch.arange(completion_mask.size(1), device=args.device).unsqueeze(0)
        completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
        token_counts = completion_mask.sum(dim=1)  # 每个样本的有效 token 数
        valid_rows = token_counts > 0  # 过滤空序列

        # ======== Phase 5: 计算奖励 ========
        rewards = calculate_rewards(prompts, completions, gt_batch, tools_batch, args.num_generations, reward_model,
                                    device=args.device, turn_outputs_batch=turn_outputs_batch,
                                    unfinished_batch=unfinished_batch)

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(messages_batch)):
                Logger(f"[DEBUG] step={step}, gt[{i}]: {repr(gt_batch[i])}")
                Logger('-' * 100)
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    plen, slen = prompt_lens[idx].item(), seq_lens[idx].item()
                    Logger(f"{'=' * 30} [DEBUG] gen[{i}][{j}] CONTEXT_BEGIN {'=' * 30}")
                    Logger(contexts[idx])
                    Logger(f"{'=' * 31} [DEBUG] gen[{i}][{j}] CONTEXT_END {'=' * 31}")
                    Logger(f"[DEBUG] gen[{i}][{j}] prompt_len={plen}, seq_len={slen}")
                    tokens = input_ids[idx, plen:slen].tolist()
                    text = tokenizer.decode(tokens, skip_special_tokens=False)
                    Logger(f"{'=' * 28} [DEBUG] gen[{i}][{j}] COMPLETION_BEGIN [{plen}:{slen}] {'=' * 28}")
                    Logger(text)
                    Logger(f"{'=' * 29} [DEBUG] gen[{i}][{j}] COMPLETION_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{i}][{j}] reward={rewards[idx].item():.4f}")
                    Logger('=' * 100)

        # ======== Phase 6: 计算 GRPO 组内优势 (Advantage) ========
        # GRPO 核心思想：同一 prompt 的 num_gen 个候选组成一组，组内标准化得到相对优势
        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # 组内均值
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)  # 组内标准差
        advantages = (rewards - mean_r) / (std_r + 1e-4)  # 标准化优势，+1e-4 防除零

        # ======== Phase 7: 计算 Policy Loss ========
        # KL 散度惩罚：使用 Schulman 的近似公式 KL ≈ exp(r) - r - 1，其中 r = ref_logp - policy_logp
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        # 重要性采样比 ratio = π_new / π_old
        ratio = torch.exp(per_token_logps - old_per_token_logps)

        if args.loss_type == "cispo":
            # CISPo (Clipped Importance Sampling Policy Optimization)：
            # 只 clamp ratio 上界，相当于对"变化过大"的更新做保守截断
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            # 标准 GRPO/PPO clip：对 ratio 做双边截断 [1-ε, 1+ε]
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            # 取 min 实现悲观下界（PPO 的核心机制）
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)

        # 对每个样本在有效 token 上取平均 loss，再对 batch 取均值
        policy_loss = (
            ((per_token_loss * completion_mask).sum(dim=1)[valid_rows] / token_counts[valid_rows].clamp(min=1)).mean()
            if valid_rows.any() else per_token_loss.sum() * 0.0)
        # 加上 MoE 辅助 loss（负载均衡），除以梯度累积步数
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        # ======== Phase 8: 梯度累积与参数更新 ========
        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step();
            scheduler.step();
            optimizer.zero_grad()
            # 同步最新策略权重到 rollout 引擎（用于下一轮 on-policy 生成）
            if is_main_process() and step % args.save_interval == 0: rollout_engine.update_policy(model)

        # ======== 日志记录 ========
        if step % args.log_interval == 0 or step == iters:
            pl = loss.item() * args.accumulation_steps       # 还原真实 policy loss（去除累积因子）
            ar = rewards.mean().item()                       # 平均奖励
            al = token_counts.float().mean().item()          # 平均响应长度（有效 token 数）
            kl = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(
                token_counts.sum().item(), 1)                # 平均 KL 散度
            gs = grouped_rewards.std(dim=1, unbiased=False).mean().item()  # 组内奖励标准差（衡量探索多样性）
            am, ast = advantages.mean().item(), advantages.std().item()    # 优势均值和标准差
            lr = optimizer.param_groups[0]['lr']
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), Reward:{ar:.4f}, KL:{kl:.4f}, GrpStd:{gs:.4f}, AdvStd:{ast:.4f}, Loss:{pl:.4f}, AvgLen:{al:.2f}, AdvMean:{am:.4f}, LR:{lr:.8f}')
            if wandb and is_main_process():
                wandb.log({"reward": ar, "kl_ref": kl, "group_reward_std": gs, "advantages_std": ast, "policy_loss": pl,
                           "avg_response_len": al, "advantages_mean": am, "learning_rate": lr})

        # ======== 定期保存 checkpoint ========
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            # 统一保存：推理权重 + 训练状态（Agent RL 需保存 scheduler 以支持断点续训）
            save_checkpoint(model, lm_config, args.save_dir, args.save_weight,
                            optimizer=optimizer, scheduler=scheduler,
                            epoch=epoch, step=step, wandb=wandb)
            model.train()

        # 主动释放大张量，减少显存碎片
        del per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask

    # 处理最后不足一个累积周期的残余梯度
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step();
        scheduler.step();
        optimizer.zero_grad()
        if is_main_process() and last_step % args.save_interval == 0: rollout_engine.update_policy(model)


if __name__ == "__main__":
    # ================================ 命令行参数解析 ================================
    parser = argparse.ArgumentParser(description="MiniMind Agent RL")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='agent', type=str, help="保存权重名称")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="学习率")
    parser.add_argument("--device", type=str, default=get_default_device(), help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型 bfloat16/float16")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="模型隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="模型层数")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="最大序列长度")
    parser.add_argument("--max_gen_len", type=int, default=768, help="单次最大生成长度")
    parser.add_argument("--max_total_len", type=int, default=2500, help="训练侧最终总长度上界")
    parser.add_argument("--data_path", type=str, default="../.dataset/agent_rl.jsonl", help="训练数据路径")
    parser.add_argument("--num_generations", type=int, default=4, help="每个prompt生成数量")
    parser.add_argument("--beta", type=float, default=0.1, help="KL散度惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="加载预训练权重名称")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否从checkpoint恢复")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Agent-RL", help="wandb项目名称")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile")
    parser.add_argument("--debug_mode", action="store_true", help="调试模式")
    parser.add_argument("--debug_interval", type=int, default=20, help="调试日志间隔")
    parser.add_argument("--thinking_ratio", type=float, default=0.1, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument("--rollout_engine", type=str, default="sglang", choices=["torch", "sglang"],
                        help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_agent", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ================================ 分布式 & 随机种子初始化 ================================
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))  # 不同 rank 使用不同种子保证探索多样性
    Logger(f'Training device: {args.device}')

    # ================================ 模型配置与 checkpoint 加载 ================================
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe),
                               num_key_value_heads=2)
    # 断点续训：从 checkpoint 恢复模型权重、优化器、scheduler 状态
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

    device_type = get_device_type(args.device)

    # MPS 上 F.scaled_dot_product_attention 性能极差（forward 慢 15x，backward 慢 100x+），
    # 强制关闭 flash_attn，使用手动 attention 实现
    if device_type == "mps" and lm_config.flash_attn:
        lm_config.flash_attn = False
        Logger('⚡ MPS: flash_attn disabled (SDPA is extremely slow on MPS, using manual attention)')

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if device_type == "cuda":
        autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
    elif device_type == "mps":
        autocast_ctx = torch.autocast(device_type="mps", dtype=dtype)
    else:
        autocast_ctx = nullcontext()

    # 统一工具配置 wandb，自动支持断点续训
    wandb = setup_wandb(args, ckp_data, run_name_prefix="Agent-RL")

    # ================================ 模型初始化 ================================
    # 策略模型（可训练）：从 SFT 权重初始化
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    # 参考模型（冻结）：用于计算 KL 散度约束，防止策略偏离太远
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    # 奖励模型：使用外部预训练的 Reward Model 对无工具调用响应进行质量评分
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    Logger(f'Loaded reward model from {args.reward_model_path}')

    # Rollout 引擎：支持 torch（本地推理）和 sglang（高性能服务化推理）两种模式
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )

    # ================================ 数据集与优化器 ================================
    train_ds = AgentRLDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)


    def collate_fn(batch):
        """Agent RL 数据集的 collate 函数：将样本按字段分组为批次字典"""
        return {'messages': [b['messages'] for b in batch], 'tools': [b['tools'] for b in batch],
                'gt': [b['gt'] for b in batch]}


    # 计算总优化器步数，用于 CosineAnnealing 学习率调度
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    # ================================ 断点续训恢复 ================================
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ================================ 编译与分布式包装 ================================
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        # RoPE 的频率张量不需要 DDP 同步（各 rank 相同且不可训练）
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    if is_main_process():
        rollout_engine.update_policy(model)  # 初始同步策略权重到 rollout 引擎

    # ================================ 训练循环 ================================
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # DDP 下确保每个 epoch 的 shuffle 不同
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        # 断点续训时跳过已训练的 step
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers,
                            pin_memory=(device_type == "cuda"), collate_fn=collate_fn)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: skip {start_step} steps')
            rl_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step,
                           wandb, use_sglang=(args.rollout_engine == "sglang"))
        else:
            rl_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb,
                           use_sglang=(args.rollout_engine == "sglang"))

    if dist.is_initialized():
        dist.destroy_process_group()
