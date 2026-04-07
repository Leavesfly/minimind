# -*- coding: utf-8 -*-
"""
OpenAI 兼容 API 聊天客户端
==============================================
本脚本实现了一个简单的命令行聊天客户端，通过 OpenAI 兼容的 API 与 MiniMind 模型交互。

主要功能：
- 支持流式和非流式输出
- 支持多轮对话历史
- 支持思考模式（Thinking）的显示
- 支持工具调用（Tool Call）

使用方法：
1. 确保 MiniMind API 服务已启动（如使用 serve_openai_api.py）
2. 修改 base_url 为实际的 API 地址
3. 运行脚本开始对话

作者：MiniMind Team
"""

from openai import OpenAI

# 创建 OpenAI 客户端
client = OpenAI(
    api_key="sk-123",  # API 密钥（实际使用时需要修改）
    base_url="http://localhost:11434/v1"  # API 服务地址
)

# 是否使用流式输出
stream = True

# 对话历史
conversation_history_origin = []
conversation_history = conversation_history_origin.copy()

# 历史消息数量（必须为偶数，表示 Q+A 对数，0 表示不携带历史对话）
history_messages_num = 0

client = OpenAI(
    api_key="sk-123",
    base_url="http://localhost:11434/v1"
)
stream = True
conversation_history_origin = []
conversation_history = conversation_history_origin.copy()
history_messages_num = 0  # 必须设置为偶数（Q+A），为0则不携带历史对话
# 主循环：持续接收用户输入并生成响应
while True:
    # 获取用户输入
    query = input('[Q]: ')
    
    # 将用户消息添加到对话历史
    conversation_history.append({"role": "user", "content": query})
    
    # 调用 API 生成响应
    response = client.chat.completions.create(
        model="minimind-local:latest",  # 模型名称
        messages=conversation_history[-(history_messages_num or 1):],  # 使用最近的对话历史
        stream=stream,  # 是否流式输出
        temperature=0.8,  # 生成温度
        max_tokens=2048,  # 最大生成长度
        top_p=0.8,  # nucleus 采样阈值
        extra_body={"chat_template_kwargs": {"open_thinking": True}, "reasoning_effort": "medium"} # 思考开关
    )
    
    # 处理非流式输出
    if not stream:
        assistant_res = response.choices[0].message.content
        print('[A]: ', assistant_res)
    # 处理流式输出
    else:
        print('[A]: ', end='', flush=True)
        assistant_res = ''
        for chunk in response:
            delta = chunk.choices[0].delta
            # 思考内容（灰色显示）
            r = getattr(delta, 'reasoning_content', None) or ""
            # 正常内容
            c = delta.content or ""
            if r:
                print(f'\033[90m{r}\033[0m', end="", flush=True)
            if c:
                print(c, end="", flush=True)
            assistant_res += c

    # 将助手响应添加到对话历史
    conversation_history.append({"role": "assistant", "content": assistant_res})
    print('\n\n')