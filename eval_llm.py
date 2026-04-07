import argparse
import random
import time
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

from model.model_lora import *
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params

warnings.filterwarnings('ignore')


def init_model(args):
    """
    初始化并加载 MiniMind 模型和分词器
    
    根据用户指定的参数，加载不同格式和类型的模型权重。支持两种加载方式：
    1. 原生 torch 权重：从本地加载 .pth 权重文件，适用于自定义训练的模型
    2. transformers 格式：使用 Hugging Face transformers 库加载，兼容标准格式
    
    加载流程：
    1. 加载分词器（Tokenizer），用于文本编码和解码
    2. 根据加载路径判断权重格式
    3. 如果是原生 torch 权重：
       - 创建 MiniMind 模型实例，配置隐藏层维度、层数、是否使用 MoE 等
       - 加载对应的 .pth 权重文件
       - 如果指定了 LoRA 权重，注入 LoRA 并加载 LoRA 权重
    4. 如果是 transformers 格式：
       - 直接使用 AutoModelForCausalLM 加载
    5. 打印模型参数信息
    6. 将模型转换为半精度（FP16）、设置为评估模式、移动到指定设备
    
    Args:
        args: 命令行参数对象，包含以下关键字段：
            - load_from: 模型加载路径
            - hidden_size: 隐藏层维度
            - num_hidden_layers: 隐藏层数量
            - use_moe: 是否使用 MoE 架构
            - inference_rope_scaling: 是否启用 RoPE 外推
            - save_dir: 模型权重保存目录
            - weight: 权重名称前缀
            - lora_weight: LoRA 权重名称
            - device: 运行设备（cuda/cpu）
    
    Returns:
        tuple: (model, tokenizer)
            - model: 加载并配置好的 MiniMind 模型
            - tokenizer: 对应的分词器
    """
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        # 加载原生 torch 权重格式
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        # 如果指定了 LoRA 权重，注入并加载 LoRA
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        # 加载 transformers 格式权重
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer


def main():
    """
    主函数：解析命令行参数并启动对话交互
    
    支持两种对话模式：
    1. 自动测试模式（input_mode=0）：使用预设的测试问题列表，自动进行多轮对话
    2. 手动输入模式（input_mode=1）：用户实时输入问题，进行交互式对话
    
    对话流程：
    1. 解析命令行参数
    2. 初始化模型和分词器
    3. 选择对话模式（自动测试或手动输入）
    4. 进入对话循环：
       - 设置随机种子（确保生成可复现）
       - 处理历史对话（根据 args.historys 携带指定轮数的历史）
       - 构造输入提示（支持预训练格式和对话格式）
       - 模型生成（使用流式输出）
       - 记录对话历史
       - 显示推理速度
    
    命令行参数说明：
    """
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str,
                        help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str,
                        help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str,
                        help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true',
                        help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()

    # 自动测试模式的预设问题列表
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]

    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 根据选择模式构造提示迭代器
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    # 对话循环：处理每个用户输入
    for prompt in prompt_iter:
        # 设置随机种子，确保生成结果的可复现性
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        # 处理历史对话：根据 args.historys 保留最近 N 轮对话
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        # 构造输入提示：预训练模型使用简单格式，微调模型使用对话模板
        if 'pretrain' in args.weight:
            inputs = tokenizer.bos_token + prompt
        else:
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True,
                                                   open_thinking=bool(args.open_thinking))

        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🧠: ', end='')
        st = time.time()
        # 模型生成：使用流式输出，实时显示生成的 token
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
        # 解码生成的回复（排除输入部分）
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        # 计算并显示推理速度
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')


if __name__ == "__main__":
    main()
