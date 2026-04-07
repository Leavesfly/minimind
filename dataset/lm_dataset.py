import json
import os
import random

import torch
from datasets import load_dataset, Features, Value
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class PretrainDataset(Dataset):
    """
    预训练数据集，初始化时一次性 tokenize 全部数据并缓存为 tensor。

    针对 Apple Silicon 统一内存架构优化：
    - 传入 device='mps' 时，数据直接放在 GPU 上，训练时零拷贝
    - vocab_size < 32767 时自动使用 int16 存储，内存减少 75%
    - __getitem__ 只做 tensor 索引，纳秒级返回

    多核并行优化：
    - 使用 datasets.map(num_proc=N) 多进程并行 tokenize，充分利用多核 CPU
    - 自动检测 CPU 核数，默认使用 min(cpu_count, 16) 个进程
    
    数据处理流程：
    1. 加载 JSON 格式的预训练数据
    2. 多进程并行 tokenize，添加 bos/eos token
    3. padding 到固定长度，生成 labels（padding 位置设为 -100）
    4. 转换为 tensor 并缓存，可选择存储到 GPU
    5. __getitem__ 直接返回缓存的 tensor，无需实时 tokenize
    """

    def __init__(self, data_path, tokenizer, max_length=512, device=None, num_proc=None):
        """
        初始化预训练数据集
        
        Args:
            data_path: 预训练数据文件路径（JSON 格式）
            tokenizer: 分词器实例
            max_length: 最大序列长度
            device: 数据存储设备，'mps' 时数据直接放在 GPU 上
            num_proc: 多进程数量，None 时自动检测
        """
        super().__init__()
        self.max_length = max_length
        raw_samples = load_dataset('json', data_files=data_path, split='train')

        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id

        # vocab_size < 32767 时用 int16 存储，节省 75% 内存
        # labels 中有 -100（ignore_index），int16 范围 [-32768, 32767] 可以容纳
        use_compact = (tokenizer.vocab_size < 32767)
        storage_dtype = torch.int16 if use_compact else torch.long

        # 自动检测 CPU 核数，限制上限避免内存压力过大
        if num_proc is None:
            num_proc = min(os.cpu_count() or 1, 16)
        # 样本数太少时多进程反而有 fork 开销，退化为单进程
        if len(raw_samples) < num_proc * 100:
            num_proc = 1

        print(f'Pre-tokenizing {len(raw_samples)} samples (storage: {storage_dtype}, workers: {num_proc})...',
              flush=True)

        def tokenize_and_pad(batch):
            """
            批量 tokenize + padding + label 生成，供 datasets.map 多进程调用
            
            Args:
                batch: 包含 'text' 字段的批次数据
            
            Returns:
                包含 'input_ids' 和 'labels' 的字典
            """
            batch_input_ids = []
            batch_labels = []
            for text in batch['text']:
                tokens = tokenizer(
                    str(text),
                    add_special_tokens=False,
                    max_length=max_length - 2,
                    truncation=True
                ).input_ids
                tokens = [bos_id] + tokens + [eos_id]
                padding_len = max_length - len(tokens)
                input_ids = tokens + [pad_id] * padding_len
                labels = [(-100 if token_id == pad_id else token_id) for token_id in input_ids]
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
            return {'input_ids': batch_input_ids, 'labels': batch_labels}

        tokenized = raw_samples.map(
            tokenize_and_pad,
            batched=True,
            batch_size=1000,
            num_proc=num_proc,
            remove_columns=raw_samples.column_names,
            desc='Tokenizing',
        )

        # 转为 tensor（datasets 的 Arrow 格式可以零拷贝转换）
        tokenized.set_format('torch')
        self.input_ids = tokenized['input_ids'].to(storage_dtype)
        self.labels = tokenized['labels'].to(storage_dtype)

        mem_mb = (self.input_ids.nbytes + self.labels.nbytes) / 1024 ** 2
        print(f'Pre-tokenize done. Shape: {self.input_ids.shape}, memory: {mem_mb:.1f}MB', flush=True)

        # 统一内存架构：数据直接放到 GPU 上，训练时零拷贝
        if device is not None and str(device) != 'cpu':
            print(f'Moving dataset to {device} (unified memory, zero-copy access)...', flush=True)
            self.input_ids = self.input_ids.to(device)
            self.labels = self.labels.to(device)

    def __len__(self):
        """返回数据集样本数量"""
        return self.input_ids.shape[0]

    def __getitem__(self, index):
        """
        获取单个样本
        
        Args:
            index: 样本索引
        
        Returns:
            (input_ids, labels) 元组，input_ids 和 labels 都是 long 类型 tensor
        """
        return self.input_ids[index].long(), self.labels[index].long()


class SFTDataset(Dataset):
    """
    SFT 数据集，初始化时一次性并行 tokenize 全部数据并缓存为 tensor。

    多核并行优化：
    - 使用 datasets.map(num_proc=N) 多进程并行完成 chat 预处理 + tokenize + label 生成
    - 自动检测 CPU 核数，默认使用 min(cpu_count, 16) 个进程
    - __getitem__ 只做 tensor 索引，纳秒级返回
    
    数据处理流程：
    1. 加载 JSONL 格式的对话数据
    2. 预处理对话消息（处理 tools、tool_calls 字段）
    3. 应用 chat template 生成 prompt 文本
    4. Tokenize 并 padding 到固定长度
    5. 生成 labels：只有 assistant 回复区域标记为对应的 token id，其他区域标记为 -100
    6. 缓存为 tensor，训练时直接返回
    """

    def __init__(self, jsonl_path, tokenizer, max_length=1024, num_proc=None):
        """
        初始化 SFT 数据集
        
        Args:
            jsonl_path: SFT 数据文件路径（JSONL 格式）
            tokenizer: 分词器实例
            max_length: 最大序列长度
            num_proc: 多进程数量，None 时自动检测
        """
        super().__init__()
        self.max_length = max_length
        features = Features({'conversations': [
            {'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'),
             'tools': Value('string'), 'tool_calls': Value('string')}]})
        raw_samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        pad_id = tokenizer.pad_token_id

        # 自动检测 CPU 核数，限制上限避免内存压力过大
        if num_proc is None:
            num_proc = min(os.cpu_count() or 1, 16)
        if len(raw_samples) < num_proc * 100:
            num_proc = 1

        print(f'Pre-tokenizing SFT {len(raw_samples)} samples (workers: {num_proc})...', flush=True)

        def _create_chat_prompt(conversations, tok):
            """
            将 conversations 转为 chat prompt 文本
            
            Args:
                conversations: 对话消息列表
                tok: 分词器实例
            
            Returns:
                应用 chat template 后的 prompt 文本
            """
            messages = []
            tools = None
            for message in conversations:
                message = dict(message)
                if message.get("role") == "system" and message.get("tools"):
                    tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
                if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                    message["tool_calls"] = json.loads(message["tool_calls"])
                messages.append(message)
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, tools=tools
            )

        def _generate_labels(input_ids, bos_marker, eos_marker, max_len):
            """
            根据 bos/eos marker 生成 labels，非 assistant 区域标记为 -100
            
            Args:
                input_ids: 输入 token ids
                bos_marker: assistant 开始标记（如 '<|assistant|>' 的 token ids）
                eos_marker: assistant 结束标记（如 '<|end|>' 的 token ids）
                max_len: 最大长度
            
            Returns:
                labels 数组，assistant 区域为对应的 token id，其他区域为 -100
            """
            labels = [-100] * len(input_ids)
            i = 0
            while i < len(input_ids):
                if input_ids[i:i + len(bos_marker)] == bos_marker:
                    start = i + len(bos_marker)
                    end = start
                    while end < len(input_ids):
                        if input_ids[end:end + len(eos_marker)] == eos_marker:
                            break
                        end += 1
                    for j in range(start, min(end + len(eos_marker), max_len)):
                        labels[j] = input_ids[j]
                    i = end + len(eos_marker) if end < len(input_ids) else len(input_ids)
                else:
                    i += 1
            return labels

        def tokenize_sft_batch(batch):
            """
            批量处理：chat 预处理 → template → tokenize → label 生成 → padding
            
            Args:
                batch: 包含 'conversations' 字段的批次数据
            
            Returns:
                包含 'input_ids' 和 'labels' 的字典
            """
            batch_input_ids = []
            batch_labels = []
            for conversations in batch['conversations']:
                conversations = pre_processing_chat(conversations)
                prompt = _create_chat_prompt(conversations, tokenizer)
                prompt = post_processing_chat(prompt)
                input_ids = tokenizer(prompt).input_ids[:max_length]
                input_ids += [pad_id] * (max_length - len(input_ids))
                labels = _generate_labels(input_ids, bos_id, eos_id, max_length)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
            return {'input_ids': batch_input_ids, 'labels': batch_labels}

        tokenized = raw_samples.map(
            tokenize_sft_batch,
            batched=True,
            batch_size=1000,
            num_proc=num_proc,
            remove_columns=raw_samples.column_names,
            desc='Tokenizing SFT',
        )

        tokenized.set_format('torch')
        self.input_ids = tokenized['input_ids'].clone()
        self.labels = tokenized['labels'].clone()

        mem_mb = (self.input_ids.nbytes + self.labels.nbytes) / 1024 ** 2
        print(f'SFT pre-tokenize done. Shape: {self.input_ids.shape}, memory: {mem_mb:.1f}MB', flush=True)

    def __len__(self):
        """返回数据集样本数量"""
        return self.input_ids.shape[0]

    def __getitem__(self, index):
        """
        获取单个样本
        
        Args:
            index: 样本索引
        
        Returns:
            (input_ids, labels) 元组
        """
        return self.input_ids[index], self.labels[index]


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }


class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    pass
