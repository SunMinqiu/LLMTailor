import os
from enum import Enum

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig

DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'' + message['role'] + '\n' + message['content'] + '' + '\n'}}{% if loop.last and add_generation_prompt %}{{'assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '' }}\n{% endif %}\n{% endfor %}"
DEFAULT_LLAMA3_CHAT_TEMPLATE = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

class Llama3SpecialTokens(str, Enum):
    user = "<|start_header_id|>user<|end_header_id|>"
    assistant = "<|start_header_id|>assistant<|end_header_id|>"
    system = "<|start_header_id|>system<|end_header_id|>"
    eos_token = "<|eot_id|>"
    bos_token = "<|begin_of_text|>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

class ZephyrSpecialTokens(str, Enum):
    user = ""
    assistant = ""
    system = ""
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

class ChatmlSpecialTokens(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    eos_token = ""
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

def format_sft(example):
    # 处理为空的 input
    prompt = example["instruction"].strip()
    if example["input"]:
        prompt += "\n\n" + example["input"].strip()

    answer = example["output"].strip()

    # 创建标准的消息格式
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer}
    ]
    # print(messages[0])
    return {"messages": messages}

def create_datasets(tokenizer, data_args, training_args, apply_chat_template=False, use_sft_format=False):

    # 1. 先加载原始数据集
    raw_datasets = DatasetDict()
    for split in data_args.splits.split(","):
        dataset = load_dataset(data_args.dataset_name, cache_dir=data_args.dataset_path, split=split)
        raw_datasets[split.strip()] = dataset

    # SFT格式化
    if use_sft_format:
        raw_datasets = raw_datasets.map(
            format_sft,
            remove_columns=["instruction", "input", "output", "__index_level_0__"],
        )


    # 2. 应用 chat template
    if apply_chat_template:
        def preprocess(samples):
            batch = [tokenizer.apply_chat_template(conv, tokenize=False) for conv in samples["messages"]]
            return {"content": batch}
        raw_datasets = raw_datasets.map(
            preprocess,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            num_proc=32,
        )

    # 3. tokenize 成 input_ids
    def tokenize_function(samples):
        result = tokenizer(
            samples["article"],
            truncation=True,
            max_length=data_args.max_seq_length,
            padding=False,
        )
        return result

    raw_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        num_proc=32,
    )

    def is_non_empty(example):
        if "input_ids" not in example:
            return False
        if isinstance(example["input_ids"], list):
            if len(example["input_ids"]) == 0:
                return False
        else:
            if int(len(example["input_ids"])) == 0:
                return False
        if "attention_mask" in example:
            return any(x == 1 for x in example["attention_mask"])
        # no attention_mask is allowed, but at least one token is required
        return True

    raw_datasets = raw_datasets.filter(is_non_empty, num_proc=32)

    train_data = raw_datasets["train"]
    valid_data = raw_datasets.get("test", None)
    if valid_data is None:
        raw_datasets = raw_datasets["train"].train_test_split(test_size=0.1)
        train_data = raw_datasets["train"]
        valid_data = raw_datasets["test"]

    # if data_args.tokenized_dataset_path is not None:
    #     print(f"Saving tokenized dataset to {data_args.tokenized_dataset_path}")
    #     to_save = DatasetDict({"train": train_data, "test": valid_data})
    #     to_save.save_to_disk(data_args.tokenized_dataset_path)

    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    print(f"A sample of train dataset: {train_data[0]}")

    return train_data, valid_data

def create_and_prepare_model(args, data_args, training_args):
    if args.use_unsloth:
        from unsloth import FastLanguageModel
    bnb_config = None
    quant_storage_dtype = None

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and args.use_unsloth
    ):
        raise NotImplementedError("Unsloth is not supported in distributed training")

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    if args.use_unsloth:
        # Load model
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            cache_dir="",
            max_seq_length=data_args.max_seq_length,
            dtype=None,
            load_in_4bit=args.use_4bit_quantization,
        )
        print("use_unsloth=True"),
    else:
        torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir="/lvs0/rccs-hpbdrt/minqiu/local_models",
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
            torch_dtype=torch_dtype,
        )

    peft_config = None
    chat_template = None
    if args.use_peft_lora and not args.use_unsloth:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
        )

    special_tokens = None
    chat_template = None
    if args.chat_template_format == "chatml":
        special_tokens = ChatmlSpecialTokens
        chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
    elif args.chat_template_format == "zephyr":
        special_tokens = ZephyrSpecialTokens
        chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE
    elif args.chat_template_format == "llama3":
        special_tokens = Llama3SpecialTokens
        chat_template = DEFAULT_LLAMA3_CHAT_TEMPLATE

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir="/lvs0/rccs-hpbdrt/minqiu/local_models",
            pad_token=special_tokens.pad_token.value,
            bos_token=special_tokens.bos_token.value,
            eos_token=special_tokens.eos_token.value,
            additional_special_tokens=special_tokens.list(),
            trust_remote_code=True,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})

        model.resize_token_embeddings(len(tokenizer))

        model.config.pad_token_id = tokenizer.pad_token_id
        tokenizer.chat_template = chat_template
        # make embedding resizing configurable?
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=False)
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir="", trust_remote_code=True, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        print("special_tokens is None")

    if args.use_unsloth:
        # Do model patching and add fast LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            random_state=training_args.seed,
            max_seq_length=data_args.max_seq_length,
        )

    def randomize_weights(model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.zeros_(param)

    # randomize_weights(model)

    return model, peft_config, tokenizer