from streamingcheck.integration.ds_patch import apply_deepspeed_patch
apply_deepspeed_patch()


import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import re
from transformers import HfArgumentParser, TrainingArguments, set_seed, DataCollatorForLanguageModeling, TrainerCallback
from trl import SFTTrainer, SFTConfig
import torch
from torch.optim import AdamW
from torch import nn  # Also needed for nn.Module type hint
from collections import defaultdict
import deepspeed
from datasets import load_from_disk
from utilsLLM import create_and_prepare_model, create_datasets
# from my_layer_flag_cb import CheckpointLayerPrinter
# from streamingcheck.integration.hf_callbacks import MyLayerFlagCB
from transformers.trainer_utils import is_main_process
import torch.distributed as dist
from streamingcheck.integration.hf_trainer import StreamingTrainer as Trainer
from streamingcheck.integration.hf_callbacks import StreamingCheckpointFlowCallback
from streamingcheck.integration.optim.adamx import AdamX
from accelerate import Accelerator
import inspect

cb = StreamingCheckpointFlowCallback(should_save_steps_before=10, early_interval=100)

def unwrap_optimizer(opt):
    while hasattr(opt, "optimizer"):
        opt = opt.optimizer
    return opt

class MyLayerFlagCB(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % state.save_steps == 0 or state.global_step <= 10:
            return control

        adamw = unwrap_optimizer(kwargs["optimizer"])
        local_layers = getattr(adamw, "layers_to_checkpoint", [])
        local_flag   = bool(local_layers)

        if dist.is_initialized():
            flag = torch.tensor([int(local_flag)], device=args.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            global_should = bool(flag.item())
        else:
            global_should = local_flag

        if not global_should:
            return control     

        if dist.is_initialized():
            world_size = dist.get_world_size()
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_layers)
            layers_to_save = next((lst for lst in gathered if lst), [])
        else:
            layers_to_save = local_layers
        control.should_save_layers = layers_to_save
        control.should_save = True          
        return control


# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "chatml|zephyr|none. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training."},
    )



@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    dataset_path: Optional[str] = field(
        default="/lvs0/rccs-hpbdrt/minqiu/local_dataset",
        metadata={"help": "Path to local dataset. Will be used instead of dataset_name if provided."},
    )
    tokenized_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-tokenized dataset. If provided, will load from this path instead of tokenizing again."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    dataset_text_field: str = field(default="text", metadata={"help": "Dataset field to use as input text."})
    max_seq_length: Optional[int] = field(default=512)
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed."},
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tokenizers adds special tokens to each sample being packed."},
    )
    splits: Optional[str] = field(
        default="train,test",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    tokenized_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-tokenized dataset. If provided, will load from this path instead of tokenizing again."},
    )

def get_parameter_groups(model: nn.Module, weight_decay: float):
    """
    Organize parameters into groups:
    1) no_decay parameters divided by layers
    2) decay parameters divided by layers
    3) other no_decay parameters (that don't match layer pattern)
    4) other decay parameters (that don't match layer pattern)
    """
    no_decay = ["bias", "layernorm", "Layer_norm", "norm"]
    attn = ["attn"]
    mlp = ["mlp"]

    # Instead of a single no_decay list, use a dict similar to layer_decay_dict
    no_decay_dict = defaultdict(lambda: {"params": []})
    layer_decay_dict = defaultdict(lambda: {"params": []})
    attn_dict = defaultdict(lambda: {"params": []})
    mlp_dict = defaultdict(lambda: {"params": []})

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Match layer pattern first
        match = re.search(r'(?:^|\.)(layers\.(\d+))\.', name)
        layer_key = int(match.group(2)) if match else name

        # Then check if parameter should have weight decay
        if any(nd in name for nd in no_decay):
            no_decay_dict[layer_key]["params"].append(param)
        else:
            layer_decay_dict[layer_key]["params"].append(param)

    # Construct final param_groups list
    param_groups = []
    param_group_id = 0  # Add a counter for param_group_id

    # Helper function to sort keys (layers first, then "other")
    def sort_keys(x): return (isinstance(x, int), x)

    # Add no_decay groups by layer
    sorted_keys = sorted(no_decay_dict.keys(), key=sort_keys)
    for key in sorted_keys:
        group_params = no_decay_dict[key]["params"]
        if group_params:  # Only add if there are parameters in this group
            param_groups.append({
                "params": group_params,
                "weight_decay": 0.0,
                "layer": key,
                "decoupled_weight_decay": False,
                "param_group_id": param_group_id
            })
            param_group_id += 1

    # Add decay groups by layer
    sorted_keys = sorted(layer_decay_dict.keys(), key=sort_keys)
    for key in sorted_keys:
        group_params = layer_decay_dict[key]["params"]
        if group_params:  # Only add if there are parameters in this group
            param_groups.append({
                "params": group_params,
                "weight_decay": weight_decay,
                "layer": key,
                "decoupled_weight_decay": True,
                "param_group_id": param_group_id
            })
            param_group_id += 1

    return param_groups


def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # 如果 unwrap_model 不接受 keep_torch_compile，就给它打个补丁
    sig = inspect.signature(Accelerator.unwrap_model)
    if "keep_torch_compile" not in sig.parameters:
        original = Accelerator.unwrap_model

        def patched_unwrap_model(self, model, *args, **kwargs):
            # 丢掉 keep_torch_compile，或者其他多余参数
            kwargs.pop("keep_torch_compile", None)
            return original(self, model, *args, **kwargs)

        Accelerator.unwrap_model = patched_unwrap_model

    # model
    model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args, training_args)

    if model_args.use_flash_attn:
        # Move model to GPU if not already
        if torch.cuda.is_available():
            model = model.to('cuda')
        else:
            raise RuntimeError("CUDA is not available but Flash Attention requires GPU.")

    # class CausalLMDataCollator(DataCollatorForLanguageModeling):
    #     def __call__(self, features, return_tensors=None):
    #         batch = super().__call__(features, return_tensors=return_tensors)
    #         pad_id = tokenizer.pad_token_id
    #         labels = batch["labels"]
    #         labels[labels == pad_id] = -100
    #         return batch

    class CausalLMDataCollator(DataCollatorForLanguageModeling):
        def __call__(self, features, return_tensors=None):
            batch = super().__call__(features, return_tensors=return_tensors)
            # —— 保险丝：确保整型字段为 long
            for k in ("input_ids", "attention_mask", "labels", "position_ids"):
                if k in batch and isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].long()
            # 忽略 pad 的标签
            pad_id = tokenizer.pad_token_id
            if "labels" in batch:
                batch["labels"][batch["labels"] == pad_id] = -100
            return batch

    collator = CausalLMDataCollator(tokenizer, mlm=False)

    # gradient ckpt
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = training_args.gradient_checkpointing and not model_args.use_unsloth
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}

    # datasets
    train_dataset, eval_dataset = create_datasets(
        tokenizer,
        data_args,
        training_args,
        apply_chat_template=model_args.chat_template_format != "none",
        use_sft_format=False,
    )

    param_groups = get_parameter_groups(model, training_args.weight_decay)
    # print(f"param_groups: {param_groups}")

    # optimizer = AdamW(param_groups, lr=training_args.learning_rate, layers_to_checkpoint=[])
    optimizer = AdamX(
    param_groups,
    lr=1e-5,
    layer_num=28,
    K=100.0,
    threshold_end=1e-5,
    layers_to_checkpoint=[], 
    update_log_path="logs/updates.jsonl",
    thr_log_path="logs/thr.jsonl",
    flag_log_path="logs/flags.jsonl",
    only_rank0=True,
)
    
    trainer = Trainer(
        model=model,
        # tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # optimizers=(optimizer, None),
        optimizers=(optimizer, None),
        callbacks=[
        cb,   # 固定步保存（优先级最高）
        # LayerSaveController(layers_to_save=[-1, 1, 2, 32, 33]),  # 如需手动指定
    ],
        data_collator=collator,
        args=TrainingArguments(
            # max_seq_length=data_args.max_seq_length,
            bf16=training_args.bf16,
            num_train_epochs=training_args.num_train_epochs,
            learning_rate=training_args.learning_rate,
            lr_scheduler_type=training_args.lr_scheduler_type,
            weight_decay=training_args.weight_decay,
            warmup_ratio=training_args.warmup_ratio,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            gradient_checkpointing=training_args.gradient_checkpointing,
            max_grad_norm=training_args.max_grad_norm,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
            logging_steps=training_args.logging_steps,
            log_level=training_args.log_level,
            logging_strategy=training_args.logging_strategy,
            save_strategy=training_args.save_strategy,
            save_steps=training_args.save_steps,
            eval_strategy=training_args.eval_strategy,
            eval_steps=training_args.eval_steps,
            # dataset_text_field=data_args.dataset_text_field,
            # dataset_kwargs={
            #     "append_concat_token": data_args.append_concat_token,
            #     "add_special_tokens": data_args.add_special_tokens,
            # },
            output_dir=training_args.output_dir,
            # packing=data_args.packing,
            # remove_unused_columns=False,
        ),
    )

    cp_cb = MyLayerFlagCB()
    cp_cb.trainer_ref = trainer

    trainer.add_callback(cp_cb)    

    orig_prepare_inputs = trainer._prepare_inputs
    def safe_prepare_inputs(inputs):
        out = orig_prepare_inputs(inputs)
        # —— 保险丝：再次矫正（避免外层 .to(dtype) 误伤）
        for k in ("input_ids", "attention_mask", "labels", "position_ids"):
            if k in out and isinstance(out[k], torch.Tensor):
                out[k] = out[k].long()
        return out
    trainer._prepare_inputs = safe_prepare_inputs  # monkey patch

    trainer.accelerator.print(f"{trainer.model}")
    if hasattr(trainer.model, "print_trainable_parameters()"):
        trainer.model.print_trainable_parameters()

    # train
    # checkpoint = "/lvs0/rccs-hpbdrt/minqiu/Llama3.1-8B_merged_models/merge5" # Change checkpoint path !!!!!
    # checkpoint = "/lvs0/rccs-hpbdrt/minqiu/Qwen2.5-7B_merged_models/merge4"
    # checkpoint = "/lvs0/rccs-hpbdrt/minqiu/Qwen2.5-7B_merged_models/merge-test"
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    
    trainer.train(resume_from_checkpoint=checkpoint)

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
