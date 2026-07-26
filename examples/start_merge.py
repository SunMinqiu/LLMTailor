import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 加入上级目录
import torch
import yaml
import json
from collections import defaultdict

from llmtailor.config import MergeConfiguration
from llmtailor.merge import MergeOptions, run_merge
from llmtailor.create_yaml import generate_merge_config_from_log

import argparse
from pathlib import Path

def _str2bool(s: str) -> bool:
    return s.lower() in {"1", "true", "t", "yes", "y", "on"}

def parse_args():
    p = argparse.ArgumentParser(description="Simple CLI config parser")

    p.add_argument("--output_path", default="merged", type=str)
    p.add_argument("--lora_merge_cache", default="/tmp/llmtailor_cache", type=str)
    p.add_argument("--config_yml", default="examples/Qwen_cut.yaml", type=str)
    p.add_argument("--copy_tokenizer", default=True, type=_str2bool,
                   help="true/false")
    p.add_argument("--lazy_unpickle", default=False, type=_str2bool,
                   help="true/false")
    p.add_argument("--low_cpu_memory", default=False, type=_str2bool,
                   help="true/false")
    p.add_argument("--num_gpu", default=8, type=int)
    p.add_argument("--streamcheck_json", default=None, type=str,
                   help="Path to checkpoint_flags.jsonl from StreamCheck")
    p.add_argument("--streamcheck_path", default=None, type=str,
                   help="Path to checkpoints directory")
    p.add_argument("--base_step", default=0, type=int,
                   help="Base checkpoint step. Layers with ckpt <= base_step use this checkpoint")
    p.add_argument("--failure_step", default=160, type=int,
                   help="Failure step. Find each layer's nearest checkpoint before this step")
    p.add_argument("--total_layers", default=28, type=int,
                   help="Total number of transformer layers in the model")
    p.add_argument("--dry_run", action="store_true",
                   help="Only generate YAML config, don't run merge")
    p.add_argument("--weight_tie", default=False, type=_str2bool,
                   help="true/false")
    args = p.parse_args()

    return args


def main(args):
    # Only generate config from log if both streamcheck_json and streamcheck_path are provided
    if args.streamcheck_json and args.streamcheck_path:
        # Read JSON data from file
        with open(args.streamcheck_json, "r") as f:
            json_data = []
            for line in f:
                json_data.append(json.loads(line.strip()))

        # Generate config file
        generate_merge_config_from_log(
            json_data,
            args.streamcheck_path,
            args.config_yml,
            args.base_step,     # base_step (layers with ckpt <= this use base checkpoint)
            args.failure_step,  # end_ckpt (max step to consider)
            args.total_layers,   # num_layers
            args.weight_tie     # weight_tie
        )
        print(f"Config generated: {args.config_yml}")

    # If dry_run, stop here after generating YAML
    if args.dry_run:
        print("--dry_run enabled. Skipping merge and trainer_state update.")
        print(f"You can review the config at: {args.config_yml}")
        print("Run again without --dry_run to execute the merge.")
        return

    # Load and validate the merge configuration
    with open(args.config_yml, "r", encoding="utf-8") as fp:
        merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))

    print(f"merge_config: {merge_config}")

    run_merge(
        merge_config,
        out_path=args.output_path,
        options=MergeOptions(
            lora_merge_cache=args.lora_merge_cache,
            cuda=torch.cuda.is_available(),
            copy_tokenizer=args.copy_tokenizer,
            lazy_unpickle=args.lazy_unpickle,
            low_cpu_memory=args.low_cpu_memory,
            num_gpus=args.num_gpu,
        ),
        failure_step=args.failure_step,
    )
    print("Merge Done!")

    # modify trainer_state.json to make it compatible with StreamCheck
    path = args.output_path + "/trainer_state.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)

        data["global_step"] = args.failure_step

        with open(path, "w") as f:
            json.dump(data, f)

        print("Trainer State Done!")
    else:
        print(f"Warning: {path} not found, skipping trainer_state update.")

if __name__ == "__main__":
    args = parse_args()
    main(args)