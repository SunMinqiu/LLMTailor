import torch
import yaml
import json
from collections import defaultdict

from llmtailor.config import MergeConfiguration
from llmtailor.merge import MergeOptions, run_merge
from llmtailor.create_yaml import generate_merge_config_from_log

OUTPUT_PATH = "/lvs0/rccs-hpbdrt/minqiu/Qwen2.5-7B_merged_models/wrap_test"  # folder to store the result in
LORA_MERGE_CACHE = "/lvs0/rccs-hpbdrt/minqiu/tmp"  # change if you want to keep these for some reason
CONFIG_YML = "examples/Qwen_cut.yaml"  # merge configuration file
COPY_TOKENIZER = True  # you want a tokenizer? yeah, that's what i thought
LAZY_UNPICKLE = False  # experimental low-memory model loader
LOW_CPU_MEMORY = False  # enable if you somehow have more VRAM than RAM+swap
NUM_GPU = 8 # change to your own number of GPUs
STREAMCHECK_JSON = "/home/users/u0001609/StreamingCheck/logs/flags.jsonl"
STREAMCHECK_PATH = "/lvs0/rccs-hpbdrt/minqiu/Wrap/Qwen2.5-7B-test1"
FAILURE_STEP = 160
TOTAL_LAYERS = 28

# Read JSON data from file
with open(STREAMCHECK_JSON, "r") as f:
    json_data = []
    for line in f:
        json_data.append(json.loads(line.strip()))

# Generate config file
generate_merge_config_from_log(
    json_data,
    STREAMCHECK_PATH,
    CONFIG_YML,
    STREAMCHECK_PATH + "/checkpoint-100", # base_path, must be earlier than FAILURE_STEP
    FAILURE_STEP,
    TOTAL_LAYERS
)

with open(CONFIG_YML, "r", encoding="utf-8") as fp:
    merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))

print(f"merge_config: {merge_config}")

run_merge(
    merge_config,
    out_path=OUTPUT_PATH,
    options=MergeOptions(
        lora_merge_cache=LORA_MERGE_CACHE,
        cuda=torch.cuda.is_available(),
        copy_tokenizer=COPY_TOKENIZER,
        lazy_unpickle=LAZY_UNPICKLE,
        low_cpu_memory=LOW_CPU_MEMORY,
        num_gpus=NUM_GPU,
    )
)
print("Merge Done!")

# modify trainer_state.json to make it compatible with StreamCheck
path = OUTPUT_PATH + "/trainer_state.json"
with open(path, "r") as f:
    data = json.load(f)

data["global_step"] = FAILURE_STEP

with open(path, "w") as f:
    json.dump(data, f)

print("Trainer State Done!")