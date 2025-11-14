# LLMTailor

LLMTailor is an enhanced fork of [mergekit](https://github.com/arcee-ai/mergekit), designed for **layer-wise merging of large language models (LLMs)** with extended support for:

✅ Compatible with our new checkpoint system StreamCheck
✅ Layer-wise model merging & selection  
✅ Optimizer state reconstruction (supports ZeRO-3 shards)  
✅ Tokenizer & embedding adaptation: these auxiliary layers in LLMs could also be selected and merged now
✅ Backward compatibility with most `mergekit` plans  

> **Note:** LLMTailor retains most of `mergekit`’s original merging capabilities while adding extensions (`llmtailor.*` fields in YAML) for training-oriented scenarios.

---

## Citing LLMTailor
The relevant research paper will be published at PDSW25. If you reference or use LLMTailor in your research, please cite:
```
@inproceedings{10.1145/3731599.3767515,
author = {Sun, Minqiu and Huang, Xin and Guo, Luanzheng and Tallent, Nathan R. and Sato, Kento and Dai, Dong},
title = {LLMTailor: A Layer-wise Tailoring Tool for Efficient Checkpointing of Large Language Models},
year = {2025},
isbn = {9798400718717},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3731599.3767515},
doi = {10.1145/3731599.3767515},
abstract = {Checkpointing is essential for fault tolerance in training large language models (LLMs). However, existing methods, regardless of their I/O strategies, periodically store the entire model and optimizer states, incurring substantial storage overhead and resource contention. Recent studies reveal that updates across LLM layers are highly non-uniform. Across training steps, some layers may undergo more significant changes, while others remain relatively stable or even unchanged. This suggests that selectively checkpointing only layers with significant updates could reduce overhead without harming training. Implementing such selective strategies requires fine-grained control over both weights and optimizer states, which no current tool provides. To address this gap, we propose LLMTailor, a checkpoint-merging framework that filters and assembles layers from different checkpoints to form a composite checkpoint. Our evaluation indicates that LLMTailor can work with different selective checkpointing strategies and effectively reduce checkpoint size (e.g., 4.3 times smaller for Llama3.1-8B) and checkpoint time (e.g., 2.8 times faster for Qwen2.5-7B) while maintaining model quality.},
booktitle = {Proceedings of the SC '25 Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis},
pages = {1366–1374},
numpages = {9},
keywords = {Checkpoint, Large Language Model, I/O optimization},
location = {
},
series = {SC Workshops '25}
}
```

## Installation
### Required Software
- Python 3.11
```
conda create -n myenv python=3.11

conda activate myenv
```

- Clone From GitHub
```bash
git clone https://github.com/SunMinqiu/LLMTailor.git
cd LLMTailor
pip install -r requirements.txt
pip install -e .
```

- Benchmark Running
For the benchmark, we use the open source project called [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main). Please follow the instructions of this project to install.

### Required Hardware
- GPU: Recommend at least one node of 8 * L40s, or 4 * H100.
- CPU: At least 64 cores.
- Memory: At least 200 GB.
- Storage: Depending on the model and training epochs, recommend at least at least 350 GB for a 7B model and 700 GB for 14B model.

## Quick Start
1. The example can be found in the /examples folder.
> **Note:** The goal of LLMTailor is a tool that support merging layer-wise checkpoints. If you only want to merge default checkpoints, please comment the first part of code in start_merge.py 
2. Modify the YAML file to whatever you like.
3. Modify the configuration in the top of this start_merge.py file. (e.g. CHECKPOINT_PATH)
4. Run this python file.
