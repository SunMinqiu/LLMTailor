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
@inproceedings{sun2025llmtailor,
  author    = {Minqiu Sun and Xin Huang and Luanzheng Guo and Nathan R. Tallent and Kento Sato and Dong Dai},
  title     = {{LLMTailor: A Layer-wise Tailoring Tool for Efficient Checkpointing of Large Language Models}},
  booktitle = {Proceedings of the 10th International Parallel Data Systems Workshop (PDSW'25)},
  year      = {2025},
  note      = {To appear},
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
>> **Note:** The goal of LLMTailor is a tool that support merging layer-wise checkpoints. If you only want to merge default checkpoints, please comment the first part of code in start_merge.py 
2. Modify the YAML file to whatever you like.
3. Modify the configuration in the top of this start_merge.py file. (e.g. CHECKPOINT_PATH)
4. Run this python file.