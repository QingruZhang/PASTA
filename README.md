<h1 align="center"> 🍝 PASTA: Post-hoc Attention Steering for LLMs 🍝 </h1>
<p align="center"> <b>Tell Your Model Where to Attend: Post-hoc Attention Steering for LLMs</b> (<a href="https://arxiv.org/pdf/2311.02262.pdf">Zhang et al. 2023</a>). 
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.7+-blue">
  <img src="https://img.shields.io/pypi/v/pastalib?color=green">  
</p>  

<p align="center"> PASTA allows a user to improve LLM controllability by simply emphasizing part of the prompt (e.g. the instruction) that the LLM should focus on. It requires no changes to LLM weights and no increase in inference time.
</p>

## Quickstart -- use PASTA for improved inference

1. Install `pastalib`:

```bash
pip install pastalib
# Alternatively,  
# clone then pip install -e .
# pip install git+https://github.com/QingruZhang/PASTA
```

2. Initialize a pre-trained LLM and PASTA.
 
```python
from pastalib.pasta import PASTA 
from transformers import AutoModelForCausalLM,AutoTokenizer

# Initialize pre-trained LLM
name = "huggyllama/llama-7b"
model = AutoModelForCausalLM.from_pretrained(name)
tokenizer = AutoTokenizer.from_pretrained(name)

# Select the attention heads to be steered, 
# following the format of {'layer_id': [head_ids]}: 
head_config = {
    "3": [17, 7, 6, 12, 18], "8": [28, 21, 24], "5": [24, 4], 
    "0": [17], "4": [3], "6": [14], "7": [13], "11": [16], 
}

# Initialize the PASTA steerer
pasta = PASTA(
    model=model,
    tokenizer=tokenizer,
    head_config=head_config, 
    alpha=0.01, # scaling coefficient
    scale_position="exclude", # downweighting unselected tokens
)
```

3. Select specific input spans to emphasize, and then run inference as normal.

```python
# Model Input 
texts = ["Mary is a doctor. She obtains her bachelor degree from UCSD. Answer the occupation of Mary and generate the answer as json format."]

# ===== Without PASTA =====
# inputs = tokenizer(texts, return_tensors="pt")
# outputs = model.generate(**inputs)
# ---------------------
# ["The answer should be in json format."]  # returns answer in the wrong format

# ===== With PASTA =====
inputs, offset_mapping = pasta.inputs_from_batch(texts)
# User highlights specific input spans
emphasized_texts = ["Answer the occupation of Mary and generate the answer as json format"]
# PASTA registers the pre_forward_hook to edit attention
with pasta.apply_steering(
    model=model, 
    strings=texts, 
    substrings=emphasized_texts, 
    model_input=inputs, 
    offsets_mapping=offset_mapping
) as steered_model: 
    outputs = steered_model.generate(**inputs, max_new_tokens=128)
# -------------------------------
# ['{"name": "Mary", "occupation": "Doctor", ...}']  # returns answer in the correct format
```

### Additional Note 

1. `pastalib` works with any models that apply causal attention by summing up query-key inner product with attention mask and can be applied to LLMs in a plug-and-play manner. For example, `LlamaForCausalLM` and `GPTJForCausalLM` from `transformers`, whose attention moduels apply attention masks following `torch.matmul(query, key) + attention_mask`. However, `pastalib` currently only supports LLAMA, LLAMA-2 and GPT-J (more models in progress!). 

2. We provide different options of `head_config` for LLAMA-7B and GPT-J in the folder of [config/head_config](config/head_config), including multi-task, task-agnostic and task-specific settings. Please see detailed discussion in our paper. 

## Overview

The overview of this repo is as follows: 

* [`pastalib`](pastalib): contains the source code of PASTA libary, which can be applied to models from huggingface `transformers`.  
* [`evaluation`](evaluation): consists of evaluation pipelines for different tasks, including data/model preprocessing and task evaluators/metrices. 
* [`config`](config): includes the `head_config` for steering attention modules of LLAMA-7B and GPT-J with PASTA. 
* [`scripts`](scripts): consists of running scripts of four tasks: JSON Formatting, Pronouns Changing, Bias in Bios, and CounterFact. 

## Evaluation 

The evaluation pipeline are mainly refactored from [REMEDI repo](https://github.com/evandez/REMEDI). Please see more details there. 

### Environment Setup 

Set up the environment with the following commands: 
```bash
conda create -n pasta python=3.10 
pip install -r requirements.txt 
pip install -e . 

python -m spacy download en_core_web_sm
python -W ignore -m nltk.downloader punkt cmudict
```

By default, the preprocessed datasets, models, and results are saved in the local directory of `./data`, `./models`, and `./results`. You can change the directory of their by setting the environment variables: 
```bash 
export CM_DATA_DIR=<data path> 
export CM_MODELS_DIR=<models path>
export CM_RESULTS_DIR=<results path>
```

### Dataset Setup

1. For CounterFact, our scripts can automatically download the dataset. 

2. For Bias in Bios, we cannot release the dataset without the authorization. The dataset must be downloaded with [the official release](https://github.com/microsoft/biosbias). After downloading the data examples into the `BIOS.pkl` file, you can run the following scripts: 
```bash
python reformat_dataset.py \
--biasbios_raw_path <path of BIOS.pkl> \
--biasbios_save_file biasbios.json 
```

Then, the preprocessed biasbios dataset file will be saved in `CM_DATA_DIR/biasbios.json`.  

### Evaluation 

Choose any `head_config` files from `config/head_config` and evaluate the performance of PASTA with the following command. 

**JSON Formatting**

```bash
python -m scripts.eval_biasbios_instruction \
--task json \
--apply_pasta \
--emphasized_text instruct \
--alpha 0.01 \
--scale_position exclude \
--pasta_head_config <head_config_path> \
--model huggyllama/llama-7b \
--prompt_idx 0 \
--batch_size 16 \
--max_new_tokens 128 \
--experiment_name llama_evaluation \
--device cuda 
```

**Pronouns Changing**

```bash 
python -m scripts.eval_biasbios_instruction \
--task pronchange \
--apply_pasta \
--emphasized_text instruct \
--alpha 0.01 \
--scale_position exclude \
--pasta_head_config <head_config_path> \
--prompt_idx 0 \
--model huggyllama/llama-7b \
--max_new_tokens 128 \
--batch_size 16 \
--experiment_name llama_evaluation \
--device cuda 
```

**Bias in Bios**

```bash 
python -m scripts.eval_bias_gen \
--model huggyllama/llama-7b \
--apply_pasta \
--alpha 0.01 \
--scale_position exclude \
--pasta_head_config <head_config_path> \
--max_length 256 \
--batch_size 16 \
--experiment_name llama_evaluation \
--device cuda 
```

**CounterFact**

```bash 
python -m scripts.eval_fact_gen \
--model huggyllama/llama-7b \
--apply_pasta \
--alpha 0.01 \
--scale_position exclude \
--pasta_head_config <head_config_path> \
--add_unmediated_fact True \
--benchmarks efficacy paraphrase generation \
--experiment_name llama_evaluation
```


## Contact

Please contact us or post an issue if you have any questions: 

* Qingru Zhang (qingru.zhang@gatech.edu) 
* Chandan Singh (chansingh@microsoft.com)
* Liyuan Liu (lucliu@microsoft.com) 
* Xiaodong Liu (xiaodl@microsoft.com)


```r
@misc{zhang2023tell,
    title={Tell Your Model Where to Attend: Post-hoc Attention Steering for LLMs}, 
    author={Qingru Zhang and Chandan Singh and Liyuan Liu and Xiaodong Liu and Bin Yu and Jianfeng Gao and Tuo Zhao},
    year={2023},
    eprint={2311.02262},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```



