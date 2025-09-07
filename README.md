# MedVLMBench: A Unified Benchmark for Generalist and Specialist Medical Vision-Language Models

MedVLMBench is the first unified benchmark for systematically evaluating generalist and medical-specialist Vision-Language Models (VLMs). This repository provides the code and resources to reproduce the experiments and extend the benchmark.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Download Datasets and Models](#download-datasets-and-models)
- [Available Models and Datasets](#available-models-and-datasets)
  - [Datasets](#datasets)
  - [Models](#models)
- [Usage](#usage)
  - [Notebook Tutorials](#notebook-tutorials)
  - [Command-Line Interface](#command-line-interface)
- [Abstract](#abstract)
- [Citation](#citation)


## Abstract

>**Background:** Vision–Language Models (VLMs) have shown promise in automating image diagnosis and interpretation in clinical settings. However, developing medical-specialist VLMs requires substantial computational resources and carefully curated datasets, and it remains unclear under which conditions generalist and medical specialist VLMs each perform best.

>**Methods:** This paper introduces MedVLMBench, the first unified benchmark for systematically evaluating generalist and medical-specialist VLMs. We assessed 18 models spanning contrastive and generative paradigms on 10 publicly available datasets across radiology, pathology, dermatology, and ophthalmology, encompassing 144 diagnostic and 80 VQA settings. MedVLMBench focusing on assessing both in-domain (ID) and out-of-domain (OOD) performance, with off-the-shelf and parameter-efficient fine-tuning (e.g., linear probing, LoRA). Diagnostic classification tasks were evaluated using AUROC, while visual question answering (VQA) tasks were assessed with BLEU-1, ROUGE-L, Exact Match, F1 Score, and GPT-based semantic scoring, covering both open- and closed-ended formats. Computational efficiency was estimated relative to the cost of full medical pretraining.

>**Results:** As expected, off-the-shelf medical VLMs generally outperformed generalist VLMs on ID tasks given their pretraining. However, with lightweight fine-tuning, general-purpose VLMs achieved superior performance in most of ID task evaluations and demonstrated better generalization on OOD tasks in approximately all comparisons. Fine-tuning required only 3% of the total parameters associated with full medical pretraining. In contrast, fine-tuned medical VLMs showed degraded performance even on ID tasks when subjected to rigorous hyperparameter optimization, further highlighting their limited adaptability.

>**Conclusions:** This study highlights the complementary strengths of medical-specialist and generalist VLMs. Specialists remain valuable in modality-aligned use cases, but we find that efficiently fine-tuned generalist VLMs can achieve comparable or even superior performance in most tasks, particularly when transferring to unseen or rare OOD medical modalities. These results suggest that generalist VLMs, rather than being constrained by their lack of medical-specific pretraining, may offer a scalable and cost-effective pathway for advancing clinical AI development.


## Getting Started

### Prerequisites

Ensure you have an environment with Python and the necessary dependencies. You can set up a conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate medvlmbenc
```

### Installation

Clone the repository:

```bash
git clone https://github.com/Nanboy-Ronan/MedVLMBench.git
cd MedVLMBench
```

### Download Datasets and Models

All pretrained models should be stored under `MedVLMBench/pretrained_models`, and all data should be stored under `MedVLMBench/data`.

```bash
mkdir pretrained_models
mkdir data
```

**Example: Downloading LLaVA**

```bash
cd pretrained_models
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
cd ..
```

## Available Models and Datasets

<details>
<summary><b>Supported Datasets</b></summary>

| Dataset | Type | Status |
|---|---|---|
| SLAKE | VQA | Done |
| PathVQA | VQA | Done |
| VQA-RAD | VQA | Done |
| FairVLMed | VQA | Done |
| PneumoniaMNIST | Diagnosis | Done |
| BreastMNIST | Diagnosis | Done |
| DermaMNIST | Diagnosis | Done |
| Camelyon17 | Diagnosis | Done |
| HAM10000 | Diagnosis | Done |
| Drishti | Diagnosis | Done |
| ChestXray | Diagnosis | Done |
| GF3300 | Diagnosis | Done |
| CheXpert | Diagnosis | Done |
| PAPILA | Diagnosis | Done |
| FairVLMed | Diagnosis | Done |

</details>

<details>
<summary><b>Supported Models</b></summary>

| Model | Type | Evaluation | Training |
|---|---|---|---|
| o3 | VQA | Done | NA |
| Gemini 2.5 Pro | VQA | Done | NA |
| InternVL3 | VQA | Done | Coming Soon |
| LLaVA-1.5 | VQA | Done | Done |
| LLaVA-Med | VQA | Done | Done |
| Gemma3 | VQA | Done | Coming Soon |
| MedGemma | VQA | Done | Done |
| Qwen2-VL | VQA | Done | Coming Soon |
| Qwen25-VL | VQA | Done | Coming Soon |
| NVILA | VQA | Done | Done |
| VILA-M3 | VQA | Done | Done |
| VILA1.5 | VQA | Done | Done |
| Lingshu | VQA | Done | Done |
| BLIP | Diagnosis/VQA | Done | Done |
| BLIP2 | Diagnosis/VQA | Done | Done |
| XrayGPTVQA | Diagnosis/VQA | Done | Done |
| BioMedCLIP | Diagnosis | Done | Done |
| CLIP | Diagnosis | Done | Done |
| MedCLIP | Diagnosis | Done | Done |
| PMCCLIP | Diagnosis | Done | Done |
| PLIP | Diagnosis | Done | Done |
| MedSigLIP | Diagnosis | Done | Done |
| PubMedCLIP | Diagnosis | Done | Done |
| SigLIP | Diagnosis | Done | Done |

</details>

## Usage

`run_train.py` is the major entry for training all models (including the lightweight adaptation).
`run_eval.py` is the major entry for off the shelf evaluation of all models.

### Notebook Tutorials

We offer some examples of how to use our package through the notebook.

| Feature | Notebook |
|---|---|
| Off-the-shelf Diagnosis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/Nanboy-Ronan/MedVLMBench/blob/main/examples/MedVLMBench_OTS_Diagnosis.ipynb) |
| Off-the-shelf VQA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/Nanboy-Ronan/MedVLMBench/blob/main/examples/MedVLMBench_OTS_VQA.ipynb) |
| LP Diagnosis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/Nanboy-Ronan/MedVLMBench/blob/main/examples/MedVLMBench_LP_Diagnosis.ipynb) |
| LoRA Adaptation VQA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/Nanboy-Ronan/MedVLMBench/blob/main/examples/MedVLMBench_LoRA_VQA.ipynb) |

### Command-Line Interface

#### Off-the-shelf Evaluation

<details>
<summary><b>Diagnosis Example</b></summary>

```bash
python run_eval.py \
--task diagnosis --usage clip-zs --dataset PAPILA --split test \
--image_path ./data \
--exp_path ./log \
--model CLIP --model_path "original_pretrained" \
--save_pred \
--cache_dir ./cache
```

</details>

<details>
<summary><b>VQA Example</b></summary>

```bash
python run_eval.py \
--task vqa --dataset SLAKE --split test \
--image_path ./data/SLAKE/imgs \
--model LLaVA-1.5 --model_path ./pretrained_models/llava-v1.5-7b \
--exp_path ./log \
--cache_dir ./cache \
--save_pred
```

</details>

#### Training

<details>
<summary><b>Diagnosis Example</b></summary>

In our code, we have implemented more fine-tune method than the things reported in the paper. Specifically, you can do linear probing (`lp`), linear probing with the image encoder (`img-lora-lp`), and CLIP with lora finetune on image encoder (`clip-img-lora`).

```bash
python run_train.py \
--task diagnosis --usage lp --dataset HAM10000 --split train \
--image_path ./data \
--output_dir ./log \
--model CLIP --model_path not_given \
--cache_dir ./cache \
--num_train_epochs 50 \
--learning_rate 5e-5
```

</details>

<details>
<summary><b>VQA Example</b></summary>

```bash
deepspeed run_train.py \
--peft lora --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
--deepspeed ./script/zero3.json \
--task vqa --dataset SLAKE \
--model LLaVA-1.5 --version v1 \
--image_path ./data/SLAKE/imgs \
--model_path ./pretrained_models/llava-v1.5-7b \
--mm_projector_type mlp2x_gelu \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--bf16 True \
--output_dir ./log \
--cache_dir ./cache \
--num_train_epochs 1 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 2 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 50000 \
--save_total_limit 1 \
--learning_rate 2e-4 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 2048 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--tune_modules L
```

</details>

## Citation

If you find this repository useful, please consider citing our paper:

```
@article{zhong2025can,
  title={Can Common VLMs Rival Medical VLMs? Evaluation and Strategic Insights},
  author={Zhong, Yuan and Jin, Ruinan and Li, Xiaoxiao and Dou, Qi},
  journal={arXiv preprint arXiv:2506.17337},
  year={2025}
}
```
