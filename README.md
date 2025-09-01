# MedVLMBench

## Abstract
**Background:** Vision–Language Models (VLMs) have shown promise in automating image diagnosis and interpretation in clinical settings. However, developing medical-specialist VLMs requires substantial computational resources and carefully curated datasets, and it remains unclear under which conditions generalist and medical specialist VLMs each perform best.

**Methods:** This paper introduces MedVLMBench, the first unified benchmark for systematically evaluating generalist and medical-specialist VLMs. We assessed 18 models spanning contrastive and generative paradigms on 10 publicly available datasets across radiology, pathology, dermatology, and ophthalmology, encompassing 144 diagnostic and 80 VQA settings. MedVLMBench focusing on assessing both in-domain (ID) and out-of-domain (OOD) performance, with off-the-shelf and parameter-efficient fine-tuning (e.g., linear probing, LoRA). Diagnostic classification tasks were evaluated using AUROC, while visual question answering (VQA) tasks were assessed with BLEU-1, ROUGE-L, Exact Match, F1 Score, and GPT-based semantic scoring, covering both open- and closed-ended formats. Computational efficiency was estimated relative to the cost of full medical pretraining.

**Results:** As expected, off-the-shelf medical VLMs generally outperformed generalist VLMs on ID tasks given their pretraining. However, with lightweight fine-tuning, general-purpose VLMs achieved superior performance in most of ID task evaluations and demonstrated better generalization on OOD tasks in approximately all comparisons. Fine-tuning required only 3\% of the total parameters associated with full medical pretraining. In contrast, fine-tuned medical VLMs showed degraded performance even on ID tasks when subjected to rigorous hyperparameter optimization, further highlighting their limited adaptability.

**Conclusions:** This study highlights the complementary strengths of medical-specialist and generalist VLMs. Specialists remain valuable in modality-aligned use cases, but we find that efficiently fine-tuned generalist VLMs can achieve comparable or even superior performance in most tasks, particularly when transferring to unseen or rare OOD medical modalities. These results suggest that generalist VLMs, rather than being constrained by their lack of medical-specific pretraining, may offer a scalable and cost-effective pathway for advancing clinical AI development.


## Pretrained VLMs
All pretrained models will be stored under `MedVLMBench/pretrained_models`.

```bash
mkdir pretrained_models
```

LLaVa
```bash
cd pretrained_models
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
cd ..
```

## Dataset
All data will be stored under `MedVLMBench/data`.

```bash
mkdir data
```

| Dataset                       | Type      | Status |
|--------------------------------|-----------|--------|
| SLAKE                      | VQA       | Done   |
| PathVQA                    | VQA       | Done   |
| VQA-RAD                    | VQA       | Done   |
| Harvard-FairVLMed10k       | VQA       | Done   |
| MIMIC_CXR              | Caption   | Done   |
| PneumoniaMNIST       | Diagnosis | Done   |
| BreastMNIST          | Diagnosis | Done   |
| DermaMNIST           | Diagnosis | Done   |
| Camelyon17           | Diagnosis | Done   |
| HAM10000             | Diagnosis | Done   |
| Drishti                        | Diagnosis | Done   |
| ChestXray            | Diagnosis | Done   |
| GF3300               | Diagnosis | Done   |
| HarvardFairVLMed10k-caption    | Caption   | Done   |
| CheXpert             | Diagnosis | Done   |
| PAPILA               | Diagnosis | Done   |
| HarvardFairVLMed10k  | Diagnosis | Done   |


## Model
| Model        | Type      | Status |
|--------------|-----------|--------|
| BLIP         | VQA/Caption/Diagnosis | Done |
| LLaVA-1.5    | VQA/Caption           | Done |
| BLIP2-2.7b   | VQA/Caption/Diagnosis | Done |
| LLaVA-Med    | VQA/Caption           | Done |
| MedGemma     | VQA                   | Done |
| Qwen2-VL     | VQA                   | Done |
| Qwen25-VL    | VQA                   | Done |
| XGenMiniV1   | VQA/Caption           | Done |
| XrayGPT      | VQA/Caption/Diagnosis | Done |
| NVILA        | VQA                   | Done |
| VILA-M3      | VQA                   | Done |
| VILA1.5      | VQA                   | Done |
| Lingshu      | VQA                   | Done |
| BioMedCLIP   | Diagnosis             | Done |
| CLIP         | Diagnosis             | Done |
| MedCLIP      | Diagnosis             | Done |
| PMCCLIP      | Diagnosis             | Done |
| PLIP         | Diagnosis             | Done |
| MedSigLIP    | Diagnosis             | Done |
| PubMedCLIP   | Diagnosis             | Done |
| SigLIP       | Diagnosis             | Done |


## Notebook Tutorial

We offer some examples of how to use our package through the notebook.

| Feature                  | Notebook                                                                 |
|---------------------------|--------------------------------------------------------------------------|
| Off-the-shelf Diagnosis          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) |
| Off-the-shelf VQA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) |
| Off-the-shelf Captioning              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) |
| LP Diagnosis              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) |
| LoRA Adaptation VQA              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) |


## Implementation Logic
`run_train.py` is the major entry for training all models (including the lightweight adaptation).
`run_eval.py` is the major entry for off the shelf evaluation of all models.

### Off-the-shelf Evaluation

#### Diagnosis

Example
```bash
python run_eval.py \
    --task diagnosis --usage clip-zs --dataset PAPILA --split test \
    --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
    --exp_path ./log \
    --model CLIP --model_path "original_pretrained" \
    --save_pred \
    --cache_dir ./cache
```


#### VQA

Example
```bash
python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path ./data/SLAKE/imgs \
    --model LLaVA-1.5 --model_path ./pretrained_models/llava-v1.5-7b \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred
```

### Training



#### Diagnosis

#### VQA