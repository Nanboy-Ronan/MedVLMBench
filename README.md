# MedVLMBench

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

| Dataset   | Status   |
|------------|------------|
| [SLAKE](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/SLAKE.md) | Done |
| VQA-RAD | Done |
| MIMIC -CXR| Done |
| Peir Gross | TODO|
| PMC-OA | TODO |
| Medtrinity-25M | TODO |
| Harvard-FairVLMed | Done |
| Quilt | TODO |
| PneumoniaMNIST | Done |
| BrestMNIST | TODO |
| DermaMNIST | TODO |
| TCGA-COAD | TODO |
| TCGA-READ | TODO |
| MIMIC | TODO |

## Model
| Model   | Status   |
|------------|------------|
| BLIP | Done |
| BLIP2 | Done |
| XGen | Done |
| LLaVa | Done |
| LLaVaMed | Done |
| PMC-VQA | TODO |
| CTCLIP | No generation supported |
| Med-Flamingo | TODO |
| XrayGPT | TODO |
| RedFM | TODO |
| Visual Med Apaca | TODO |
| BioMedGPT | TODO |
| Med PaLM | TODO |

## Implementation Logic
### Training
#### Diagnosis
First finetune the linear layer using `run_train.py` then load the pretrained model for evaluation in `run_eval.py`