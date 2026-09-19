# Quilt-LLaVA and Patho-R1 in MedVLMBench

Both are generative VQA models. Quilt-LLaVA-v1.5-7B uses the LLaVA-1.5-7B architecture (Vicuna 7B, CLIP ViT-L/14-336, two-layer MLP projector). Patho-R1-7B uses Qwen2.5-VL-7B. Use `--task vqa`, with `--model Quilt-LLaVA` or `--model Patho-R1`; `--usage mdagent` and `--usage ucagent` use the same model wrappers.

The scripts in `train_infer_bash/quilt_llava/` and `train_infer_bash/patho_r1/` accept `MODE=eval|train|mdagent|ucagent`, `DATASET`, `IMAGE_PATH`, `MODEL_PATH`, `EXP_PATH`, `SPLIT`, and `SEED`. Example:

```bash
IMAGE_PATH=/data/PathVQA MODE=eval bash script/yuan/train_infer_bash/quilt_llava/quilt_llava_vqa.sh
IMAGE_PATH=/data/PathVQA MODE=mdagent bash script/yuan/train_infer_bash/patho_r1/patho_r1_vqa.sh
```

The batch command notebooks `batch_run/quilt-llava.ipynb` and `batch_run/patho-r1.ipynb` follow the existing `batch_run` convention: they print commands for official-checkpoint evaluation, MDAgents, UCAgents, and the same three uses with discovered LoRA adapters. Quilt-LLaVA also prints framework LoRA training commands. Patho-R1's separate custom SFT command cell is disabled by default and requires both an explicit notebook opt-in and a permission environment variable. Edit the dataset/checkpoint paths in the notebook before copying commands to a shell at the repository root. These notebooks do not execute model workloads themselves.

After framework LoRA training, evaluate an adapter checkpoint with `MODEL_PATH=/path/to/adapter` and `MODEL_BASE=/path/to/original/checkpoint`. Training outputs are separate from off-the-shelf results.

Patho-R1 model weights are gated at `WenchuanZhang/Patho-R1-7B`: each researcher must obtain access with an institutional account. Its [model terms](https://huggingface.co/WenchuanZhang/Patho-R1-7B) prohibit derivative training without prior written permission. The repository publishes inference examples but no scripts reproducing the complete original training. The paper reports ms-swift for continued pretraining, LLaMA-Factory for SFT, and verl for RL.

Our `MODE=train` entry point is a **separate custom Hugging Face Trainer/PEFT LoRA SFT**, not official training and not the existing Qwen/Lingshu LLaMA-Factory recipe. It is opt-in with `PATHO_R1_USE_CUSTOM_SFT=yes` and, only after written permission, `PATHO_R1_TRAINING_AUTHORIZED=yes`; the Python entry point checks `--patho_r1_training_authorized True` independently. The implementation supports LoRA on the vision encoder (`V`), visual merger (`M`), and decoder (`L`). Do not mix its adapted result into the matched-adaptation analysis without first aligning the LLaMA-Factory data, split, hyperparameters, and adapter/export procedure across models. The off-the-shelf Patho-R1 evaluation does not use this training entry point.

For exposure analyses, both official papers report PathVQA evaluation, but neither report PathVQA as a named training set. This does not establish image-level disjointness. Quilt-LLaVA uses Quilt-1M/Quilt-Instruct; Patho-R1 uses PubMed, Quilt, PathGen, textbooks, and pathology notes. Audit source/image overlap before treating PathVQA as an independent pathology test. Patho-R1's paper also evaluates MedXpertQA and OmniMedVQA; do not assume their pathology subsets are absent from its training sources.

Primary sources: [Quilt-LLaVA repository](https://github.com/aldraus/quilt-llava), [Quilt checkpoint/config](https://huggingface.co/wisdomik/Quilt-Llava-v1.5-7b), [Patho-R1 repository](https://github.com/Wenchuan-Zhang/Patho-R1), [Patho-R1 paper](https://arxiv.org/abs/2505.11404).
