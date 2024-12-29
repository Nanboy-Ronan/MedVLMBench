#!/bin/bash

export HF_HOME=/research/d5/gds/yzhong22/misc/cache
# SLAKE, LLaVA-1.5
python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path /research/d5/gds/yzhong22/datasets/SLAKE/imgs \
    --model LLaVA-1.5 --model_path /research/d5/gds/yzhong22/misc/pretrained/llava-v1.5-7b \
    --exp_path /research/d5/gds/yzhong22/experiments/med_vlm_benchmark \
    --cache_dir /research/d5/gds/yzhong22/misc/cache \
    --save_pred


python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path /research/d5/gds/yzhong22/datasets/SLAKE/imgs \
    --model LLaVA-1.5 --model_path /research/d5/gds/yzhong22/experiments/med_vlm_benchmark/vqa/SLAKE/LLaVA-1.5/train_lora_VML_seed42_llava \
    --model_base /research/d5/gds/yzhong22/misc/pretrained/llava-v1.5-7b \
    --exp_path /research/d5/gds/yzhong22/experiments/med_vlm_benchmark \
    --cache_dir /research/d5/gds/yzhong22/misc/cache \
    --save_pred

# SLAKE, BLIP
python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path /research/d5/gds/yzhong22/datasets/SLAKE/imgs \
    --model BLIP \
    --exp_path /research/d5/gds/yzhong22/experiments/med_vlm_benchmark \
    --cache_dir /research/d5/gds/yzhong22/misc/cache \
    --save_pred