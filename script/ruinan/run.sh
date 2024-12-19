#!/bin/bash

# SLAKE, LLaVA-1.5
python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path ./data/SLAKE/imgs \
    --model LLaVA-1.5 --model_path ./pretrained_models/llava-v1.5-7b \
    --exp_path ./med_vlm_benchmark \
    --cache_dir ./cache \
    --save_pred

# PathVQA, LLaVA-1.5
python run_eval.py \
    --task vqa --dataset PathVQA --split test \
    --image_path ./data/SLAKE/imgs \
    --model LLaVA-1.5 --model_path ./pretrained_models/llava-v1.5-7b \
    --exp_path ./med_vlm_benchmark \
    --cache_dir ./cache \
    --save_pred

# SLAKE, BLIP
python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path ./data/SLAKE/imgs \
    --model BLIP --model_path not_given \
    --exp_path ./med_vlm_benchmark \
    --cache_dir ./cache \
    --save_pred

# SLAKE, BLIP2-2.7b
python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path ./data/SLAKE/imgs \
    --model BLIP2-2.7b --model_path not_given \
    --exp_path ./med_vlm_benchmark \
    --cache_dir ./cache \
    --save_pred

# SLAKE, XGenMiniV1
python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path ./data/SLAKE/imgs \
    --model XGenMiniV1 --model_path not_given \
    --exp_path ./med_vlm_benchmark \
    --cache_dir ./cache \
    --save_pred 