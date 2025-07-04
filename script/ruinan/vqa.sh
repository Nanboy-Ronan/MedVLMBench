#!/bin/bash

# SLAKE, LLaVA-1.5
python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path ./data/SLAKE/imgs \
    --model LLaVA-1.5 --model_path ./pretrained_models/llava-v1.5-7b \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# PathVQA, LLaVA-1.5
python run_eval.py \
    --task vqa --dataset PathVQA --split test \
    --image_path ./data/SLAKE/imgs \
    --model LLaVA-1.5 --model_path ./pretrained_models/llava-v1.5-7b \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# VQARAD, LLaVA-1.5
python run_eval.py \
    --task vqa --dataset VQA-RAD --split test \
    --image_path ./data/SLAKE/imgs \
    --model LLaVA-1.5 --model_path ./pretrained_models/llava-v1.5-7b \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# SLAKE, NVILA-8B 
CUDA_VISIBLE_DEVICES=4 python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path ./data/SLAKE/imgs \
    --model NVILA --model_path Efficient-Large-Model/NVILA-8B \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# PathVQA, NVILA-8B
CUDA_VISIBLE_DEVICES=5 python run_eval.py \
    --task vqa --dataset PathVQA --split test \
    --image_path ./data/VQARAD/imgs \
    --model NVILA --model_path Efficient-Large-Model/NVILA-8B \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# VQARAD, NVILA-8B
CUDA_VISIBLE_DEVICES=6 python run_eval.py \
    --task vqa --dataset VQA-RAD --split test \
    --image_path ./data/VQARAD/imgs \
    --model NVILA --model_path Efficient-Large-Model/NVILA-8B \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# SLAKE, VILA-M3 
CUDA_VISIBLE_DEVICES=4 python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path ./data/SLAKE/imgs \
    --model VILA-M3 --model_path MONAI/Llama3-VILA-M3-8B \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# PathVQA, VILA-M3
CUDA_VISIBLE_DEVICES=5 python run_eval.py \
    --task vqa --dataset PathVQA --split test \
    --image_path ./data/SLAKE/imgs \
    --model VILA-M3 --model_path MONAI/Llama3-VILA-M3-8B \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# VQARAD, VILA-M3
CUDA_VISIBLE_DEVICES=6 python run_eval.py \
    --task vqa --dataset VQA-RAD --split test \
    --image_path ./data/SLAKE/imgs \
    --model VILA-M3 --model_path MONAI/Llama3-VILA-M3-8B \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# SLAKE, VILA1.5-8B
CUDA_VISIBLE_DEVICES=4 python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path ./data/SLAKE/imgs \
    --model VILA1.5 --model_path Efficient-Large-Model/Llama-3-VILA1.5-8B \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# PathVQA, VILA1.5-8B
CUDA_VISIBLE_DEVICES=5 python run_eval.py \
    --task vqa --dataset PathVQA --split test \
    --image_path ./data/SLAKE/imgs \
    --model VILA1.5 --model_path Efficient-Large-Model/Llama-3-VILA1.5-8B \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# VQARAD, VILA1.5-8B
CUDA_VISIBLE_DEVICES=6 python run_eval.py \
    --task vqa --dataset VQA-RAD --split test \
    --image_path ./data/SLAKE/imgs \
    --model VILA1.5 --model_path Efficient-Large-Model/Llama-3-VILA1.5-8B \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred


# SLAKE, BLIP
python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path ./data/SLAKE/imgs \
    --model BLIP --model_path not_given \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# PathVQA, BLIP
python run_eval.py \
    --task vqa --dataset PathVQA --split test \
    --image_path ./data/SLAKE/imgs \
    --model BLIP --model_path not_given \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# VQARAD, BLIP
python run_eval.py \
    --task vqa --dataset VQARAD --split test \
    --image_path ./data/SLAKE/imgs \
    --model BLIP --model_path not_given \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# SLAKE, BLIP2-2.7b
python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path ./data/SLAKE/imgs \
    --model BLIP2-2.7b --model_path not_given \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# PathVQA, BLIP2-2.7b
CUDA_VISIBLE_DEVICES=5 python run_eval.py \
    --task vqa --dataset PathVQA --split test \
    --image_path ./data/SLAKE/imgs \
    --model BLIP2-2.7b --model_path not_given \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred


# VQARAD, BLIP2-2.7b
CUDA_VISIBLE_DEVICES=3 python run_eval.py \
    --task vqa --dataset VQARAD --split test \
    --image_path ./data/SLAKE/imgs \
    --model BLIP2-2.7b --model_path not_given \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# SLAKE, XGenMiniV1
python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path ./data/SLAKE/imgs \
    --model XGenMiniV1 --model_path not_given \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# PathVQA, XGenMiniV1
python run_eval.py \
    --task vqa --dataset PathVQA --split test \
    --image_path ./data/SLAKE/imgs \
    --model XGenMiniV1 --model_path not_given \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# VQARAD, XGenMiniV1
python run_eval.py \
    --task vqa --dataset VQARAD --split test \
    --image_path ./data/SLAKE/imgs \
    --model XGenMiniV1 --model_path not_given \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# SLAKE, XrayGPT (done)
python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path ./data/SLAKE/imgs \
    --model XrayGPT --model_path not_given \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred 

# PathVQA, XrayGPT (done)
python run_eval.py \
    --task vqa --dataset PathVQA --split test \
    --image_path ./data/SLAKE/imgs \
    --model XrayGPT --model_path not_given \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# VQARAD, XrayGPT (done)
python run_eval.py \
    --task vqa --dataset VQARAD --split test \
    --image_path ./data/SLAKE/imgs \
    --model XrayGPT --model_path not_given \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred 