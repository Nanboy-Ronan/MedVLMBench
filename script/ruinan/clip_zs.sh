# Zeroshot
# PneumoniaMNIST
python run_eval.py \
    --task diagnosis --usage clip-zs --dataset PneumoniaMNIST --split test \
    --image_path ./data \
    --exp_path ./log \
    --model CLIP --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset PneumoniaMNIST --split test \
    --image_path ./data \
    --exp_path ./log \
    --model BLIP --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset PneumoniaMNIST --split test \
    --image_path ./data \
    --exp_path ./log \
    --model BioMedCLIP --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset PneumoniaMNIST --split test \
    --image_path ./data \
    --exp_path ./log \
    --model MedCLIP --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset PneumoniaMNIST --split test \
    --image_path ./data \
    --exp_path ./log \
    --model BLIP2-2.7b --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset PneumoniaMNIST --split test \
    --image_path ./data \
    --exp_path ./log \
    --model PMCCLIP --model_path "original_pretrained" \
    --cache_dir ./cache

# BreastMNIST
python run_eval.py \
    --task diagnosis --usage clip-zs --dataset BreastMNIST --split test \
    --image_path ./data \
    --exp_path ./log \
    --model CLIP --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset BreastMNIST --split test \
    --image_path ./data \
    --exp_path ./log \
    --model BLIP --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset BreastMNIST --split test \
    --image_path ./data \
    --exp_path ./log \
    --model BioMedCLIP --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset BreastMNIST --split test \
    --image_path ./data \
    --exp_path ./log \
    --model MedCLIP --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset BreastMNIST --split test \
    --image_path ./data \
    --exp_path ./log \
    --model BLIP2-2.7b --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset BreastMNIST --split test \
    --image_path ./data \
    --exp_path ./log \
    --model PMCCLIP --model_path "original_pretrained" \
    --cache_dir ./cache

# DermaMNIST
python run_eval.py \
    --task diagnosis --usage clip-zs --dataset DermaMNIST --split test \
    --image_path ./data \
    --exp_path ./log \
    --model CLIP --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset DermaMNIST --split test \
    --image_path ./data \
    --exp_path ./log \
    --model BLIP --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset DermaMNIST --split test \
    --image_path ./data \
    --exp_path ./log \
    --model BioMedCLIP --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset DermaMNIST --split test \
    --image_path ./data \
    --exp_path ./log \
    --model MedCLIP --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset DermaMNIST --split test \
    --image_path ./data \
    --exp_path ./log \
    --model PMCCLIP --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset DermaMNIST --split test \
    --image_path ./data \
    --exp_path ./log \
    --model BLIP2-2.7b --model_path "original_pretrained" \
    --cache_dir ./cache

# Training
CUDA_VISIBLE_DEVICES=5 python run_train.py \
    --task diagnosis --usage clip-img-lora --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model BLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 50 \
    --learning_rate 5e-5