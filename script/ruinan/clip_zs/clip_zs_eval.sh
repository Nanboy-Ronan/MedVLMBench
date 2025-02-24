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


# Camelyon17
python run_eval.py \
    --task diagnosis --usage clip-zs --dataset Camelyon17 --split test \
    --image_path ./data/camelyon17_v1.0/patches \
    --exp_path ./log \
    --model CLIP --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset Camelyon17 --split test \
    --image_path ./data \
    --exp_path ./log \
    --model BLIP --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset Camelyon17 --split test \
    --image_path ./data \
    --exp_path ./log \
    --model BioMedCLIP --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset Camelyon17 --split test \
    --image_path ./data \
    --exp_path ./log \
    --model MedCLIP --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset Camelyon17 --split test \
    --image_path ./data \
    --exp_path ./log \
    --model BLIP2-2.7b --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset Camelyon17 --split test \
    --image_path ./data \
    --exp_path ./log \
    --model PMCCLIP --model_path "original_pretrained" \
    --cache_dir ./cache

# DermaMNIST
# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset DermaMNIST --split test \
#     --image_path ./data \
#     --exp_path ./log \
#     --model CLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset DermaMNIST --split test \
#     --image_path ./data \
#     --exp_path ./log \
#     --model BLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset DermaMNIST --split test \
#     --image_path ./data \
#     --exp_path ./log \
#     --model BioMedCLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset DermaMNIST --split test \
#     --image_path ./data \
#     --exp_path ./log \
#     --model MedCLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset DermaMNIST --split test \
#     --image_path ./data \
#     --exp_path ./log \
#     --model PMCCLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset DermaMNIST --split test \
#     --image_path ./data \
#     --exp_path ./log \
#     --model BLIP2-2.7b --model_path "original_pretrained" \
#     --cache_dir ./cache
    