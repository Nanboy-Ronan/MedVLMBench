# Zeroshot
python run_eval.py \
    --task diagnosis --usage clip-zs --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --exp_path ./log \
    --model CLIP --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --exp_path ./log \
    --model BLIP --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --exp_path ./log \
    --model BioMedCLIP --model_path "original_pretrained" \
    --cache_dir ./cache

python run_eval.py \
    --task diagnosis --usage clip-zs --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --exp_path ./log \
    --model MedCLIP --model_path "original_pretrained" \
    --cache_dir ./cache