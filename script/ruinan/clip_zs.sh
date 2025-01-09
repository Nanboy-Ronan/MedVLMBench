# Zeroshot
python run_eval.py \
    --task diagnosis --usage clip-zs --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --exp_path ./log \
    --model CLIP --model_path "original_pretrained" \
    --cache_dir ./cache