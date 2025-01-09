# Zeroshot
CUDA_VISIBLE_DEVICES=2 python run_train.py \
    --task diagnosis --usage clip-zs --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model CLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 10 \
    --learning_rate 5e-5