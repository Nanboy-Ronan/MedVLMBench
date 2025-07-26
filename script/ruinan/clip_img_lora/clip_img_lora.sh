# HAM10000
CUDA_VISIBLE_DEVICES=4 python run_train.py \
    --task diagnosis --usage clip-img-lora --dataset Camelyon17 --split train \
    --image_path ./data/camelyon17_v1.0/patches \
    --output_dir ./log \
    --model CLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 10 \
    --learning_rate 5e-5