# Train Xray
# TODO
CUDA_VISIBLE_DEVICES=1 python run_train.py \
    --task diagnosis --usage lora_lp --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model BLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 10 \
    --learning_rate 5e-5


CUDA_VISIBLE_DEVICES=1 python run_train.py \
    --task diagnosis --usage lora_lp --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model XrayGPT --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 10 \
    --learning_rate 5e-5


CUDA_VISIBLE_DEVICES=1 python run_train.py \
    --task diagnosis --usage lora_lp --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model BioMedCLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 10 \
    --learning_rate 5e-5


CUDA_VISIBLE_DEVICES=1 python run_train.py \
    --task diagnosis --usage lora_lp --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model CLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 10 \
    --learning_rate 5e-5