# Evaluation only
python run_eval.py \
    --task diagnosis --usage lp --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --exp_path ./log \
    --model BLIP --model_path "/fast/rjin02/MedVLMBench/log/diagnosis/PneumoniaMNIST/BLIP/train_VML_seed42/checkpoint-11780/pytorch_model.bin" \
    --cache_dir ./cache

# Train
CUDA_VISIBLE_DEVICES=1 python run_train.py \
    --task diagnosis --usage lp --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model BLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 10 \
    --learning_rate 5e-5


CUDA_VISIBLE_DEVICES=2 python run_train.py \
    --task diagnosis --usage lp --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model XrayGPT --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 10 \
    --learning_rate 5e-5