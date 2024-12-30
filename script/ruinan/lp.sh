# Evaluation only
# Xray
# Done
python run_eval.py \
    --task diagnosis --usage lp --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --exp_path ./log \
    --model BLIP --model_path "/fast/rjin02/MedVLMBench/log/diagnosis/PneumoniaMNIST/BLIP/train_VML_seed42/checkpoint-11780/pytorch_model.bin" \
    --cache_dir ./cache


python run_eval.py \
    --task diagnosis --usage lp --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --exp_path ./log \
    --model XrayGPT --model_path "/fast/rjin02/MedVLMBench/log/diagnosis/PneumoniaMNIST/BLIP/train_VML_seed42/checkpoint-11780/pytorch_model.bin" \
    --cache_dir ./cache


python run_eval.py \
    --task diagnosis --usage lp --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --exp_path ./log \
    --model BioMedCLIP --model_path "/fast/rjin02/MedVLMBench/log/diagnosis/PneumoniaMNIST/BLIP/train_VML_seed42/checkpoint-11780/pytorch_model.bin" \
    --cache_dir ./cache

# BREAST
python run_eval.py \
    --task diagnosis --usage lp --dataset BreastMNIST --split train \
    --image_path ./data \
    --exp_path ./log \
    --model BLIP --model_path "/fast/rjin02/MedVLMBench/log/diagnosis/PneumoniaMNIST/BLIP/train_VML_seed42/checkpoint-11780/pytorch_model.bin" \
    --cache_dir ./cache


python run_eval.py \
    --task diagnosis --usage lp --dataset BreastMNIST --split train \
    --image_path ./data \
    --exp_path ./log \
    --model XrayGPT --model_path "/fast/rjin02/MedVLMBench/log/diagnosis/PneumoniaMNIST/BLIP/train_VML_seed42/checkpoint-11780/pytorch_model.bin" \
    --cache_dir ./cache


python run_eval.py \
    --task diagnosis --usage lp --dataset BreastMNIST --split train \
    --image_path ./data \
    --exp_path ./log \
    --model BioMedCLIP --model_path "/fast/rjin02/MedVLMBench/log/diagnosis/PneumoniaMNIST/BLIP/train_VML_seed42/checkpoint-11780/pytorch_model.bin" \
    --cache_dir ./cache

# DermaMNIST
python run_eval.py \
    --task diagnosis --usage lp --dataset DermaMNIST --split train \
    --image_path ./data \
    --exp_path ./log \
    --model BLIP --model_path "/fast/rjin02/MedVLMBench/log/diagnosis/PneumoniaMNIST/BLIP/train_VML_seed42/checkpoint-11780/pytorch_model.bin" \
    --cache_dir ./cache


python run_eval.py \
    --task diagnosis --usage lp --dataset DermaMNIST --split train \
    --image_path ./data \
    --exp_path ./log \
    --model XrayGPT --model_path "/fast/rjin02/MedVLMBench/log/diagnosis/PneumoniaMNIST/BLIP/train_VML_seed42/checkpoint-11780/pytorch_model.bin" \
    --cache_dir ./cache


python run_eval.py \
    --task diagnosis --usage lp --dataset DermaMNIST --split train \
    --image_path ./data \
    --exp_path ./log \
    --model BioMedCLIP --model_path "/fast/rjin02/MedVLMBench/log/diagnosis/PneumoniaMNIST/BLIP/train_VML_seed42/checkpoint-11780/pytorch_model.bin" \
    --cache_dir ./cache



# Train Xray
# Done
CUDA_VISIBLE_DEVICES=1 python run_train.py \
    --task diagnosis --usage lp --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model BLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 10 \
    --learning_rate 5e-5

# Ongoing
CUDA_VISIBLE_DEVICES=2 python run_train.py \
    --task diagnosis --usage lp --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model XrayGPT --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 10 \
    --learning_rate 5e-5

# Ongoing
CUDA_VISIBLE_DEVICES=3 python run_train.py \
    --task diagnosis --usage lp --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model BioMedCLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 10 \
    --learning_rate 5e-5

# Train BREAST
CUDA_VISIBLE_DEVICES=4 python run_train.py \
    --task diagnosis --usage lp --dataset BreastMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model BLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 10 \
    --learning_rate 5e-5

# Ongoing
CUDA_VISIBLE_DEVICES=5 python run_train.py \
    --task diagnosis --usage lp --dataset BreastMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model XrayGPT --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 10 \
    --learning_rate 5e-5

# Ongoing
CUDA_VISIBLE_DEVICES=6 python run_train.py \
    --task diagnosis --usage lp --dataset BreastMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model BioMedCLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 10 \
    --learning_rate 5e-5

# Train DermaMNIST
CUDA_VISIBLE_DEVICES=0 python run_train.py \
    --task diagnosis --usage lp --dataset DermaMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model BLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 10 \
    --learning_rate 5e-5

# Ongoing
CUDA_VISIBLE_DEVICES=1 python run_train.py \
    --task diagnosis --usage lp --dataset DermaMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model XrayGPT --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 10 \
    --learning_rate 5e-5

# Ongoing
CUDA_VISIBLE_DEVICES=7 python run_train.py \
    --task diagnosis --usage lp --dataset DermaMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model BioMedCLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 10 \
    --learning_rate 5e-5