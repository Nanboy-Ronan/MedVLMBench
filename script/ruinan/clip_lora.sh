# Training
# PneumoniaMNIST
python run_train.py \
    --task diagnosis --usage clip-img-lora --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model CLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 50 \
    --learning_rate 5e-5

python run_train.py \
    --task diagnosis --usage clip-img-lora --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model BLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 2 \
    --learning_rate 5e-5


python run_train.py \
    --task diagnosis --usage clip-img-lora --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model BLIP2-2.7b --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 2 \
    --learning_rate 5e-5


python run_train.py \
    --task diagnosis --usage clip-img-lora --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model BioMedCLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 2 \
    --learning_rate 5e-5


python run_train.py \
    --task diagnosis --usage clip-img-lora --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --output_dir ./demo \
    --model MedCLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 2 \
    --learning_rate 5e-5


python run_train.py \
    --task diagnosis --usage clip-img-lora --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --output_dir ./demo \
    --model PMCCLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 2 \
    --learning_rate 5e-5