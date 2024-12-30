# Evaluation only
python run_eval.py \
    --task diagnosis --usage lp --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --exp_path ./log \
    --model BLIP --model_path "/fast/rjin02/MedVLMBench/log/diagnosis/PneumoniaMNIST/BLIP/train_VML_seed42/checkpoint-74/pytorch_model.bin" \
    --cache_dir ./cache

# Train
python run_train.py \
    --task diagnosis --usage lp --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model BLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 50 \
    --learning_rate 5e-5