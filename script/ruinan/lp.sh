# Evaluation only
python run_eval.py \
    --task diagnosis --usage clip-zs --dataset PneumoniaMNIST --split test \
    --image_path ./data \
    --model BLIP --model_path not_given \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

# Train
python run_train.py \
    --task diagnosis --usage lp --dataset PneumoniaMNIST --split train \
    --image_path ./data \
    --output_dir ./log \
    --model BLIP --model_path not_given \
    --cache_dir ./cache