python run_eval.py \
    --task diagnosis --dataset PneumoniaMNIST --split test \
    --image_path ./data \
    --model BLIP --model_path not_given \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred