python run_eval.py \
    --task caption --dataset Harvard-FairVLMed10k --split test \
    --image_path /fast/rjin02/DataSets/Harvard-FairVLMed10k \
    --model LLaVA-1.5 --model_path ./pretrained_models/llava-v1.5-7b \
    --exp_path ./log \
    --save_pred