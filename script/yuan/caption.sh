# FairVLMed10k, LLaVA-Med
export HF_HOME=/research/d5/gds/yzhong22/misc/cache

python run_eval.py \
    --task caption --dataset Harvard-FairVLMed10k --split test \
    --image_path /research/d5/gds/yzhong22/datasets/Harvard-FairVLMed10k \
    --model LLaVA-Med --model_path /research/d5/gds/yzhong22/misc/pretrained/llava-med-v1.5-mistral-7b \
    --exp_path /research/d5/gds/yzhong22/experiments/med_vlm_benchmark \
    --save_pred