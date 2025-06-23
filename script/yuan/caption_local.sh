# FairVLMed10k, LLaVA-1.5
python run_eval.py \
    --task caption --dataset Harvard-FairVLMed10k --split test \
    --image_path /mnt/hdd/data/Harvard-FairVLMed10k \
    --model LLaVA-1.5 --model_path /mnt/hdd/weights/llava-v1.5-7b \
    --exp_path /mnt/hdd/experiments/med_vlm_benchmark \
    --save_pred

# FairVLMed10k, LLaVA-Med
python run_eval.py \
    --task caption --dataset Harvard-FairVLMed10k --split test \
    --image_path /mnt/hdd/data/Harvard-FairVLMed10k \
    --model LLaVA-Med --model_path /mnt/hdd/weights/llava-med-v1.5-mistral-7b \
    --exp_path /mnt/hdd/experiments/med_vlm_benchmark \
    --save_pred