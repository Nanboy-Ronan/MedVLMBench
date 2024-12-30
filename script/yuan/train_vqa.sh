# SLAKE, LLaVA-1.5
deepspeed run_train.py \
    --peft lora --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./script/zero3.json \
    --task vqa --dataset SLAKE \
    --model LLaVA-1.5 --version v1 \
    --image_path /research/d5/gds/yzhong22/datasets/SLAKE/imgs \
    --model_path /research/d5/gds/yzhong22/misc/pretrained/llava-v1.5-7b \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /research/d5/gds/yzhong22/experiments/med_vlm_benchmark \
    --cache_dir /research/d5/gds/yzhong22/misc/cache \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --tune_modules L

deepspeed run_train.py \
    --peft lora --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./script/zero3.json \
    --task vqa --dataset SLAKE \
    --model LLaVA-Med --version mistral_instruct \
    --image_path /research/d5/gds/yzhong22/datasets/SLAKE/imgs \
    --model_path /research/d5/gds/yzhong22/misc/pretrained/llava-med-v1.5-mistral-7b \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /research/d5/gds/yzhong22/experiments/med_vlm_benchmark \
    --cache_dir /research/d5/gds/yzhong22/misc/cache \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --tune_modules L


python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path /research/d5/gds/yzhong22/datasets/SLAKE/imgs \
    --model LLaVA-1.5 --model_path /research/d5/gds/yzhong22/experiments/med_vlm_benchmark/vqa/SLAKE/LLaVA-1.5/train__M_seed42_llava \
    --model_base /research/d5/gds/yzhong22/misc/pretrained/llava-v1.5-7b \
    --exp_path /research/d5/gds/yzhong22/experiments/med_vlm_benchmark \
    --cache_dir /research/d5/gds/yzhong22/misc/cache \
    --save_pred

python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path /research/d5/gds/yzhong22/datasets/SLAKE/imgs \
    --model LLaVA-Med --model_path /research/d5/gds/yzhong22/experiments/med_vlm_benchmark/vqa/SLAKE/LLaVA-Med/train_lora_L_seed42_llava_mistral \
    --model_base /research/d5/gds/yzhong22/misc/pretrained/llava-med-v1.5-mistral-7b \
    --exp_path /research/d5/gds/yzhong22/experiments/med_vlm_benchmark \
    --save_pred