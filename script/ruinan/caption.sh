export WANDB_DISABLED=true
export CUDA_LAUNCH_BLOCKING=1

################################## Off-the-shelf Usage ##################################

# LLaVa, Harvard-FairVLMed10k
python run_eval.py \
    --task caption --dataset Harvard-FairVLMed10k --split test \
    --image_path /data/rjin02/project/FairMedFM-DNE/data/FairVLMed10k \
    --model LLaVA-1.5 --model_path ./pretrained_models/llava-v1.5-7b \
    --exp_path ./log \
    --save_pred

# VILA1.5, Harvard-FairVLMed10k
CUDA_VISIBLE_DEVICES=0 python run_eval.py \
    --task caption --dataset Harvard-FairVLMed10k --split test \
    --image_path /data/rjin02/project/FairMedFM-DNE/data/FairVLMed10k \
    --model VILA1.5 --model_path Efficient-Large-Model/Llama-3-VILA1.5-8B \
    --exp_path ./log \
    --save_pred


# VILA-M3, Harvard-FairVLMed10k
CUDA_VISIBLE_DEVICES=1 python run_eval.py \
    --task caption --dataset Harvard-FairVLMed10k --split test \
    --image_path /data/rjin02/project/FairMedFM-DNE/data/FairVLMed10k \
    --model VILA-M3 --model_path MONAI/Llama3-VILA-M3-8B \
    --exp_path ./log \
    --save_pred

# VILA1.5, MIMIC-CXR
python run_eval.py \
    --task caption --dataset MIMIC_CXR --split test \
    --image_path /data/rjin02/project/DataSets/mimic_cxr \
    --model VILA1.5 --model_path Efficient-Large-Model/Llama-3-VILA1.5-8B \
    --exp_path ./log \
    --save_pred

# VILA-M3, MIMIC-CXR
CUDA_VISIBLE_DEVICES=1 python run_eval.py \
    --task caption --dataset MIMIC_CXR --split test \
    --image_path /data/rjin02/project/DataSets/mimic_cxr \
    --model VILA-M3 --model_path MONAI/Llama3-VILA-M3-8B \
    --exp_path ./log \
    --save_pred


################################## SFT Usage ##################################
# VILA1.5, Harvard-FairVLMed10k
deepspeed --include localhost:4 --master_port 29604 run_train.py \
    --peft lora --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./script/zero2.json \
    --task caption --dataset Harvard-FairVLMed10k \
    --model VILA1.5 \
    --image_path /data/rjin02/project/FairMedFM-DNE/data/FairVLMed10k \
    --model_path Efficient-Large-Model/Llama-3-VILA1.5-8B \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./log \
    --cache_dir ./cache \
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
    --gradient_checkpointing False \
    --dataloader_num_workers 0 \
    --tune_modules ML

# VILA1.5, Harvard-FairVLMed10k
CUDA_VISIBLE_DEVICES=0 python run_eval.py \
    --task caption --dataset Harvard-FairVLMed10k --split test \
    --image_path /data/rjin02/project/FairMedFM-DNE/data/FairVLMed10k \
    --model VILA1.5 --model_path /bigdata/rjin02/MedVLMBench/log/caption/Harvard-FairVLMed10k/VILA1.5/train_lora_ML_seed42_vila \
    --exp_path ./log \
    --save_pred


# VILA-M3, Harvard-FairVLMed10k
deepspeed --include localhost:5 --master_port 29606 run_train.py \
    --peft lora --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./script/zero2.json \
    --task caption --dataset Harvard-FairVLMed10k \
    --model VILA-M3 \
    --image_path /data/rjin02/project/FairMedFM-DNE/data/FairVLMed10k \
    --model_path MONAI/Llama3-VILA-M3-8B \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./log \
    --cache_dir ./cache \
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
    --gradient_checkpointing False \
    --dataloader_num_workers 0 \
    --tune_modules ML


# VILA-M3, Harvard-FairVLMed10k
CUDA_VISIBLE_DEVICES=1 python run_eval.py \
    --task caption --dataset Harvard-FairVLMed10k --split test \
    --image_path /data/rjin02/project/FairMedFM-DNE/data/FairVLMed10k \
    --model VILA-M3 --model_path /bigdata/rjin02/MedVLMBench/log/caption/Harvard-FairVLMed10k/VILA-M3/train_lora_ML_seed42_vila_m3 \
    --exp_path ./log \
    --save_pred


# VILA1.5, MIMIC-CXR                                  
deepspeed --include localhost:4 --master_port 29604 run_train.py \
    --peft lora --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./script/zero2.json \
    --task caption --dataset MIMIC_CXR \
    --model VILA1.5 \
    --image_path /data/rjin02/project/DataSets/mimic_cxr \
    --model_path Efficient-Large-Model/Llama-3-VILA1.5-8B \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./log \
    --cache_dir ./cache \
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
    --gradient_checkpointing False \
    --dataloader_num_workers 0 \
    --tune_modules ML

# VILA1.5, MIMIC-CXR
python run_eval.py \
    --task caption --dataset MIMIC_CXR --split test \
    --image_path /data/rjin02/project/DataSets/mimic_cxr \
    --model VILA1.5 --model_path /bigdata/rjin02/MedVLMBench/log/caption/MIMIC_CXR/VILA1.5/train_lora_ML_seed42_vila \
    --exp_path ./log \
    --save_pred


# VILA-M3, MIMIC-CXR
deepspeed --include localhost:5 --master_port 29606 run_train.py \
    --peft lora --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./script/zero2.json \
    --task caption --dataset MIMIC_CXR \
    --model VILA-M3 \
    --image_path /data/rjin02/project/DataSets/mimic_cxr \
    --model_path MONAI/Llama3-VILA-M3-8B \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./log \
    --cache_dir ./cache \
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
    --gradient_checkpointing False \
    --dataloader_num_workers 0 \
    --tune_modules ML


# VILA-M3, MIMIC-CXR
CUDA_VISIBLE_DEVICES=1 python run_eval.py \
    --task caption --dataset MIMIC_CXR --split test \
    --image_path /data/rjin02/project/DataSets/mimic_cxr \
    --model VILA-M3 --model_path /bigdata/rjin02/MedVLMBench/log/caption/MIMIC_CXR/VILA-M3/train_lora_ML_seed42_vila_m3 \
    --exp_path ./log \
    --save_pred