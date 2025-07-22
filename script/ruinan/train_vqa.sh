export WANDB_DISABLED=true
export CUDA_LAUNCH_BLOCKING=1

# SLAKE, NVILA
deepspeed --include localhost:4 run_train.py \
    --peft lora --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./script/zero2.json \
    --task vqa --dataset SLAKE \
    --model NVILA --version v1 \
    --image_path ./data/SLAKE/imgs \
    --model_path Efficient-Large-Model/NVILA-8B \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./log \
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


python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path ./data/SLAKE/imgs \
    --model NVILA --model_path /bigdata/rjin02/MedVLMBench/log/vqa/SLAKE/NVILA/train_lora_L_seed42_nvila \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

################################## VILA1.5-8B ##################################
# SLAKE, VILA1.5-8B
deepspeed --include localhost:0 --master_port 29601 run_train.py \
    --peft lora --lora_r 8 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./script/zero2.json \
    --task vqa --dataset SLAKE \
    --model VILA1.5 \
    --image_path ./data/SLAKE/imgs \
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
    --dataloader_num_workers 4 \
    --tune_modules ML

CUDA_VISIBLE_DEVICE=4 python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path ./data/SLAKE/imgs \
    --model VILA1.5 --model_path /bigdata/rjin02/MedVLMBench/log/vqa/SLAKE/VILA1.5/train_lora_ML_seed42_vila \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred


# PathVQA, VILA1.5-8B
deepspeed --include localhost:1 --master_port 29602 run_train.py \
    --peft lora --lora_r 8 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./script/zero2.json \
    --task vqa --dataset PathVQA \
    --model VILA1.5 \
    --image_path ./notnedded \
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
    --dataloader_num_workers 4 \
    --tune_modules ML

CUDA_VISIBLE_DEVICES=5 python run_eval.py \
    --task vqa --dataset PathVQA --split test \
    --image_path /notgiven \
    --model VILA1.5 --model_path /bigdata/rjin02/MedVLMBench/log/vqa/PathVQA/VILA1.5/train_lora_ML_seed42_vila \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred


# VQA-RAD, VILA1.5-8B
deepspeed --include localhost:2 --master_port 29603 run_train.py \
    --peft lora --lora_r 8 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./script/zero2.json \
    --task vqa --dataset VQA-RAD \
    --model VILA1.5 \
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
    --dataloader_num_workers 4 \
    --tune_modules ML

CUDA_VISIBLE_DEVICES=6 python run_eval.py \
    --task vqa --dataset VQA-RAD --split test \
    --image_path ./notgiven \
    --model VILA1.5 --model_path /bigdata/rjin02/MedVLMBench/log/vqa/VQA-RAD/VILA1.5/train_lora_ML_seed42_vila \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred


# FairVLMed, VILA1.5-8B
deepspeed --include localhost:3 --master_port 29604 run_train.py \
    --peft lora --lora_r 8 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./script/zero2.json \
    --task vqa --dataset Harvard-FairVLMed10k \
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

CUDA_VISIBLE_DEVICES=7 python run_eval.py \
    --task vqa --dataset Harvard-FairVLMed10k --split test \
    --image_path /data/rjin02/project/FairMedFM-DNE/data/FairVLMed10k \
    --model VILA1.5 --model_path /bigdata/rjin02/MedVLMBench/log/vqa/Harvard-FairVLMed10k/VILA1.5/train_lora_ML_seed42_vila \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred

################################## VILA-M3 ##################################

# SLAKE, VILA-M3
deepspeed --include localhost:4 --master_port 29601 run_train.py \
    --peft lora --lora_r 8 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./script/zero2.json \
    --task vqa --dataset SLAKE \
    --model VILA-M3 \
    --image_path ./data/SLAKE/imgs \
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
    --num_train_epochs 5 \
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
    --dataloader_num_workers 4 \
    --tune_modules ML

CUDA_VISIBLE_DEVICES=0 python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path ./data/SLAKE/imgs \
    --model VILA-M3 --model_path /bigdata/rjin02/MedVLMBench/log/vqa/SLAKE/VILA-M3/train_lora_ML_seed42_vila_m3 \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred


# PathVQA, VILA-M3
deepspeed --include localhost:1 --master_port 29602 run_train.py \
    --peft lora --lora_r 8 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./script/zero2.json \
    --task vqa --dataset PathVQA \
    --model VILA-M3 \
    --image_path ./notnedded \
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
    --dataloader_num_workers 4 \
    --tune_modules ML

CUDA_VISIBLE_DEVICES=1 python run_eval.py \
    --task vqa --dataset PathVQA --split test \
    --image_path /notgiven \
    --model VILA-M3 --model_path /bigdata/rjin02/MedVLMBench/log/vqa/PathVQA/VILA-M3/train_lora_ML_seed42_vila_m3 \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred


# VQA-RAD, VILA-M3
deepspeed --include localhost:0 --master_port 29603 run_train.py \
    --peft lora --lora_r 8 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./script/zero2.json \
    --task vqa --dataset VQA-RAD \
    --model VILA-M3 \
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
    --num_train_epochs 5 \
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
    --dataloader_num_workers 4 \
    --tune_modules ML

CUDA_VISIBLE_DEVICES=3 python run_eval.py \
    --task vqa --dataset VQA-RAD --split test \
    --image_path ./notgiven \
    --model VILA-M3 --model_path /bigdata/rjin02/MedVLMBench/log/vqa/VQA-RAD/VILA-M3/train_lora_ML_seed42_vila_m3 \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred


# FairVLMed, VILA-M3
deepspeed --include localhost:4 --master_port 29609 run_train.py \
    --peft lora --lora_r 8 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./script/zero2.json \
    --task vqa --dataset Harvard-FairVLMed10k \
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
    --dataloader_num_workers 4 \
    --tune_modules ML

CUDA_VISIBLE_DEVICES=3 python run_eval.py \
    --task vqa --dataset Harvard-FairVLMed10k --split test \
    --image_path /data/rjin02/project/FairMedFM-DNE/data/FairVLMed10k \
    --model VILA-M3 --model_path /bigdata/rjin02/MedVLMBench/log/vqa/Harvard-FairVLMed10k/VILA-M3/train_lora_ML_seed42_vila_m3 \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred


################################## Lingshu ##################################
conda activate torch2.6
# SLAKE, Lingshu
CUDA_VISIBLE_DEVICE=0 python run_train.py \
    --peft lora --lora_r 8 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --task vqa --dataset SLAKE \
    --model Lingshu \
    --image_path ./data/SLAKE/imgs \
    --model_path lingshu-medical-mllm/Lingshu-7B \
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
    --dataloader_num_workers 4 \
    --tune_modules ML

CUDA_VISIBLE_DEVICES=4 python run_eval.py \
    --task vqa --dataset SLAKE --split test \
    --image_path ./data/SLAKE/imgs \
    --model Lingshu --model_path /bigdata/rjin02/MedVLMBench/log/vqa/SLAKE/Lingshu/train_lora_ML_seed42_lingshu \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred


# PathVQA, Lingshu
CUDA_VISIBLE_DEVICES=1 python run_train.py \
    --peft lora --lora_r 8 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --task vqa --dataset PathVQA \
    --model Lingshu \
    --image_path ./notnedded \
    --model_path lingshu-medical-mllm/Lingshu-7B \
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
    --dataloader_num_workers 4 \
    --tune_modules ML


CUDA_VISIBLE_DEVICES=5 python run_eval.py \
    --task vqa --dataset PathVQA --split test \
    --image_path /notgiven \
    --model Lingshu --model_path /bigdata/rjin02/MedVLMBench/log/vqa/PathVQA/Lingshu/train_lora_ML_seed42_lingshu \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred


# VQA-RAD, Lingshu
CUDA_VISIBLE_DEVICES=2 python run_train.py \
    --peft lora --lora_r 8 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --task vqa --dataset VQA-RAD \
    --model Lingshu \
    --model_path lingshu-medical-mllm/Lingshu-7B \
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

CUDA_VISIBLE_DEVICES=6 python run_eval.py \
    --task vqa --dataset VQA-RAD --split test \
    --image_path ./notgiven \
    --model Lingshu --model_path /bigdata/rjin02/MedVLMBench/log/vqa/VQA-RAD/Lingshu/train_lora_ML_seed42_lingshu \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred


# FairVLMed, Lingshu
CUDA_VISIBLE_DEVICES=3 python run_train.py \
    --peft lora --lora_r 8 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --task vqa --dataset Harvard-FairVLMed10k \
    --model Lingshu \
    --image_path /data/rjin02/project/FairMedFM-DNE/data/FairVLMed10k \
    --model_path lingshu-medical-mllm/Lingshu-7B \
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

CUDA_VISIBLE_DEVICES=7 python run_eval.py \
    --task vqa --dataset Harvard-FairVLMed10k --split test \
    --image_path /data/rjin02/project/FairMedFM-DNE/data/FairVLMed10k \
    --model Lingshu --model_path /bigdata/rjin02/MedVLMBench/log/vqa/Harvard-FairVLMed10k/Lingshu/train_lora_ML_seed42_lingshu \
    --exp_path ./log \
    --cache_dir ./cache \
    --save_pred