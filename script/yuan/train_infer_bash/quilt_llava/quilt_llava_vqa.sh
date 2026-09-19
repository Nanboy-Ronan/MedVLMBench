#!/usr/bin/env bash
set -euo pipefail

# MODE=eval|train|mdagent|ucagent. Run from any directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
MODE="${MODE:-eval}"
DATASET="${DATASET:-PathVQA}"
IMAGE_PATH="${IMAGE_PATH:?Set IMAGE_PATH to the dataset image directory}"
MODEL_PATH="${MODEL_PATH:-wisdomik/Quilt-Llava-v1.5-7b}"
EXP_PATH="${EXP_PATH:-$ROOT/output}"
SPLIT="${SPLIT:-test}"
SEED="${SEED:-42}"

if [[ "$MODE" == train ]]; then
  # Match the LLaVA-1.5 LoRA protocol: 2 samples/GPU x 8 accumulation
  # gives an effective batch of 16 per GPU (times the number of GPUs).
  DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./script/zero3.json}"
  CACHE_DIR="${CACHE_DIR:-${HF_HOME:-$EXP_PATH/cache}}"
  if [[ ! -f "$DEEPSPEED_CONFIG" ]]; then
    echo "DeepSpeed config not found: $DEEPSPEED_CONFIG (set DEEPSPEED_CONFIG to its location)" >&2
    exit 2
  fi
  deepspeed --master_port "${MASTER_PORT:-29599}" run_train.py \
    --peft lora --lora_r 16 --lora_alpha 32 --mm_projector_lr 2e-5 \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --task vqa --dataset "$DATASET" --model Quilt-LLaVA --version v1 \
    --image_path "$IMAGE_PATH" --model_path "$MODEL_PATH" \
    --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2 \
    --mm_use_im_start_end False --mm_use_im_patch_token False \
    --image_aspect_ratio pad --group_by_modality_length True \
    --bf16 True --bits 16 --output_dir "$EXP_PATH" --cache_dir "$CACHE_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 --evaluation_strategy no \
    --save_strategy steps --save_steps 50000 --save_total_limit 1 \
    --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 \
    --lr_scheduler_type cosine --logging_steps 1 --tf32 True \
    --model_max_length 2048 --gradient_checkpointing False \
    --dataloader_num_workers 4 --tune_modules ML --seed "$SEED"
elif [[ "$MODE" == eval || "$MODE" == mdagent || "$MODE" == ucagent ]]; then
  args=(--task vqa --dataset "$DATASET" --split "$SPLIT"
        --image_path "$IMAGE_PATH" --model Quilt-LLaVA
        --model_path "$MODEL_PATH" --exp_path "$EXP_PATH"
        --seed "$SEED" --save_pred)
  if [[ -n "${MODEL_BASE:-}" ]]; then args+=(--model_base "$MODEL_BASE"); fi
  if [[ "$MODE" != eval ]]; then args+=(--usage "$MODE"); fi
  python run_eval.py "${args[@]}"
else
  echo "MODE must be eval, train, mdagent, or ucagent" >&2
  exit 2
fi
