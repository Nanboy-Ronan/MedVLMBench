#!/usr/bin/env bash
set -euo pipefail

# The official Patho-R1 repository supplies an inference recipe, not its training scripts.
# MODE=train below is a separate Hugging Face Trainer/PEFT LoRA adaptation, not a
# reproduction of official CPT + SFT + RL or the benchmark's LLaMA-Factory recipe.
# The official checkpoint is gated and its terms restrict derivative training.
# MODE=eval|train|mdagent|ucagent. Obtain checkpoint access before running.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
MODE="${MODE:-eval}"
DATASET="${DATASET:-PathVQA}"
IMAGE_PATH="${IMAGE_PATH:?Set IMAGE_PATH to the dataset image directory}"
MODEL_PATH="${MODEL_PATH:-WenchuanZhang/Patho-R1-7B}"
EXP_PATH="${EXP_PATH:-$ROOT/output}"
SPLIT="${SPLIT:-test}"
SEED="${SEED:-42}"

if [[ "$MODE" == train ]]; then
  if [[ "${PATHO_R1_USE_CUSTOM_SFT:-}" != yes ]]; then
    echo "This is a custom HF Trainer/PEFT LoRA adaptation, not the official or matched LLaMA-Factory training workflow. Set PATHO_R1_USE_CUSTOM_SFT=yes only if that is the intended experiment." >&2
    exit 2
  fi
  if [[ "${PATHO_R1_TRAINING_AUTHORIZED:-}" != yes ]]; then
    echo "Patho-R1 fine-tuning requires prior written permission from the rights holders. Set PATHO_R1_TRAINING_AUTHORIZED=yes only after obtaining it." >&2
    exit 2
  fi
  python run_train.py \
    --task vqa --dataset "$DATASET" --model Patho-R1 \
    --model_path "$MODEL_PATH" --image_path "$IMAGE_PATH" \
    --output_dir "$EXP_PATH" --seed "$SEED" \
    --patho_r1_training_authorized True \
    --peft lora --tune_modules L --lora_r 16 --lora_alpha 32 \
    --bf16 True --bits 16 --num_train_epochs 1 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 --save_strategy epoch \
    --gradient_checkpointing True
elif [[ "$MODE" == eval || "$MODE" == mdagent || "$MODE" == ucagent ]]; then
  args=(--task vqa --dataset "$DATASET" --split "$SPLIT"
        --image_path "$IMAGE_PATH" --model Patho-R1
        --model_path "$MODEL_PATH" --exp_path "$EXP_PATH"
        --seed "$SEED" --save_pred)
  if [[ -n "${MODEL_BASE:-}" ]]; then args+=(--model_base "$MODEL_BASE"); fi
  if [[ "$MODE" != eval ]]; then args+=(--usage "$MODE"); fi
  python run_eval.py "${args[@]}"
else
  echo "MODE must be eval, train, mdagent, or ucagent" >&2
  exit 2
fi
