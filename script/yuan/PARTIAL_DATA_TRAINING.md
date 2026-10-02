# Fixed partial-data training

## MedVLMBench

Append ONE of these to the existing `run_train.py` / DeepSpeed command:

```bash
--train_fraction 0.1 --fraction_seed 42
# OR
--max_train_samples 1000 --fraction_seed 42
# OR (use exactly the shared selection; do not add fraction/max arguments)
--train_subset_manifest /absolute/path/shared_subset/train_subset_manifest.json
```

Selection happens once before the Trainer, not once per epoch. Samples are drawn
without replacement; only their iteration order may change each epoch. The unit
is a dataset row (for VQA, one question/answer, NOT one unique image or patient).
Fraction count is `max(1, round(N * fraction))`; max count is `min(N, max_train_samples)`.
The source is the training dataset after its existing filters (e.g. SLAKE English).
Full-data training with no subset option keeps its existing output paths.

Example output:

```text
EXP/vqa/SLAKE/partial-data-exp/10pct/Quilt-LLaVA/train_lora_ML_seed42_quilt_llava/
EXP/vqa/SLAKE/partial-data-exp/1000samples/Quilt-LLaVA/train_lora_ML_seed42_quilt_llava/
```

The model and run-name structure below the subset folder is unchanged. Non-default
subset seeds append `-fseed43` etc. to the subset folder to prevent collisions.
The run saves `train_subset_manifest.json`; `experiment_manifest.json` also records
the selection. Evaluation automatically keeps the partial-data folder when its
checkpoint path contains `partial-data-exp/<label>`; the evaluation dataset itself
is NOT subsampled. Preserve that part of the path when exporting/merging weights.

Quilt/Patho shell launchers accept `TRAIN_FRACTION`, `MAX_TRAIN_SAMPLES`,
`TRAIN_SUBSET_MANIFEST`, `FRACTION_SEED` in train mode. For example, prepend
`TRAIN_FRACTION=0.1 FRACTION_SEED=42` to the existing Quilt training command.

## Use exactly the same VQA examples in LLaMA-Factory

Run preparation once in the MedVLMBench environment (no model weights are loaded):

```bash
python script/yuan/prepare_train_subset.py \
  --dataset SLAKE \
  --image_path /research/d5/gds/yzhong22/datasets/SLAKE/imgs \
  --train_fraction 0.1 --fraction_seed 42 \
  --output_dir /research/d5/gds/yzhong22/datasets/shared_subsets/SLAKE/10pct
```

Use `--max_train_samples 1000` instead for an exact cap. `--image_path None` works
for the benchmark's HF-backed PathVQA/VQA-RAD loaders. The output directory must be
new. Preparation reads the existing
`/research/d5/gds/yzhong22/datasets/DATASET/vqa_sharegpt/train.json`,
checks the full row count and selected rows against the native dataset, then preserves
the selected `conversations` and `images` records verbatim. No images are copied or
encoded. It checks questions, answers, image counts, image references and file existence.
Default source root is `/research/d5/gds/yzhong22/datasets`; override the JSON location
with `--sharegpt_path /absolute/path/train.json`. The existing images must be accessible.
The native dataset still opens selected images during correspondence checks.
The script covers the six registered VQA datasets; native subset training also
works with the other training tasks.

Outputs:
- `train_subset_manifest.json`: same native training indices and ordered annotation fingerprint.
- `train.json`: unchanged selected ShareGPT records; their source indices are in the manifest.
- `dataset_info.json`: registration as `shared_subset`.

For MedVLMBench, add the generated manifest to the existing command:

```bash
--train_subset_manifest /research/d5/gds/yzhong22/datasets/shared_subsets/SLAKE/10pct/train_subset_manifest.json
```

For LLaMA-Factory, activate its environment, run from its repository root:

```bash
llamafactory-cli train configs/2025-vlmbench/vqa_partial/qwen2.5-vl/10pct/train_slake_sft.yaml
```

For another model, retain that model's existing recipe and apply these settings:

```yaml
dataset_dir: /research/d5/gds/yzhong22/datasets/shared_subsets/SLAKE/10pct
dataset: shared_subset
max_samples: null
val_size: 0
streaming: false
packing: false
neat_packing: false
overwrite_cache: true
# Remove tokenized_path, or use a fresh path dedicated to this model AND subset.
output_dir: /research/d5/gds/yzhong22/experiments/med_vlm_benchmark/vqa/SLAKE/partial-data-exp/10pct/patho_r1-7b-lora-ML
```

Ready-made train/merge YAMLs are in LLaMA-Factory's
`configs/2025-vlmbench/vqa_partial/{qwen2.5-vl,gemma3}/{10pct,...,90pct}/`
for SLAKE, PathVQA and VQA-RAD. They preserve the original model and training
hyperparameters, point to the fixed shared subset, disable additional truncation,
validation splitting and packing, and keep train/merge paths paired under
`DATASET/partial-data-exp/<ratio>/`. Original full-data YAMLs remain unchanged.
The preparation script only generates subset JSON, registration and manifest, not images or training configs.

This guarantees the same **input examples**, not identical token counts or training
steps: tokenization, batch sizes and DDP padding may differ. LF can discard malformed
examples during preprocessing, so verify its reported training example count equals
`selected_size` (packing is disabled). Both frameworks should use the same original
annotations and images. The manifest detects reordered/changed VQA annotations and
length mismatches; it does not hash the original image pixels. Other tasks without
supported annotation stores validate dataset/task/split/length/indices only.

Selection uses a private Python RNG. Older manifests from the previous Torch-based
fraction sampler can still be reused explicitly; the same numerical seed alone does
not reproduce the old sampler. Never independently apply the manifest's row indices
to an existing LF JSON unless you have verified identical row ordering/filtering.

## LLaMA-Factory's built-in subset settings

In the upstream loader, `max_samples: 1000` takes the first 1000 rows of each loaded
dataset, before preprocessing; it is not a random 10% sampler. `num_samples` in a
`dataset_info.json` entry randomly selects rows when loading, and can duplicate rows
if its target exceeds the dataset size. These selections are not repeated each epoch
in ordinary non-streaming training, but neither guarantees matching MedVLMBench's
selection. For a matched comparison, use the exported static file above and leave
`max_samples` unset/null. An existing `tokenized_path` can bypass these raw-data options.

Source: https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/loader.py
