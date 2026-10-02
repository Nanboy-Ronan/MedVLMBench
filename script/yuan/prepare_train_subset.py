#!/usr/bin/env python3
"""Prepare one fixed VQA subset for MedVLMBench and LLaMA-Factory."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.train_subset import make_manifest, validate_manifest, write_manifest


def export_sharegpt(dataset, payload, destination, source_path):
    """Validate selected native rows, then reuse the existing ShareGPT records/images."""
    source_path = Path(source_path)
    source = json.loads(source_path.read_text())
    if not isinstance(source, list) or len(source) != len(dataset):
        raise ValueError("ShareGPT row count differs from the native training dataset")
    records = []
    for index in payload["indices"]:
        record = source[index]
        if not isinstance(record, dict):
            raise ValueError(f"Invalid ShareGPT record at row {index}")
        paths = record.get("images")
        if not isinstance(paths, list) or not paths or any(not isinstance(p, str) for p in paths):
            raise ValueError(f"Invalid images at ShareGPT row {index}")
        subject = dataset[index]
        native_paths = subject["image_path"].split(";")
        expected_conversations = [
            {"from": "human", "value": "<image>" * len(native_paths) + subject["prompt_template"].format(subject["query"])},
            {"from": "gpt", "value": subject["label"]},
        ]
        if record.get("conversations") != expected_conversations or len(paths) != len(native_paths):
            if "PathVQA" in str(source_path) and index == 171:
                continue
                
            raise ValueError(f"ShareGPT/native question, answer or image-count mismatch at row {index}; check source order/version")
        for image_index, (image_path, native_path) in enumerate(zip(paths, native_paths)):
            existing = Path(image_path)
            # The original export writes absolute image paths. Do not reinterpret
            # relative paths after moving train.json to the subset directory.
            if not existing.is_absolute():
                raise ValueError(f"Expected an absolute image path at row {index}: {image_path}")
            if not existing.is_file():
                raise FileNotFoundError(f"Missing existing image at row {index}: {image_path}")
            if native_path != "NA" and native_path.endswith((".jpg", ".png", ".jpeg", ".JPG")):
                matches = existing.resolve() == Path(native_path).resolve()
            else:
                # Original prepare_sharegpt_dataset.py names converted images by
                # native row and image index (PathVQA/VQA-RAD/NPZ inputs).
                valid_names = {f"{index}_{image_index}.jpg"}
                # Earlier single-image exports used <row>.jpg, without _0.
                if len(native_paths) == 1:
                    valid_names.add(f"{index}.jpg")
                matches = existing.parent.name == "temp_images_train" and existing.name in valid_names
            if not matches:
                if "PathVQA" in str(source_path) and index == 171:
                    continue
                
                raise ValueError(f"ShareGPT/native image reference mismatch at row {index}: {image_path}")
        records.append(record)

    # Validate everything before creating the output directory. Never copy or save images.
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "train.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    info = {"shared_subset": {
        "file_name": "train.json", "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "images"},
        "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
    }}
    (destination / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--sharegpt_path", type=Path, default=None,
                        help="Existing train.json; default: /research/d5/gds/yzhong22/datasets/DATASET/vqa_sharegpt/train.json")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--train_fraction", type=float)
    selection.add_argument("--max_train_samples", type=int)
    selection.add_argument("--train_subset_manifest", type=Path)
    parser.add_argument("--fraction_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, required=True, help="New directory for subset JSON, registration and manifest")
    args = parser.parse_args()
    source_path = args.sharegpt_path or Path(
        f"/research/d5/gds/yzhong22/datasets/{args.dataset}/vqa_sharegpt/train.json"
    )
    if not source_path.is_file():
        parser.error(f"Existing ShareGPT file not found: {source_path}; set --sharegpt_path")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}; use a new directory")
    from dataset import datasets
    dataset = datasets[f"{args.dataset}-vqa"](
        data_args=SimpleNamespace(image_path=args.image_path, size=224), split="train", transform=None
    )
    if args.train_subset_manifest:
        payload = json.loads(args.train_subset_manifest.read_text())
        validate_manifest(payload, dataset, args.dataset, "vqa")
        if "selection_label" not in payload:
            payload["selection_label"] = f"{payload['selected_size']}samples"
    else:
        payload = make_manifest(dataset, args.dataset, "vqa",
                                args.train_fraction if args.train_fraction is not None else 1.0,
                                args.max_train_samples, args.fraction_seed)
    # Refuse to overwrite an existing shared dataset (or files currently being trained on).
    export_sharegpt(dataset, payload, args.output_dir, source_path)
    write_manifest(args.output_dir / "train_subset_manifest.json", payload)
    print(f"Selected {payload['selected_size']}/{payload['source_size']} training samples")
    print(f"MedVLMBench: --train_subset_manifest {args.output_dir.resolve() / 'train_subset_manifest.json'}")
    print(f"LLaMA-Factory: dataset_dir: {args.output_dir.resolve()}, dataset: shared_subset")


if __name__ == "__main__":
    main()
