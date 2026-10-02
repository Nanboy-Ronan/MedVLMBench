"""Fixed training subsets and portable manifests (no model dependencies)."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile


def validate_selection(fraction=1.0, maximum=None, manifest=None):
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("train_fraction must satisfy 0 < fraction <= 1")
    if maximum is not None and (type(maximum) is not int or maximum <= 0):
        raise ValueError("max_train_samples must be a positive integer")
    if sum((fraction != 1.0, maximum is not None, bool(manifest))) > 1:
        raise ValueError("Use only one of train_fraction, max_train_samples, train_subset_manifest")


def subset_label(fraction=1.0, maximum=None, manifest=None, seed=42):
    validate_selection(fraction, maximum, manifest)
    if manifest:
        payload = json.loads(Path(manifest).read_text())
        label = payload.get("selection_label", f"{payload['selected_size']}samples")
        if not isinstance(label, str) or not label or any(c not in '0123456789abcdefghijklmnopqrstuvwxyz.-' for c in label) or label in {'.', '..'}:
            raise ValueError("Invalid selection_label in subset manifest")
        return label
    suffix = f"-fseed{seed}" if seed != 42 else ""
    if maximum is not None:
        return f"{maximum}samples{suffix}"
    if fraction < 1:
        return f"{fraction * 100:.12g}pct{suffix}"
    return None


def partial_output_dir(root, task, dataset, model, run_name, label):
    parts = [str(root), task, dataset]
    if label:
        parts += ["partial-data-exp", label]
    return os.path.join(*parts, model, run_name)


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Unsupported metadata value: {type(value).__name__}")


def source_fingerprint(dataset):
    """Hash ordered VQA annotations without decoding images or encoding mount paths.

    Other dataset types without these metadata stores return None; their manifests
    still validate task, dataset name, split, length and index bounds.
    """
    if hasattr(dataset, "samples") and isinstance(dataset.samples, list):
        records = ({k: v for k, v in r.items() if k != "_abs_image_path"} for r in dataset.samples)
    elif hasattr(dataset, "ds") and hasattr(dataset.ds, "columns") and hasattr(dataset.ds, "to_dict"):
        records = dataset.ds.to_dict(orient="records")  # pandas, e.g. FairVLMed
    elif hasattr(dataset, "ds") and hasattr(dataset.ds, "column_names"):
        columns = [c for c in dataset.ds.column_names if c not in {"image", "images"}]
        records = dataset.ds.select_columns(columns)
    else:
        return None
    digest = hashlib.sha256()
    for row in records:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False, default=_json_default).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def make_manifest(dataset, dataset_name, task, fraction=1.0, maximum=None, seed=42):
    validate_selection(fraction, maximum)
    size = len(dataset)
    if size == 0:
        raise ValueError("Cannot select a subset of an empty training dataset")
    count = min(size, maximum) if maximum is not None else max(1, round(size * fraction))
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    indices = sorted(indices[:count])
    return {
        "schema_version": 2, "dataset": dataset_name, "task": task, "split": "train",
        "source_size": size, "selected_size": count,
        "source_fingerprint": source_fingerprint(dataset),
        "selection_label": subset_label(fraction, maximum, seed=seed) or "100pct",
        "fraction_requested": fraction, "max_train_samples": maximum,
        "fraction_realized": count / size, "fraction_seed": seed,
        "sampling_algorithm": "python-random-shuffle-v1", "indices": indices,
    }


def validate_manifest(payload, dataset, dataset_name, task):
    if payload.get("dataset") not in {dataset_name, getattr(dataset, "name", dataset_name)}:
        raise ValueError("Subset manifest belongs to a different dataset")
    if payload.get("task", task) != task or payload.get("split") != "train":
        raise ValueError("Subset manifest must belong to this task's training split")
    if payload.get("source_size") != len(dataset):
        raise ValueError("Training source size changed; regenerate the shared subset")
    indices = payload.get("indices")
    if not isinstance(indices, list) or not indices or any(type(i) is not int or not 0 <= i < len(dataset) for i in indices):
        raise ValueError("Invalid subset indices")
    if len(set(indices)) != len(indices) or payload.get("selected_size") != len(indices):
        raise ValueError("Duplicate indices or inconsistent selected_size")
    fingerprint = payload.get("source_fingerprint")
    if fingerprint is not None and fingerprint != source_fingerprint(dataset):
        raise ValueError("Training annotations/order changed; regenerate the shared subset")
    return indices


def write_manifest(path, payload):
    """Atomic per-rank write; refuse to mix different subsets in the same run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != payload:
            raise ValueError(f"Different subset already recorded at {path}; use a new output root")
        return
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.subset-', suffix='.tmp')
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint_subset_label(checkpoint):
    """Carry a partial-data experiment folder through evaluation of its checkpoint."""
    parts = Path(checkpoint).parts
    if "partial-data-exp" not in parts:
        return None
    position = parts.index("partial-data-exp")
    if position + 1 >= len(parts):
        raise ValueError("Missing subset label in checkpoint path")
    return parts[position + 1]
