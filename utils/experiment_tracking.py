"""Machine-readable resource and reproducibility metadata for benchmark runs."""

from __future__ import annotations

import json
import os
import platform
import socket
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _json_safe(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, torch.dtype):
        return str(value)
    return str(value)


def _args_dict(args):
    if is_dataclass(args):
        raw = asdict(args)
    else:
        raw = vars(args).copy()
    raw.pop("logger", None)
    return _json_safe(raw)


def _underlying_module(model):
    for name in ("model", "encoder"):
        candidate = getattr(model, name, None)
        if isinstance(candidate, torch.nn.Module):
            return candidate
    return model if isinstance(model, torch.nn.Module) else None


def _parameter_counts(model):
    module = _underlying_module(model)
    if module is None:
        return {"total": None, "trainable": None, "trainable_fraction": None, "count_method": None}
    # ds_numel is the unpartitioned size under DeepSpeed ZeRO-3.
    total = sum(int(getattr(parameter, "ds_numel", parameter.numel())) for parameter in module.parameters())
    trainable = sum(
        int(getattr(parameter, "ds_numel", parameter.numel()))
        for parameter in module.parameters() if parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_fraction": trainable / total if total else None,
        "count_method": "torch_parameters_or_deepspeed_ds_numel",
    }


class TrainingTokenCounter:
    """Count the actual batches supplied to Trainer.training_step."""

    fields = (
        "microbatches",
        "examples_seen",
        "input_tokens",
        "padded_input_tokens",
        "supervised_tokens",
        "images_seen",
    )

    def __init__(self):
        self.counts = {key: 0 for key in self.fields}
        self.has_input_ids = False
        self.has_attention_mask = False
        self.has_labels = False

    def observe(self, inputs):
        if not isinstance(inputs, dict):
            return
        self.counts["microbatches"] += 1
        ids = inputs.get("input_ids")
        mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        if isinstance(ids, torch.Tensor):
            self.has_input_ids = True
            self.counts["examples_seen"] += ids.shape[0]
            self.counts["padded_input_tokens"] += ids.numel()
            if isinstance(mask, torch.Tensor) and mask.shape == ids.shape:
                self.has_attention_mask = True
                self.counts["input_tokens"] += int(mask.sum().item())
        elif isinstance(labels, torch.Tensor) and labels.ndim:
            self.counts["examples_seen"] += labels.shape[0]
        if isinstance(labels, torch.Tensor) and isinstance(ids, torch.Tensor):
            self.has_labels = True
            self.counts["supervised_tokens"] += int(labels.ne(-100).sum().item())

        images = inputs.get("images")
        if isinstance(images, torch.Tensor):
            self.counts["images_seen"] += images.shape[0] if images.ndim >= 4 else 1
        elif isinstance(images, (list, tuple)):
            self.counts["images_seen"] += len(images)
        elif isinstance(inputs.get("image_grid_thw"), torch.Tensor):
            self.counts["images_seen"] += inputs["image_grid_thw"].shape[0]
        elif isinstance(inputs.get("pixel_values"), torch.Tensor):
            pixels = inputs["pixel_values"]
            if pixels.ndim >= 4:
                self.counts["images_seen"] += pixels.shape[0]

    def result(self):
        result = self.counts.copy()
        if not self.has_input_ids:
            result["input_tokens"] = None
            result["padded_input_tokens"] = None
        elif not self.has_attention_mask:
            result["input_tokens"] = None
        if not self.has_labels:
            result["supervised_tokens"] = None
        return result


def _hardware():
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": len(gpus),
        "gpus": gpus,
    }


class ExperimentTracker:
    """Write ``experiment_manifest.json`` before and after a train/eval run."""

    def __init__(self, output_dir, phase, args, model, dataset):
        self.output_dir = output_dir
        self.path = os.path.join(output_dir, "experiment_manifest.json")
        self.phase = phase
        self.args = args
        self.model = model
        self.dataset = dataset
        self.started = None
        self.payload = {}
        self.training_counter = None
        self.trainer = None

    def start(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.payload = {
            "schema_version": 1,
            "status": "running",
            "phase": self.phase,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "name": getattr(self.dataset, "name", getattr(self.args, "dataset", None)),
                "split": getattr(self.dataset, "split", getattr(self.args, "split", None)),
                "n_samples": len(self.dataset),
                "source_n_samples": len(getattr(self.dataset, "dataset", self.dataset)),
            },
            "model": {
                "name": getattr(self.model, "name", getattr(self.args, "model", None)),
                "checkpoint": getattr(self.args, "model_path", None),
                "model_type": getattr(self.model, "model_type", None),
                "parameters": _parameter_counts(self.model),
            },
            "arguments": _args_dict(self.args),
            "hardware": _hardware(),
        }
        self._write()

    def attach_trainer(self, trainer):
        """Record the final train dataset/model and observe each training microbatch."""
        if self.phase != "train" or trainer is None:
            return
        self.trainer = trainer
        train_dataset = getattr(trainer, "train_dataset", None)
        if train_dataset is not None:
            self.payload["dataset"]["trainer_n_samples"] = len(train_dataset)
        self.payload["model"]["parameters"] = _parameter_counts(trainer.model)
        self.training_counter = TrainingTokenCounter()
        original_training_step = trainer.training_step

        def measured_training_step(model, inputs, *args, **kwargs):
            self.training_counter.observe(inputs)
            return original_training_step(model, inputs, *args, **kwargs)

        trainer.training_step = measured_training_step
        self.payload["training"] = {
            "token_count_status": "running",
            "token_count_method": "observed_Trainer_training_step_inputs",
            "token_count_note": (
                "input_tokens excludes padding when attention_mask is available; "
                "LLaVA image placeholders are counted as one token and do not "
                "include vision-tower patch tokens. Counts accumulate across "
                "actual training microbatches, including repeat epochs."
            ),
            "per_device_train_batch_size": getattr(self.args, "per_device_train_batch_size", None),
            "gradient_accumulation_steps": getattr(self.args, "gradient_accumulation_steps", None),
            "world_size": getattr(self.args, "world_size", 1),
            "planned_epochs": getattr(self.args, "num_train_epochs", None),
        }
        batch_size = self.payload["training"]["per_device_train_batch_size"]
        accumulation = self.payload["training"]["gradient_accumulation_steps"]
        if batch_size is not None and accumulation is not None:
            self.payload["training"]["nominal_global_batch_size"] = (
                int(batch_size) * int(accumulation) * int(self.payload["training"]["world_size"])
            )
        self._write()

    def finish(self, status, error=None):
        elapsed = time.perf_counter() - self.started if self.started is not None else None
        resource = {"wall_time_seconds": elapsed}
        if torch.cuda.is_available():
            resource["peak_gpu_memory_bytes_by_device"] = [
                torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count())
            ]
            resource["peak_gpu_reserved_bytes_by_device"] = [
                torch.cuda.max_memory_reserved(index) for index in range(torch.cuda.device_count())
            ]
        self.payload.update(
            {
                "status": status,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "resources": resource,
            }
        )
        if self.training_counter is not None:
            counts = self.training_counter.result()
            world_size = int(getattr(self.args, "world_size", 1))
            scope = "full_run" if world_size == 1 else "local_process"
            if status == "completed" and world_size > 1 and torch.distributed.is_initialized():
                try:
                    device = (
                        torch.device("cuda", torch.cuda.current_device())
                        if torch.distributed.get_backend() == "nccl"
                        else torch.device("cpu")
                    )
                    values = torch.tensor(
                        [counts[key] or 0 for key in TrainingTokenCounter.fields],
                        device=device, dtype=torch.long,
                    )
                    torch.distributed.all_reduce(values)
                    counts.update(zip(TrainingTokenCounter.fields, values.cpu().tolist()))
                    if not self.training_counter.has_input_ids:
                        counts["input_tokens"] = counts["padded_input_tokens"] = None
                    elif not self.training_counter.has_attention_mask:
                        counts["input_tokens"] = None
                    if not self.training_counter.has_labels:
                        counts["supervised_tokens"] = None
                    scope = "full_run"
                except Exception as exc:
                    self.payload["training"]["aggregation_error"] = str(exc)
            self.payload["training"].update({
                "token_count_status": "completed" if status == "completed" else "partial",
                "token_count_scope": scope,
                "observed": counts,
                "optimizer_steps_completed": getattr(getattr(self.trainer, "state", None), "global_step", None),
                "epochs_completed": getattr(getattr(self.trainer, "state", None), "epoch", None),
                "hf_trainer_reported_flos": getattr(getattr(self.trainer, "state", None), "total_flos", None),
                "hf_flos_note": (
                    "Hugging Face Trainer estimate; may omit multimodal vision compute "
                    "and should not be treated as measured hardware FLOPs."
                ),
            })
            train_size = self.payload["dataset"].get("trainer_n_samples")
            seen = counts["examples_seen"]
            if train_size and seen and counts["input_tokens"] is not None:
                self.payload["dataset"]["estimated_input_tokens_per_epoch"] = round(
                    counts["input_tokens"] * train_size / seen
                )
                self.payload["dataset"]["token_estimate_method"] = (
                    "trainer_dataset_size * observed_input_tokens / observed_examples_seen; "
                    "an estimate when preprocessing or sampling varies by epoch"
                )
            if train_size and seen and counts["supervised_tokens"] is not None:
                self.payload["dataset"]["estimated_supervised_tokens_per_epoch"] = round(
                    counts["supervised_tokens"] * train_size / seen
                )
        if error:
            self.payload["error"] = error
        self._write()

    def _write(self):
        if self.phase == "train":
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else int(os.environ.get("RANK", "0"))
            )
            if rank != 0:
                return
        temporary = self.path + ".tmp"
        with open(temporary, "w") as stream:
            json.dump(_json_safe(self.payload), stream, indent=2, sort_keys=True)
        os.replace(temporary, self.path)


def merge_worker_manifests(output_dir, worker_dirs, args):
    """Merge per-GPU inference manifests into one run-level manifest."""
    manifests = []
    for worker_dir in worker_dirs:
        path = os.path.join(worker_dir, "experiment_manifest.json")
        with open(path) as stream:
            manifests.append(json.load(stream))
    payload = {
        "schema_version": 1,
        "status": "completed" if all(item["status"] == "completed" for item in manifests) else "failed",
        "phase": "evaluation",
        "distributed": True,
        "world_size": len(manifests),
        "started_at_utc": min(item["started_at_utc"] for item in manifests),
        "finished_at_utc": max(item["finished_at_utc"] for item in manifests),
        "dataset": manifests[0]["dataset"],
        "model": manifests[0]["model"],
        "arguments": _args_dict(args),
        "hardware": manifests[0]["hardware"],
        "resources": {
            "wall_time_seconds": max(item["resources"]["wall_time_seconds"] for item in manifests),
            "workers": [item["resources"] for item in manifests],
        },
    }
    path = os.path.join(output_dir, "experiment_manifest.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(_json_safe(payload), stream, indent=2, sort_keys=True)
    os.replace(temporary, path)
