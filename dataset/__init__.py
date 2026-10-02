import os
import random
import numpy as np
import pandas as pd
import torch
import json
from easydict import EasyDict as edict
from collections import Counter

from dataset.utils import get_transform
from dataset.vqa import SLAKE, PathVQA, VQARAD, HarvardFairVLMed10kVQA, MedXpertQA, OmniMedVQA
from dataset.caption import HarvardFairVLMed10kCaption, MIMIC_CXRCaption
from dataset.diagnosis import (
    PneumoniaMNIST,
    BreastMNIST,
    DermaMNIST,
    Camelyon17,
    HAM10000Dataset,
    DrishtiDataset,
    ChestXrayDataset,
    GF3300Dataset,
    CXPDataset,
    PAPILADataset,
    FairVLMed10kDataset,
)

datasets = {
    "SLAKE-vqa": SLAKE,
    "PathVQA-vqa": PathVQA,
    "VQA-RAD-vqa": VQARAD,
    "Harvard-FairVLMed10k-vqa": HarvardFairVLMed10kVQA,
    "MedXpertQA-vqa": MedXpertQA,
    "OmniMedVQA-vqa": OmniMedVQA,
    "MIMIC_CXR-caption": MIMIC_CXRCaption,
    "PneumoniaMNIST-diagnosis": PneumoniaMNIST,
    "BreastMNIST-diagnosis": BreastMNIST,
    "DermaMNIST-diagnosis": DermaMNIST,
    "Camelyon17-diagnosis": Camelyon17,
    "HAM10000-diagnosis": HAM10000Dataset,
    "Drishti": DrishtiDataset,
    "ChestXray-diagnosis": ChestXrayDataset,
    "GF3300-diagnosis": GF3300Dataset,
    "HarvardFairVLMed10k-caption": HarvardFairVLMed10kCaption,
    "CheXpert-diagnosis": CXPDataset,
    "PAPILA-diagnosis": PAPILADataset,
    "HarvardFairVLMed10k-diagnosis": FairVLMed10kDataset,
}


class FractionalDataset(torch.utils.data.Dataset):
    """A deterministic subset that preserves benchmark dataset metadata."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)
        for attr in ("name", "modality", "split"):
            if hasattr(dataset, attr):
                setattr(self, attr, getattr(dataset, attr))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __getattr__(self, name):
        if name in {"dataset", "indices"}:
            raise AttributeError(name)
        return getattr(self.dataset, name)


def _apply_train_fraction(dataset, args, split):
    from utils.train_subset import make_manifest, validate_manifest, validate_selection, write_manifest

    fraction = float(getattr(args, "train_fraction", 1.0))
    maximum = getattr(args, "max_train_samples", None)
    manifest_path = getattr(args, "train_subset_manifest", None)
    validate_selection(fraction, maximum, manifest_path)
    if fraction == 1.0 and maximum is None and not manifest_path:
        return dataset
    if split != "train":
        raise ValueError("Training subsets can only be applied to the training split")
    dataset_name = getattr(args, "dataset", dataset.name)
    task = getattr(args, "task", "vqa")
    if manifest_path:
        with open(manifest_path) as stream:
            payload = json.load(stream)
        indices = validate_manifest(payload, dataset, dataset_name, task)
    else:
        payload = make_manifest(dataset, dataset_name, task, fraction, maximum,
                                int(getattr(args, "fraction_seed", 42)))
        indices = payload["indices"]
    subset = FractionalDataset(dataset, indices)
    subset.subset_manifest = payload
    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        write_manifest(os.path.join(output_dir, "train_subset_manifest.json"), payload)
    return subset


def get_dataset(args, image_processor_callable=None, split=None):

    g = torch.Generator()
    g.manual_seed(args.seed)

    def seed_worker(worker_id):
        np.random.seed(args.seed)
        random.seed(args.seed)

    dataset_name = datasets[f"{args.dataset}-{args.task}"]

    assert args.split in ["train", "validation", "test", "all"]

    if split is None:
        assert args.split in ["train", "validation", "test", "all"]
        split = args.split

    assert image_processor_callable is not None or args.task != "diagnosis"

    llava_train_models = {"LLaVA-1.5", "LLaVA-Med", "Quilt-LLaVA"}

    # LLaVA training performs its own image padding and preprocessing in the
    # trainer dataset wrapper. Passing a transform here causes double-processing
    # and type mismatches for PIL/tensor/BatchFeature inputs.
    if getattr(args, "model", None) in llava_train_models and args.task in {"vqa", "caption"} and split == "train":
        transform = None
    elif image_processor_callable is not None:
        transform = image_processor_callable
    else:
        transform = get_transform(args)

    dataset = dataset_name(data_args=edict(image_path=args.image_path, size=224), split=split, transform=transform)
    dataset = _apply_train_fraction(dataset, args, split)

    try:
        args.logger.info("Loaded dataset: " + dataset.name)
        args.logger.info(f"Dataset size: {len(dataset)}")
    except:
        print("Logger is not set.")

    if args.task == "diagnosis":
        report_label_distribution(dataset, args)

    return dataset


def report_label_distribution(dataset, args):
    label_counts = Counter()
    for i in range(len(dataset)):
        label = dataset[i]["label"].item()
        label_counts[label] += 1

    total = sum(label_counts.values())
    distribution = {label: count / total for label, count in label_counts.items()}

    args.logger.info("Label Distribution:")
    for label, freq in distribution.items():
        args.logger.info(f"Label {label}: {freq:.2%} ({label_counts[label]} samples)")

    num_classes = max(label_counts.keys()) + 1
    weights = [0.0] * num_classes
    for lbl, cnt in label_counts.items():
        weights[lbl] = total / (cnt * num_classes)

    dataset.class_weights = torch.tensor(weights, dtype=torch.float)
    args.logger.info(f"Class weights: {dataset.class_weights.tolist()}")
