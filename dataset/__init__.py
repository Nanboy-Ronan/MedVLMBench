import os
import random
import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict
from collections import Counter

from dataset.utils import get_transform
from dataset.vqa import SLAKE, PathVQA, VQARAD, HarvardFairVLMed10kVQA
from dataset.caption import HarvardFairVLMed10kCaption, MIMIC_CXRCaption
from dataset.diagnosis import PneumoniaMNIST, BreastMNIST, DermaMNIST, Camelyon17, HAM10000Dataset, DrishtiDataset, ChestXrayDataset, GF3300Dataset

datasets = {
    "SLAKE": SLAKE,
    "PathVQA": PathVQA,
    "VQA-RAD": VQARAD,
    "Harvard-FairVLMed10k": HarvardFairVLMed10kVQA,
    "MIMIC_CXR": MIMIC_CXRCaption,
    "PneumoniaMNIST": PneumoniaMNIST,
    "BreastMNIST": BreastMNIST,
    "DermaMNIST": DermaMNIST,
    "Camelyon17": Camelyon17,
    "HAM10000": HAM10000Dataset,
    "Drishti": DrishtiDataset,
    "ChestXray": ChestXrayDataset,
    "GF3300": GF3300Dataset,
    "HarvardFairVLMed10k-caption": HarvardFairVLMed10kCaption
}


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

    if image_processor_callable is not None:
        transform = image_processor_callable
    else:
        transform = get_transform(args)

    dataset = dataset_name(data_args=edict(image_path=args.image_path, size=224), split=split, transform=transform)

    try:
        args.logger.info("Loaded dataset: " + dataset.name)
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
