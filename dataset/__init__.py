import os
import random
import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict

from dataset.utils import get_transform
from dataset.vqa import SLAKE, PathVQA, VQARAD
from dataset.caption import HarvardFairVLMed10k, MIMIC_CXR
from dataset.diagnosis import PneumoniaMNIST, BreastMNIST, DermaMNIST

datasets = {
    "SLAKE": SLAKE,
    "PathVQA": PathVQA,
    "VQA-RAD": VQARAD,
    "Harvard-FairVLMed10k": HarvardFairVLMed10k,
    "MIMIC_CXR": MIMIC_CXR,
    "PneumoniaMNIST": PneumoniaMNIST,
    "BreastMNIST": BreastMNIST,
    "DermaMNIST": DermaMNIST
}


def get_dataset(args, image_processor_callable=None, split=None):

    g = torch.Generator()
    g.manual_seed(args.seed)

    def seed_worker(worker_id):
        np.random.seed(args.seed)
        random.seed(args.seed)

    dataset_name = datasets[args.dataset]
    
    assert args.split in ["train", "validation", "test", "all"]

    if split is None:
        assert args.split in ["train", "validation", "test", "all"]
        split = args.split

    if image_processor_callable is not None:
        transform = image_processor_callable
    else:
        transform = get_transform(args)
        
    dataset = dataset_name(
        data_args=edict(image_path=args.image_path, size=224), split=split, transform=transform
    )

    print("Loaded dataset: " + dataset.name)

    return dataset
