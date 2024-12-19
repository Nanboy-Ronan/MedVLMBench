import os
import random
import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict

from dataset.utils import get_transform
from dataset.vqa import SLAKE
from dataset.vqa import PathVQA

datasets = {
        "SLAKE": SLAKE,
        "PathVQA": PathVQA
     }


def get_dataset(args, image_processor_callable=None):

    g = torch.Generator()
    g.manual_seed(args.random_seed)

    def seed_worker(worker_id):
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

    dataset_name = datasets[args.dataset]

    assert args.split in ["train", "validation", "test", "all"]
    
    if image_processor_callable is not None:
        transform = image_processor_callable
    else:
        transform = get_transform(args)

    dataset = dataset_name(data_args=edict(image_path=args.image_path), split=args.split, transform=transform)

    print("Loaded dataset: " + dataset.name)

    return dataset
