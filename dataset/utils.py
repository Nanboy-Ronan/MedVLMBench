import os
import random

# import albumentations as albu
# import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

# from albumentations.augmentations import transforms as atransforms
# from albumentations.core.composition import Compose
# from einops import rearrange
from torch.utils.data import Dataset, WeightedRandomSampler
from easydict import EasyDict as edict

import dataset


def get_transform(args):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    return transform


def get_dataset(args, split, image_processor_callable=None):

    g = torch.Generator()
    g.manual_seed(args.random_seed)

    def seed_worker(worker_id):
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

    dataset_name = getattr(dataset, args.dataset)

    assert split in ["train", "validation", "test", "all"]

    if image_processor_callable is not None:
        transform = image_processor_callable
    else:
        transform = get_transform(args)

    dataset = dataset_name(data_args=edict(image_path=args.image_path), split=split, transform=transform)

    print("Loaded dataset: " + dataset.name)

    return dataset
