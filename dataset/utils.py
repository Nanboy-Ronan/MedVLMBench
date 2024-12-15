import os
import random

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from albumentations.augmentations import transforms as atransforms
from albumentations.core.composition import Compose
from einops import rearrange
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, WeightedRandomSampler
from easydict import EasyDict as edict

import dataset


def get_dataset(args, split):

    g = torch.Generator()
    g.manual_seed(args.random_seed)

    def seed_worker(worker_id):
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

    dataset_name = getattr(dataset, args.dataset)
    # data = dataset_name(meta, args.sensitive_name,
    #                     transform, path_to_images=image_path)

    if args.dataset == "Slake":
        assert split in ["train", "validation", "test", "all"]
        # if split == "all":
            # data = dataset_name(edict(image_path="./data/SLAKE/imgs"), transform=transforms.PILToTensor())
        # else:
        data = dataset_name(edict(image_path="./data/SLAKE/imgs"), split=split, transform=transforms.PILToTensor())
    else:
        raise NotImplementedError()

    if split == "train":
        sampler = None

        data_loader = torch.utils.data.DataLoader(
            data,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=(args.method != "resampling"),
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
        )
    elif split == "test":
        data_loader = torch.utils.data.DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
        )
    elif split == "all":
        # currently same as test as "all" is designed to zero-shot the model on the entire dataset
        data_loader = torch.utils.data.DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
        )
    else:
        raise NotImplementedError()
    
    print("loaded dataset ", args.dataset)

    return data, data_loader