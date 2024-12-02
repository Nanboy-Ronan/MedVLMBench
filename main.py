import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import parse_args

if __name__ == "__main__":
    args = parse_args.collect_args()
    args = create_exerpiment_setting(args)

    logger = basics.setup_logger(
        "train", args.save_folder, "history.log", screen=True, tofile=True)
    logger.info("Using following arguments for training.")
    logger.info(args)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    

    train_data, train_dataloader, train_meta = get_dataset(args, split="train")
    test_data, test_dataloader, test_meta = get_dataset(args, split="test")