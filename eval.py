import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from parse_args import collect_args_eval
from utils import basics
from model.utils import get_model
from dataset.utils import get_dataset
from eval.utils import get_benchmark


def create_exerpiment_setting(args):
    # get hash
    args.save_folder = os.path.join(
        args.exp_path,
        args.task,
        args.dataset,
        args.model,
        f"seed{args.random_seed}",
    )

    basics.creat_folder(args.save_folder)

    return args


if __name__ == "__main__":
    args = collect_args_eval()
    args = create_exerpiment_setting(args)

    logger = basics.setup_logger("log", args.save_folder, "log.log", screen=True, tofile=True)
    logger.info("Using following arguments for evaluation.")
    logger.info(args)

    args.logger = logger

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    if args.device == "cuda":
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = get_model(args)

    if model.image_processor is None:
        all_data, all_dataloader = get_dataset(args, split="all")
    else:
        all_data, all_dataloader = get_dataset(args, split="all", image_processor=model.image_processor)

    all_data, all_dataloader = get_dataset(args, split="all")
    # train_data, train_dataloader = get_dataset(args, split="train")
    # test_data, test_dataloader = get_dataset(args, split="test")

    benchmark = get_benchmark(args=args, dataset=all_data)
    benchmark.evaluate(args=args, model=model)

    args.logger.info("End of the evaluation")
