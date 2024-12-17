import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import parse_args
from utils import basics
from models.utils import get_model
from data.utils import get_dataset
from eval.utils import get_eval_engine


def create_exerpiment_setting(args):
    # get hash
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.lr = args.blr

    args.save_folder = os.path.join(
        args.exp_path,
        args.task,
        args.dataset,
        args.model,
        f"seed{args.random_seed}",
    )

    args.resume_path = args.save_folder
    basics.creat_folder(args.save_folder)

    # try:
    #     with open(f"configs/datasets/{args.dataset}.json", "r") as f:
    #         data_setting = json.load(f)
    #         data_setting["augment"] = False
    #         data_setting["test_meta_path"] = data_setting[
    #             f"test_{str.lower(args.sensitive_name)}_meta_path"]
    #         args.data_setting = data_setting

    #         if args.pos_class is not None:
    #             args.data_setting["pos_class"] = args.pos_class
    # except:
    #     args.data_setting = None

    # try:
    #     with open(f"configs/models/{args.model}.json", "r") as f:
    #         args.model_setting = json.load(f)
    # except:
    #     args.model_setting = None

    return args


if __name__ == "__main__":
    args = parse_args.collect_args()
    args = create_exerpiment_setting(args)

    logger = basics.setup_logger("train", args.save_folder, "history.log", screen=True, tofile=True)
    logger.info("Using following arguments for training.")
    logger.info(args)

    args.logger = logger

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = get_model(args)

    if model.image_processor is None:
        all_data, all_dataloader = get_dataset(args, split="all")
    else:
        all_data, all_dataloader = get_dataset(args, split="all", image_processor_callable=model.image_processor)

    all_data, all_dataloader = get_dataset(args, split="all")
    # train_data, train_dataloader = get_dataset(args, split="train")
    # test_data, test_dataloader = get_dataset(args, split="test")

    benchmark = get_eval_engine(args=args, dataset=all_data)
    benchmark.evaluate(args=args, model=model)

    args.logger.info("End of the Program")
