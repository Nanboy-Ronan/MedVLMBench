import json
import os
import random
import argparse
from utils import constants

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import basics
from models import get_model
from data import get_dataset
from eval import get_eval_engine


def collect_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--task", default="vqa", choices=constants.TASKS)
    parser.add_argument(
        "--dataset",
        default="SLAKE",
        choices=constants.DATASETS,
    )
    parser.add_argument("--image_path", type=str, help="local path to images")
    parser.add_argument("--split", type=str, default="all", help="dataset split for evaluation")

    # evaluation
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=10, help="logging frequency during evaluation")
    parser.add_argument("--save_pred", action="store_true", help="whether save predictions during evaluation")

    parser.add_argument("--hash_id", type=str, default="")

    # network
    parser.add_argument(
        "--model",
        default="BLIP",
        choices=constants.MODELS,
    )
    parser.add_argument("--context_length", default=77)
    parser.add_argument("--model_path", type=str, default="", help="explicitly indentify checkpoint path to resume.")

    # misc
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--eval_print_freq", type=int, default=100, help="logging frequency (step)")
    parser.add_argument("--exp_path", type=str, default="./output")
    parser.add_argument("--wandb_name", type=str, default="baseline")
    parser.add_argument("--if_wandb", type=bool, default=False)

    args = parser.parse_args()

    assert args.dataset in getattr(
        constants, f"{str.upper(args.task)}_DATASETS"
    ), f"dataset {args.dataset} is not supported for task {args.task}"

    args.save_folder = os.path.join(
        args.exp_path,
        args.task,
        args.dataset,
        args.model,
        f"seed{args.random_seed}",
    )

    basics.creat_folder(args.save_folder)

    if args.cache_dir is not None:
        os.environ["HF_HOME"] = args.cache_dir

    return args


if __name__ == "__main__":
    args = collect_args()

    logger = basics.setup_logger("eval", args.save_folder, "eval.log", screen=True, tofile=True)
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

    model = get_model(args=args, device=args.device)
    dataset = get_dataset(args, image_processor_callable=model.image_processor_callable)

    eval_engine = get_eval_engine(args=args, dataset=dataset)
    eval_engine.evaluate(args=args, model=model)

    args.logger.info("End of the evaluation")
