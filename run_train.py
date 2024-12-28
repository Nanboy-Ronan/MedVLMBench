import json
import os, sys
import random
import argparse
from utils import constants

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from utils import basics
from model import get_model
from dataset import get_dataset

from train import get_train_engine


@dataclass
class Arguments(transformers.TrainingArguments):
    # data
    task: str = field(default="vqa")
    dataset: str = field(default="SLAKE")
    image_path: str = field(default="")
    image_aspect_ratio: str = field(default="pad")

    # train
    optim: str = field(default="adamw_torch")
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    tune_modules: str = "VML"  # V for vision tower, M for multimodal projector, L for LLM
    lora_enable: bool = False
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    mm_projector_lr: Optional[float] = None  # for LLMs
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True, metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    group_by_modality_length: bool = field(default=False)
    deepspeed_plugin: Optional[str] = field(default=None)

    # network
    model: str = field(default="LLaVA")
    version: str = field(default="v1")
    context_length: int = field(default=77)
    model_path: str = field(default=None, metadata={"help": "explicitly indentify checkpoint path to resume."})
    model_base: str = field(default=None)
    freeze_backbone: bool = field(default=False)
    ## LlaVA
    tune_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-2)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")

    # misc
    # exp_path: str = field(default="")
    # device: Optional[str] = field(default="cuda")
    cache_dir: Optional[str] = field(default=None)
    if_wandb: Optional[str] = False
    wandb_name: Optional[str] = field(default=None)


def setup_args(args):
    assert args.task in constants.TASKS, f"Task {args.task} is not supported. Supported tasks: {constants.TASKS}"
    assert args.model in constants.MODELS, f"Model {args.model} is not supported. Supported models: {constants.MODELS}"
    assert (
        args.dataset in constants.DATASETS
    ), f"Dataset {args.task} is not supported. Supported datasets: {constants.DATASETS}"
    assert args.dataset in getattr(
        constants, f"{str.upper(args.task)}_DATASETS"
    ), f"dataset {args.dataset} is not supported for task {args.task}"

    args.output_dir = os.path.join(
        args.output_dir,
        args.task,
        args.dataset,
        args.model,
        f"train_{args.tune_modules}_seed{args.seed}",
    )
    args.split = "train"
    args.tune_mm_mlp_adapter = "M" in args.tune_modules

    basics.creat_folder(args.output_dir)

    return args


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((Arguments))
    args = parser.parse_args_into_dataclasses()[0]
    # args = parser.parse_args()
    args = setup_args(args)
    # args = collect_args()

    logger = basics.setup_logger("train", args.output_dir, "train.log", screen=True, tofile=True)
    logger.info("Using following arguments for training.")
    logger.info(args)

    args.logger = logger

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_wrapped = get_model(args=args, device=args.device.type)
    model_wrapped.load_for_training(args.model_path)

    dataset = get_dataset(args)
    train_engine = get_train_engine(args, model_wrapped=model_wrapped, dataset=dataset)
    train_engine.train()

    logger.info("End of the training")


# def collect_args():
#     parser = argparse.ArgumentParser()

#     # data
#     parser.add_argument("--task", default="vqa", choices=constants.TASKS)
#     parser.add_argument(
#         "--dataset",
#         default="SLAKE",
#         choices=constants.DATASETS,
#     )
#     parser.add_argument("--image_path", type=str, default="", help="local path to images")
#     parser.add_argument("--split", type=str, default="all", help="dataset split for evaluation")
#     parser.add_argument("--image_aspect_ratio", type=str, default="pad")

#     # train
#     parser.add_argument("--learning_rate", type=float, default=2e-4)
#     parser.add_argument("--mm_projector_lr", type=float, default=2e-5)  # for LLMs
#     parser.add_argument("--per_device_train_batch_size", type=int, default=16)
#     parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
#     parser.add_argument("--num_train_epochs", type=int, default=1)
#     parser.add_argument("--weight_decay", type=float, default=0.0)
#     parser.add_argument("--warmup_ratio", type=float, default=0.03)
#     parser.add_argument("--lr_scheduler_type", type=str, default="cosine")

#     parser.add_argument("--freeze_backbone", type=bool, default=False)
#     parser.add_argument("--lora_enable", action="store_true")
#     parser.add_argument("--lora_r", type=int, default=128)
#     parser.add_argument("--lora_alpha", type=int, default=256)
#     parser.add_argument("--lora_dropout", type=float, default=0.05)
#     parser.add_argument("--lora_weight_path", type=str)
#     parser.add_argument("--lora_bias", type=str, default="none")
#     parser.add_argument("--bf16", type=bool, default=False)
#     parser.add_argument("--fp16", type=bool, default=False)
#     parser.add_argument("--bits", type=int, default=16)
#     parser.add_argument("--gradient_checkpointing", type=bool, default=True)
#     parser.add_argument("--model_max_length", type=int, default=2048)

#     # network
#     parser.add_argument(
#         "--model",
#         default="BLIP",
#         choices=constants.MODELS,
#     )
#     parser.add_argument("--version", type=str, default="v1")
#     parser.add_argument("--context_length", default=77)
#     parser.add_argument("--model_path", type=str, default="", help="explicitly indentify checkpoint path to resume.")
#     ## LLaVA
#     parser.add_argument("--mm_vision_select_layer", type=int, default=-2)
#     parser.add_argument("--mm_projector_type", type=str, default="mlp2x_gelu")
#     parser.add_argument("--mm_use_im_start_end", type=bool, default=False)
#     parser.add_argument("--mm_use_im_patch_token", type=bool, default=False)
#     parser.add_argument("--mm_vision_select_feature", default=None)
#     parser.add_argument("--mm_patch_merge_type", default=None)
#     parser.add_argument("--pretrain_mm_mlp_adapter", default=None)

#     # misc
#     parser.add_argument("--device", default="cuda")
#     parser.add_argument("--cache_dir", default=None)
#     parser.add_argument("--eval_print_freq", type=int, default=100, help="logging frequency (step)")
#     parser.add_argument("--exp_path", type=str, default="./output")
#     parser.add_argument("--wandb_name", type=str, default="baseline")
#     parser.add_argument("--if_wandb", type=bool, default=False)

#     args = parser.parse_args()

#     assert args.dataset in getattr(
#         constants, f"{str.upper(args.task)}_DATASETS"
#     ), f"dataset {args.dataset} is not supported for task {args.task}"

#     args.save_folder = os.path.join(
#         args.exp_path,
#         args.task,
#         args.dataset,
#         args.model,
#         f"seed{args.random_seed}",
#     )

#     basics.creat_folder(args.save_folder)

#     if args.cache_dir is not None:
#         os.environ["HF_HOME"] = args.cache_dir
#         os.environ["TRANSFORMERS_CACHE"] = args.cache_dir

#     return args
