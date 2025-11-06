import copy
import glob
import os
import random
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from model.release.vila.constants import DEFAULT_IMAGE_TOKEN
from model.release.vila.data.base import BaseDataset
from model.release.vila.media import Image, Video
from model.release.vila.mm_utils import dynamic_process_images_and_prompt, process_images
from model.release.vila.train.args import DataArguments
from model.release.vila.utils import io, make_list
from model.release.vila.utils.logging import logger
from model.release.vila.utils.media import extract_media
from model.release.vila.utils.tokenizer import preprocess_conversation


def _process_image(image: List[Any], data_args: DataArguments) -> torch.Tensor:
    return process_images(image, data_args.image_processor, data_args)


def _remove_media_tokens(text: str) -> str:
    for token in ["<image>", "<video>"]:
        text = text.replace(token + "\n", "").replace("\n" + token, "").replace(token, "")
    return text.strip()
