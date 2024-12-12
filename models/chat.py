import torch.nn as nn
from .base import BaseModel
from easydict import EasyDict as edict


class ChatMetaModel(BaseModel):
    def __init__(self, model_args):
        super().__init__(model_args)

        self.constants = edict(
            IGNORE_INDEX=-100,
            IMAGE_TOKEN_INDEX=-200,
            DEFAULT_IMAGE_TOKEN="<image>",
            DEFAULT_IMAGE_PATCH_TOKEN="<im_patch>",
            DEFAULT_IM_START_TOKEN="<im_start>",
            DEFAULT_IM_END_TOKEN="<im_end>",
            IMAGE_PLACEHOLDER="<image-placeholder>",
        )

    def infer_vision_language(self, image, qs):
        pass

    def infer_language(self, qs):
        pass

    def init_for_training(self):
        pass

    def save(self, output_folder):
        pass
