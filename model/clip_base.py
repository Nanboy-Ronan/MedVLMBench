import torch.nn as nn
from base import BaseModel
from easydict import EasyDict as edict


class CLIPLPModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)


    def forward_clip(self, image, qs, image_size=None):
        pass

    def encode_text(self, qs):
        pass

    def forward(self):
        pass

    def save(self, output_folder):
        pass
