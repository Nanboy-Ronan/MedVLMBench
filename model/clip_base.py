import torch.nn as nn
from model.base import BaseModel
from easydict import EasyDict as edict


class CLIPModel(BaseModel, nn.Module):
    def __init__(self, args):
        super().__init__(args)

    def forward_clip(self, images, text_features):
        pass

    def encode_text(self, text):
        pass

    def forward(self, images):
        pass
