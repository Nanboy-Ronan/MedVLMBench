import torch.nn as nn
from model.base import BaseModel
from easydict import EasyDict as edict


class LPModel(BaseModel, nn.Module):
    def __init__(self, args):
        super().__init__(args)

    def forward(self, images):
        pass

    def encode_text(self, text):
        pass