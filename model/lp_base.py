import torch.nn as nn
from model.base import BaseModel
from easydict import EasyDict as edict


class LPModel(BaseModel, nn.Module):
    def __init__(self, args, num_classes):
        super().__init__(args)
        self.num_classes = num_classes

    def forward(self, images):
        pass

    def encode_text(self, text):
        pass