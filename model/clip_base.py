import torch.nn as nn
from model.base import BaseModel
from easydict import EasyDict as edict


class CLIPModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.encoder = model

        self.head = torch.nn.Linear(self.encoder.feat_dim, 2)

        for param in self.encoder.parameters():
            param.requires_grad = False


    def forward(self):
        return self.head(self.encoder(x))

    def save(self, output_folder):
        pass
