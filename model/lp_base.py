import torch
import torch.nn as nn
from collections import OrderedDict
from model.base import BaseModel
from easydict import EasyDict as edict


class LPModel(BaseModel, nn.Module):
    def __init__(self, args, encoder, num_classes):
        super().__init__(args)
        self.num_classes = num_classes
        self.encoder = encoder
        self.head = torch.nn.Linear(self.encoder.feat_dim, num_classes)
        self.model = nn.Sequential(
            OrderedDict(
                    [
                        ("encoder", self.encoder),
                        ("head", self.head)
                    ]
                )
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

    def load_for_training(self, model_path):
        pass
        
    def load_from_pretrained(self, model_path, device, **kwargs):
        model_ckpt = torch.load(model_path)
        self.model.load_state_dict(model_ckpt)
        self.model.to(device)

    def forward(self, images):
        return self.model(images)