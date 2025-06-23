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
        self.head = torch.nn.Linear(self.encoder.feat_dim, self.num_classes)
        
        # Combine encoder and head into a single model attribute for consistent interface
        self.model = nn.Sequential(
            OrderedDict(
                    [
                        ("encoder", self.encoder),
                        ("head", self.head)
                    ]
                )
        )

        # Freeze the encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    def extract_features(self, images):
        """
        Subclasses must implement this method to extract features from the encoder's output.
        """
        raise NotImplementedError

    def forward(self, images):
        features = self.extract_features(images)
        return self.head(features)

    def load_for_training(self, model_path):
        pass
        
    def load_from_pretrained(self, model_path, device, **kwargs):
        # The model's state_dict is now from the nn.Sequential wrapper
        model_ckpt = torch.load(model_path)
        self.model.load_state_dict(model_ckpt)
        self.model.to(device)