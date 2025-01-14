import torch
import torch.nn as nn
from model.base import BaseModel
from easydict import EasyDict as edict


class CLIPBase(BaseModel, nn.Module):
    def __init__(self, text, num_classes, model, *args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.prototype = text
        self.num_classes = num_classes
        self.model = model

    def forward(self, images, text_features):
        pass
    
    def encode_img(self, images):
        pass

    @torch.no_grad()
    def encode_text(self, text):
        pass
    
    def load_for_training(self, model_path):
        pass
        
    def load_from_pretrained(self, model_path, device, **kwargs):
        model_ckpt = torch.load(model_path)
        self.model.load_state_dict(model_ckpt)
        self.model.to(device)
