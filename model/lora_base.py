import torch
import torch.nn as nn
import copy
from collections import OrderedDict
from model.base import BaseModel
from easydict import EasyDict as edict
from peft import LoftQConfig, LoraConfig, get_peft_model


class LoRALPModel(BaseModel, nn.Module):
    def __init__(self, args, lora_config, encoder, num_classes):
        super().__init__(args)
        self.num_classes = num_classes
        self.lora_config = lora_config
        self.encoder = encoder
        self.head = torch.nn.Linear(encoder.feat_dim, num_classes)
        self.encoder = copy.deepcopy(get_peft_model(self.encoder, lora_config))
        self.model = nn.Sequential(
            OrderedDict(
                    [
                        ("encoder", self.encoder),
                        ("head", self.head)
                    ]
                )
        )
    
    def load_for_training(self, model_path):
        pass
        
    def load_from_pretrained(self, model_path, device, **kwargs):
        model_ckpt = torch.load(model_path)
        self.model.load_state_dict(model_ckpt)
        self.model.to(device)

    def forward(self, images):
        self.model(images)