import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.lp_base import LPModel

from medclip import MedCLIPModel, MedCLIPVisionModelViT


# class MedCLIP(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         self.model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
#         self.model.from_pretrained()
#         self.vision_model = self.model.vision_model
#         self.feat_dim = 512

#     def forward_clip(self, images, text_features):
#         image_features = self.model.encode_image(images)
#         text_features = F.normalize(text_features, dim=-1)
#         logit_scale = self.model.logit_scale.exp()

#         logits = logit_scale * image_features @ text_features.T

#         return logits

#     def encode_text(self, text):
#         input_ids = text["input_ids"].to(next(self.model.parameters()).device)

#         if "attention_mask" in text.keys():
#             attention_mask = text["attention_mask"].to(next(self.model.parameters()).device)
#         else:
#             attention_mask = None

#         return self.model.text_model(input_ids, attention_mask)

#     def forward(self, images):
#         return self.vision_model(images)

#     def from_pretrained(self, path):
#         self.model.from_pretrained() # cannot be used given path is not given


class MedCLIPLPForDiagnosis(LPModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        self.model.from_pretrained()
        self.vision_model = self.model.vision_model
        self.vision_model.feat_dim = 512
        if "lp" in self.args.usage:
            from wrappers import LinearProbeWrapper
            self.model = LinearProbeWrapper(self.vision_model, self.num_classes)


    def forward(self, images):
        return self.model.head(self.model.encoder(images))

    def load_for_training(self, path):
        pass
    
    def load_from_pretrained(self, model_path, device, **kwargs):
        model_ckpt = torch.load(model_path)
        self.model.load_state_dict(model_ckpt)
        self.model.to(device)