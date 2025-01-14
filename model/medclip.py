import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from medclip import constants

from model.lp_base import LPModel
from model.lora_base import LoRALPModel
from model.clip_base import CLIPBase
from torchvision.transforms.functional import to_pil_image
from peft import LoftQConfig, LoraConfig, get_peft_model

from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision import transforms

from transformers.utils import TensorType
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import is_torch_tensor

from transformers import BatchFeature
from typing import Union, List, Optional


class MedCLIPFeatureExtractor:
    def __init__(self, 
                 crop_size=(224, 224),
                 do_center_crop=True,
                 do_convert_rgb=True,
                 do_normalize=True,
                 do_pad_square=True,
                 do_rescale=True,
                 do_resize=True,
                 image_mean=(0, 0, 0),
                 image_std=(0.26862954, 0.26130258, 0.27577711),
                 rescale_factor=0.5862785803043838,
                 size=224):
        self.crop_size = crop_size
        self.do_center_crop = do_center_crop
        self.do_convert_rgb = do_convert_rgb
        self.do_normalize = do_normalize
        self.do_pad_square = do_pad_square
        self.do_rescale = do_rescale
        self.do_resize = do_resize
        self.image_mean = image_mean
        self.image_std = image_std
        self.rescale_factor = rescale_factor
        self.size = size

        # Define transformations using torchvision
        transform_list = []

        if self.do_convert_rgb:
            transform_list.append(transforms.Lambda(lambda img: img.convert('RGB')))

        if self.do_pad_square:
            transform_list.append(transforms.Lambda(self.pad_to_square))

        if self.do_resize:
            transform_list.append(transforms.Resize(self.size))

        if self.do_center_crop:
            transform_list.append(transforms.CenterCrop(self.crop_size))

        # if self.do_rescale:
        #     transform_list.append(transforms.Lambda(lambda img: transforms.ToTensor()(img) * self.rescale_factor))

        transform_list.append(transforms.ToTensor())

        if self.do_rescale:
            transform_list.append(transforms.Lambda(lambda tensor: tensor * self.rescale_factor))  # Apply rescaling

        if self.do_normalize:
            transform_list.append(transforms.Normalize(mean=self.image_mean, std=self.image_std))

        self.transform = transforms.Compose(transform_list)

    def pad_to_square(self, img):
        max_dim = max(img.size)
        padding = [(max_dim - img.size[0]) // 2, (max_dim - img.size[1]) // 2]
        padding += [max_dim - img.size[0] - padding[0], max_dim - img.size[1] - padding[1]]
        return transforms.functional.pad(img, padding, fill=0, padding_mode='constant')

    def __call__(self, image):
        image = self.transform(image)
        return image

class ImageProcessorLPCallable:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image):
        device = image.device
        image_batch_pil = [to_pil_image(img_tensor) for img_tensor in image]
        # breakpoint()
        image = [torch.tensor(self.image_processor(pil_image)) for pil_image in image_batch_pil]
        image = torch.stack(image).to(device)
        return image 

class MedCLIPForDiagnosis(CLIPBase):
    def __init__(self, text, num_classes, *args, **kwargs) -> None:
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        model.from_pretrained()
        super().__init__(text=text, num_classes=num_classes, model=model)
    
        self.processor = MedCLIPProcessor()
        self.image_processor = self.processor.image_processor
        self.tokenizer = self.processor.tokenizer
        self.image_processor = ImageProcessorLPCallable(MedCLIPFeatureExtractor())
        self.image_processor_evaluation = self.image_processor
        self.model.cuda()
        self.prototype = self.encode_text(self.prototype)

    @torch.no_grad()
    def encode_text(self, text):
        assert len(text) == self.num_classes
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        return self.model.encode_text(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    
    def forward(self, images):
        image_outputs = self.model.encode_image(images)

        image_features = F.normalize(image_outputs, dim=-1)
        text_features = F.normalize(self.prototype, dim=-1).to(images.device)

        logits = 100.0 * image_features @ text_features.T
        
        return logits

class MedCLIPLoRAForDiagnosis(CLIPBase):
    def __init__(self, text, num_classes, *args, **kwargs) -> None:
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        model.from_pretrained()
        super().__init__(text=text, num_classes=num_classes, model=model)
    
        self.processor = MedCLIPProcessor()
        self.image_processor = self.processor.image_processor
        self.tokenizer = self.processor.tokenizer
        self.image_processor = ImageProcessorLPCallable(MedCLIPFeatureExtractor())
        self.image_processor_evaluation = self.image_processor
        self.model.cuda()
        self.prototype = self.encode_text(self.prototype)

    @torch.no_grad()
    def encode_text(self, text):
        assert len(text) == self.num_classes
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        return self.model.encode_text(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    
    def forward(self, images):
        image_outputs = self.model.encode_image(images)

        image_features = F.normalize(image_outputs, dim=-1)
        text_features = F.normalize(self.prototype, dim=-1).to(images.device)

        logits = 100.0 * image_features @ text_features.T
        
        return logits
    
class MedCLIPLPForDiagnosis(LPModel):
    def __init__(self, *args, **kwargs) -> None:
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        model.from_pretrained()
        vision_model = model.vision_model
        vision_model.feat_dim = 512
        super().__init__(encoder=vision_model, *args, **kwargs)

        self.image_processor = MedCLIPFeatureExtractor()
        self.image_processor = ImageProcessorLPCallable(self.image_processor)
        self.image_processor_evaluation = self.image_processor


class MedCLIPLoRALPForDiagnosis(LoRALPModel):
    def __init__(self, *args, **kwargs) -> None:
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        model.from_pretrained()
        vision_model = model.vision_model
        vision_model.feat_dim = 512
        lora_config = LoraConfig(target_modules=["qkv"])
        super().__init__(args=args, lora_config=lora_config, encoder=vision_model, num_classes=kwargs['num_classes'])
        
        self.image_processor = MedCLIPFeatureExtractor()
        self.image_processor = ImageProcessorLPCallable(self.image_processor)
        self.image_processor_evaluation = self.image_processor