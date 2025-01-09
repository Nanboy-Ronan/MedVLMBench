import torch
import torch.nn as nn
import torch.nn.functional as F

from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor

from model.lp_base import LPModel
from model.lora_base import LoRALPModel
from torchvision.transforms.functional import to_pil_image
from peft import LoftQConfig, LoraConfig, get_peft_model


import re
import random
from collections import defaultdict
import pdb
from typing import Union, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision import transforms

from transformers import CLIPFeatureExtractor, CLIPProcessor
from transformers.utils import TensorType
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import is_torch_tensor

import nltk
from PIL import Image
from sklearn.preprocessing import OrdinalEncoder

from medclip import constants

class MedCLIPFeatureExtractor(CLIPFeatureExtractor):
    def __init__(self, 
        do_resize=True, 
        size=224, 
        resample=Image.BICUBIC, 
        do_center_crop=True, 
        crop_size=224, 
        do_normalize=True, 
        image_mean=constants.IMG_MEAN, 
        image_std=constants.IMG_STD, 
        do_convert_rgb=False,
        do_pad_square=True,
        **kwargs):
        super().__init__(do_resize, size, resample, do_center_crop, crop_size, do_normalize, image_mean, image_std, do_convert_rgb, **kwargs)
        self.do_pad_square = do_pad_square
    
    def __call__(self, 
        images: Union[Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]], 
        return_tensors: Optional[Union[str, TensorType]] = None, 
        **kwargs) -> BatchFeature:
        """
        Main method to prepare for the model one or several image(s).

        <Tip warning={true}>

        NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
        PIL images.

        </Tip>

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
        """
        # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        # transformations (convert rgb + resizing + center cropping + normalization)
        if self.do_convert_rgb:
            images = [self.convert_rgb(image) for image in images]

        if self.do_pad_square:
            images = [self.pad_img(image,min_size=self.size) for image in images]
        
        if self.do_resize and self.size is not None and self.resample is not None:
            images = [
                self.resize(image=image, size=self.size, resample=self.resample)
                for image in images
            ]
        if self.do_center_crop and self.crop_size is not None:
            images = [self.center_crop(image, self.crop_size) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

        # add a RGB dim for each image
        images_ = []
        for image in images:
            if len(image.shape) == 2:
                image = image[None]
            images_.append(image)
        images = images_

        # return as BatchFeature
        data = {"pixel_values": images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs

    def pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size["shortest_edge"], x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im
    
    def convert_rgb(self, image):
        if image.mode != "RGB":
            return image.convert("RGB")
        return image
    
    def resize(self, image, size, resample):
        """Resize the image to the specified size."""
        if isinstance(image, np.ndarray):
            # Convert back to PIL Image for resizing
            image = Image.fromarray(image)
        return image.resize(size, resample)

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

class ImageProcessorLPCallable:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image):
        device = image.device
        image_batch_pil = [to_pil_image(img_tensor) for img_tensor in image]
        image = [torch.tensor(self.image_processor(pil_image)) for pil_image in image_batch_pil]
        image = torch.stack(image).to(device)
        return image 


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