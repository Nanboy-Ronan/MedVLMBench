import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from model.base import BaseModel
from easydict import EasyDict as edict


class ImageProcessorCallable:
    """
    A callable wrapper for image processors to handle batches of tensors.
    It converts tensors to PIL images, applies the processor, and stacks the results.
    """
    def __init__(self, image_processor, transform_func=None):
        self.image_processor = image_processor
        if transform_func is None:
            self.transform_func = lambda x: x
        else:
            self.transform_func = transform_func

    def __call__(self, image_batch_tensor: torch.Tensor) -> torch.Tensor:
        device = image_batch_tensor.device
        pil_images = [to_pil_image(img_tensor) for img_tensor in image_batch_tensor]
        
        try:
            # Attempt to process the batch of PIL images directly (more efficient)
            processed = self.image_processor(pil_images, return_tensors="pt")
            return processed['pixel_values'].to(device)
        except (ValueError, TypeError, AttributeError):
            # Fallback to processing images one by one
            processed_tensors = [self.transform_func(self.image_processor(pil_image)) for pil_image in pil_images]
            return torch.stack(processed_tensors).to(device)


class CLIPBase(BaseModel, nn.Module):
    def __init__(self, text, num_classes, model, args=None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.prototype_text = text
        self.num_classes = num_classes
        self.model = model
        self.prototype = None  # Will be initialized by subclasses
        self.logit_scale = None # Should be set by subclasses

    def initialize_prototypes(self):
        """Initializes text prototypes. Should be called at the end of subclass __init__."""
        if self.prototype is None:
            with torch.no_grad():
                self.prototype = self.encode_text(self.prototype_text)

    def forward(self, images):
        image_features = self.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)

        text_features = F.normalize(self.prototype, dim=-1).to(images.device)

        if self.logit_scale is None:
            # Fallback for models without a logit scale
            logits = 100.0 * image_features @ text_features.T
        else:
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.T

        return logits

    def encode_image(self, images):
        raise NotImplementedError("Subclasses must implement `encode_image`")

    @torch.no_grad()
    def encode_text(self, text):
        raise NotImplementedError("Subclasses must implement `encode_text`")

    def load_for_training(self, model_path):
        pass

    def load_from_pretrained(self, model_path, device, **kwargs):
        model_ckpt = torch.load(model_path)
        self.model.load_state_dict(model_ckpt)
        self.model.to(device)