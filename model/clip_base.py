import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torchvision.transforms.functional import to_pil_image
from model.base import BaseModel
from easydict import EasyDict as edict
from utils.utils import maybe_zero_3


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

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        try:
            processed = self.image_processor(image, return_tensors="pt")
        except TypeError as e:
            if "got an unexpected keyword argument 'return_tensors'" in str(e):
                processed = self.image_processor(image)
            else:
                raise

        if isinstance(processed, torch.Tensor):
            if processed.dim() == 3:
                return processed

        if 'pixel_values' not in processed:
            raise ValueError("The image processor must return 'pixel_values' in the processed output.")
        
        if processed['pixel_values'].dim() == 3:
            return processed['pixel_values']
        elif processed['pixel_values'].dim() == 4:
            return processed['pixel_values'].squeeze(0)
        
        return None


class CLIPBase(BaseModel, nn.Module):
    def __init__(self, text, num_classes, model=None, args=None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.prototype_text = text
        self.num_classes = num_classes
        self.model = model
        self.prototype = None
        self.setup_encoders()

    def initialize_prototypes(self):
        """Initializes text prototypes. Should be called at the end of subclass __init__."""
        if self.prototype is None:
            with torch.no_grad():
                self.prototype = self.encode_text(self.prototype_text)
    
    def setup_encoders(self):
        self.vision_model = self.model.vision_model
        self.text_model = self.model.text_model
        self.text_embed_dim = self.model.text_embed_dim
        self.vision_embed_dim = self.model.vision_embed_dim

    def forward(self, pixel_values, input_ids):
        return self.model.forward(input_ids=input_ids, pixel_values=pixel_values)

    def encode_image(self, images):
        return self.model.get_image_features(images)

    def encode_text(self, text):
        return self.model.get_text_features(text)

    def load_for_training(self, model_path):
        pass

    def load_from_pretrained(self, model_path, device, **kwargs):
        model_ckpt = torch.load(model_path)
        self.model.load_state_dict(model_ckpt)
        self.model.to(device)
    

class LPModel(CLIPBase):
    def __init__(self, text, num_classes, model=None, args=None):
        super().__init__(text, num_classes, model)
        self.num_classes = num_classes

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.head = torch.nn.Linear(self.vision_embed_dim, self.num_classes)


    def forward(self, images):
        with torch.no_grad():
            image_features = self.encode_image(images)
        return self.head(image_features)

    def get_parameters_info(self):
        tuned_parameters = []
        tuned_parameter_size = 0
        all_parameter_size = 0
        
        for n, p in self.named_parameters():
            all_parameter_size += maybe_zero_3(p, ignore_status=True).numel()
            if p.requires_grad:
                tuned_parameters.append(n)
                tuned_parameter_size += maybe_zero_3(p, ignore_status=True).numel()

        return all_parameter_size, tuned_parameter_size, tuned_parameters


class LoRALPModel(BaseModel, nn.Module):
    def __init__(self, args, model, num_classes, lora_config):
        super().__init__(args)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.vision_model = get_peft_model(self.model.vision_model, lora_config)
        
        self.head = nn.Linear(self.feat_dim, self.num_classes).to(self.device)
        

    def forward(self, images):
        image_features = self.encode_image(images)
        
        return self.head(image_features)


class LoRAClipLossModel(CLIPBase):
    """
    LoRA with CLIP Loss Model.
    Adds LoRA adapters to the CLIP image encoder and fine-tunes them using the
    standard CLIP contrastive loss. This method does not use a separate classification head.
    """
    def __init__(self, model_name_or_path, class_names, lora_config, device='cpu', args=None):
        super().__init__(model_name_or_path, class_names, device, args)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.vision_model = get_peft_model(self.model.vision_model, lora_config)
        
        self.model.logit_scale.requires_grad = True

        self.initialize_text_prototypes()
        

    def forward(self, pixel_values, input_ids):
        self.model.forward(pixel_values=pixel_values, input_ids=input_ids)