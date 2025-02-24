import torch
import numpy as np
import torch.nn.functional as F
from model.lp_base import LPModel
from model.lora_base import LoRALPModel
from model.clip_base import CLIPBase
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPProcessor, CLIPModel, CLIPProcessor

class ImageProcessorLPCallable:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image):
        device = image.device
        image_batch_pil = [to_pil_image(img_tensor) for img_tensor in image]
        image = [torch.tensor(self.image_processor(pil_image)["pixel_values"][0]) for pil_image in image_batch_pil]
        image = torch.stack(image).to(device)
        return image 

class PLIPForDiagnosis(CLIPBase):
    def __init__(self, text, num_classes, *args, **kwargs) -> None:
        model = CLIPModel.from_pretrained("vinid/plip")
        super().__init__(text=text, num_classes=num_classes, model=model)
        self.processor = CLIPProcessor.from_pretrained("vinid/plip")
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor
        self.image_processor = ImageProcessorLPCallable(self.image_processor)
        self.image_processor_evaluation = self.image_processor
        self.prototype = self.encode_text(self.prototype)
    
    @torch.no_grad()
    def encode_text(self, text):
        assert len(text) == self.num_classes
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)

        text_outputs = self.model.get_text_features(**inputs)
        return text_outputs
    
    def forward(self, images):    
        image_outputs = self.model.get_image_features(images)

        image_features = F.normalize(image_outputs, dim=-1)
        text_features = F.normalize(self.prototype, dim=-1).to(images.device)

        logits = 100.0 * image_features @ text_features.T
        
        return logits