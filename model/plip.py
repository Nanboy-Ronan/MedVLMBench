import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from model.clip_base import CLIPBase, ImageProcessorCallable

class PLIPForDiagnosis(CLIPBase):
    def __init__(self, text, num_classes, *args, **kwargs) -> None:
        model = CLIPModel.from_pretrained("vinid/plip")
        processor = CLIPProcessor.from_pretrained("vinid/plip")
        super().__init__(text=text, num_classes=num_classes, model=model, *args, **kwargs)

        self.tokenizer = processor.tokenizer
        
        # Define the function to extract the tensor from the processor's output
        transform_func = lambda p: torch.tensor(p['pixel_values'][0])
        self.image_processor = ImageProcessorCallable(processor.image_processor, transform_func=transform_func)
        self.image_processor_evaluation = self.image_processor
        
        # PLIP uses a fixed logit scale of 100.0
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(100.0)))
        
        self.initialize_prototypes()
    
    @torch.no_grad()
    def encode_text(self, text):
        assert len(text) == self.num_classes
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
        return self.model.get_text_features(**inputs)
    
    def encode_image(self, images):
        return self.model.get_image_features(images)