import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel, CLIPProcessor, CLIPFeatureExtractor, CLIPTokenizer
from model.clip_base import CLIPBase, ImageProcessorCallable
from model.lp_base import LPModel
from model.lora_base import LoRALPModel


class CLIPForDiagnosis(CLIPBase):
    def __init__(self, text, num_classes, args=None, *kargs, **kwargs) -> None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        if args and args.usage == "clip-img-lora":
            lora_config = LoraConfig(target_modules=["k_proj", "v_proj", "q_proj"])
            for name, para in model.named_parameters():
                para.requires_grad = False
            model.vision_model = get_peft_model(model.vision_model, lora_config)
        
        super().__init__(text=text, num_classes=num_classes, model=model, args=args, **kwargs)
        
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        image_processor_hf = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        
        transform_func = lambda p: torch.tensor(p['pixel_values'][0])
        self.image_processor = ImageProcessorCallable(image_processor_hf, transform_func=transform_func)
        self.image_processor_evaluation = self.image_processor

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


class CLIPLPForDiagnosis(LPModel):
    def __init__(self, *args, **kwargs) -> None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        vision_model = model.vision_model
        vision_model.feat_dim = 768
        super().__init__(encoder=vision_model, *args, **kwargs)
        
        image_processor_hf = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        transform_func = lambda p: torch.tensor(p['pixel_values'][0])
        self.image_processor = ImageProcessorCallable(image_processor_hf, transform_func=transform_func)
        self.image_processor_evaluation = self.image_processor
    
    def extract_features(self, images):
        return self.encoder(images)["last_hidden_state"][:, 0, :]


class CLIPLoRALPForDiagnosis(LoRALPModel):
    def __init__(self, args, *kargs, **kwargs) -> None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        vision_model = model.vision_model
        vision_model.feat_dim = 768
        lora_config = LoraConfig(target_modules=["k_proj", "v_proj", "q_proj"])
        super().__init__(args=args, lora_config=lora_config, encoder=vision_model, num_classes=kwargs['num_classes'])

        image_processor_hf = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        transform_func = lambda p: torch.tensor(p['pixel_values'][0])
        self.image_processor = ImageProcessorCallable(image_processor_hf, transform_func=transform_func)
        self.image_processor_evaluation = self.image_processor
    
    def extract_features(self, images):
        return self.encoder(images)["last_hidden_state"][:, 0, :]