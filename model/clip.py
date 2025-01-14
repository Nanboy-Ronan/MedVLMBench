import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from transformers import CLIPModel
from transformers import CLIPProcessor, CLIPFeatureExtractor, CLIPTokenizer
from model.lp_base import LPModel
from model.lora_base import LoRALPModel
from model.clip_base import CLIPBase
from torchvision.transforms.functional import to_pil_image
from peft import LoftQConfig, LoraConfig, get_peft_model


class ImageProcessorLPCallable:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image):
        device = image.device
        image_batch_pil = [to_pil_image(img_tensor) for img_tensor in image]
        image = [torch.tensor(self.image_processor(pil_image)["pixel_values"][0]) for pil_image in image_batch_pil]
        image = torch.stack(image).to(device)
        return image 


class CLIPForDiagnosis(CLIPBase):
    def __init__(self, text, num_classes, *args, **kwargs) -> None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        super().__init__(text=text, num_classes=num_classes, model=model)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.image_processor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
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


class CLIPLoRAForDiagnosis(CLIPBase):
    def __init__(self, text, num_classes, *args, **kwargs) -> None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        super().__init__(text=text, num_classes=num_classes, model=model)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.image_processor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
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


class CLIPLPForDiagnosis(LPModel):
    def __init__(self, backbone="ViT-B/32", *args, **kwargs) -> None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        vision_model = model.vision_model
        vision_model.feat_dim = 768
        super().__init__(encoder=vision_model, *args, **kwargs)
        self.image_processor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_processor = ImageProcessorLPCallable(self.image_processor)
        self.image_processor_evaluation = self.image_processor
    
    def forward(self, images):
        return self.head(self.encoder(images)["last_hidden_state"][:, 0, :])


class CLIPLoRALPForDiagnosis(LoRALPModel):
    def __init__(self, backbone="ViT-B/32", *args, **kwargs) -> None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        vision_model = model.vision_model
        vision_model.feat_dim = 768
        lora_config = LoraConfig(target_modules=["k_proj", "v_proj", "q_proj"])
        super().__init__(args=args, lora_config=lora_config, encoder=vision_model, num_classes=kwargs['num_classes'])

        self.image_processor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_processor = ImageProcessorLPCallable(self.image_processor)
        self.image_processor_evaluation = self.image_processor
    
    def forward(self, images):
        return self.head(self.encoder(images)["last_hidden_state"][:, 0, :])