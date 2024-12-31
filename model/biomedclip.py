import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from open_clip import create_model_from_pretrained, get_tokenizer
from model.clip_base import CLIPModel
from model.lp_base import LPModel
from transformers import BlipImageProcessor


class BiomedCLIP(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model, _ = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )
        self.feat_dim = 512

    def forward_clip(self, images, text_features):
        image_features = self.model.encode_image(images, normalize=True)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = self.model.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.t()

        return logits

    def encode_text(self, text):
        return self.model.encode_text(text.to(next(self.model.parameters()).device), normalize=False)

    def forward(self, images):
        return self.model.visual(images)

    def from_pretrained(self, path):
        pass

class ImageProcessorLPCallable:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image):
        image_batch_pil = [to_pil_image(img_tensor) for img_tensor in image]
        image = [model.image_processor(pil_image) for pil_image in image_batch_pil]
        image = torch.stack(image)
        return image 

class BioMedCLIPLPForDiagnosis(LPModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model, _ = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )  
        # TODO: different normalization for different dataset
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        normalize = transforms.Normalize(mean=mean, std=std)
        self.image_processor = transform = transforms.Compose(
                [
                    transforms.Resize(
                        224, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        self.image_processor_evaluation = ImageProcessorLPCallable(self.image_processor)
        self.vision_model = self.model.visual
        self.vision_model.feat_dim = 512

        if "lp" in self.args.usage:
            from wrappers import LinearProbeWrapper
            self.model = LinearProbeWrapper(self.vision_model)
    
    def load_for_training(self, model_path):
        pass
        
    def load_from_pretrained(self, model_path, device, **kwargs):
        model_ckpt = torch.load(model_path)
        self.model.load_state_dict(model_ckpt)
        self.model.to(device)
    
    def forward(self, x):
        return self.model.head(self.model.encoder(x))