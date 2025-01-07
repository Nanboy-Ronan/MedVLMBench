import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from transformers import CLIPModel
from transformers import CLIPProcessor, CLIPFeatureExtractor
from model.lp_base import LPModel
from model.lora_base import LoRALPModel
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


class CLIPLPForDiagnosis(LPModel):
    def __init__(self, backbone="ViT-B/32", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model, processor = clip.load(backbone, device="cpu")
        self.image_processor = processor
        self.vision_model = self.model.visual
        self.vision_model.feat_dim = 512
        self.image_processor_evaluation = ImageProcessorLPCallable(self.image_processor)
        if "lp" in self.args.usage:
            from wrappers import LinearProbeWrapper
            self.model = LinearProbeWrapper(self.vision_model, self.num_classes)


    def forward(self, images):
        return self.model.head(self.model.encoder(images))


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