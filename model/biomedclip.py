import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from open_clip import create_model_from_pretrained, get_tokenizer
from model.clip_base import CLIPBase
from model.lp_base import LPModel
from model.lora_base import LoRALPModel
from peft import LoftQConfig, LoraConfig, get_peft_model


class ImageProcessorLPCallable:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image):
        device = image.device
        image_batch_pil = [to_pil_image(img_tensor) for img_tensor in image]
        image = [self.image_processor(pil_image) for pil_image in image_batch_pil]
        image = torch.stack(image).to(device)
        return image 


class BiomedCLIPForDiagnosis(CLIPBase):
    def __init__(self, text, num_classes, *args, **kwargs) -> None:
        model, processor = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )

        super().__init__(text=text, num_classes=num_classes, model=model)
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.image_processor = processor
        self.image_processor = ImageProcessorLPCallable(self.image_processor)
        self.image_processor_evaluation = self.image_processor
        self.prototype = self.encode_text(self.prototype)

    @torch.no_grad()
    def encode_text(self, text):
        inputs = self.tokenizer(text, context_length=256)
        return self.model.encode_text(inputs, normalize=False).to(next(self.model.parameters()).device)
    
    def forward(self, images):    
        image_outputs = self.model.encode_image(images)

        image_features = F.normalize(image_outputs, dim=-1)
        text_features = F.normalize(self.prototype, dim=-1).to(images.device)

        logits = 100.0 * image_features @ text_features.T
        
        return logits


class BiomedCLIPLoRAForDiagnosis(CLIPBase):
    def __init__(self, args, text, num_classes, **kwargs) -> None:
        model, processor = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )

        if args.usage == "clip-img-lora":
            lora_config = LoraConfig(target_modules=["qkv"])
            for name, para in model.named_parameters():
                para.requires_grad = False
            model.visual = get_peft_model(model.visual, lora_config)
        else:
            raise NotImplementedError()


        super().__init__(text=text, num_classes=num_classes, model=model)
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.image_processor = processor
        self.image_processor = ImageProcessorLPCallable(self.image_processor)
        self.image_processor_evaluation = self.image_processor
        self.prototype = self.encode_text(self.prototype)

    @torch.no_grad()
    def encode_text(self, text):
        inputs = self.tokenizer(text, context_length=256)
        return self.model.encode_text(inputs, normalize=False).to(next(self.model.parameters()).device)
    
    def forward(self, images):    
        image_outputs = self.model.encode_image(images)

        image_features = F.normalize(image_outputs, dim=-1)
        text_features = F.normalize(self.prototype, dim=-1).to(images.device)

        logits = 100.0 * image_features @ text_features.T
        
        return logits


class BioMedCLIPLPForDiagnosis(LPModel):
    def __init__(self, *args, **kwargs) -> None:
        model, _ = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )  
        vision_model = model.visual
        vision_model.feat_dim = 512
        super().__init__(encoder=vision_model, *args, **kwargs)
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
        self.image_processor = ImageProcessorLPCallable(self.image_processor)
        self.image_processor_evaluation = self.image_processor



class BioMedCLIPLoRALPForDiagnosis(LoRALPModel):
    def __init__(self, *args, **kwargs) -> None:
        model, _ = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )  
        vision_model = model.visual
        vision_model.feat_dim = 512
        lora_config = LoraConfig(target_modules=["qkv"])
        super().__init__(args=args, lora_config=lora_config, encoder=vision_model, num_classes=kwargs['num_classes'])
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
        self.image_processor = ImageProcessorLPCallable(self.image_processor)
        self.image_processor_evaluation = self.image_processor
