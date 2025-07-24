import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import (
    SiglipModel,
    SiglipProcessor,
    SiglipImageProcessor,
    SiglipTokenizer,
)
# Make sure these base classes are correctly imported from your project structure
from model.base import BaseModel
from model.lora_base import LoRALPModel
from model.clip_base import CLIPBase, ImageProcessorCallable, LPModel


class MedSigLIPForDiagnosis(CLIPBase):
    """
    Wrapper around `google/medsiglip-448` for zero-/few-shot medical
    image classification or retrieval.
    """
    def __init__(self, text, num_classes, args=None, *kargs, **kwargs):
        # Load MedSigLIP checkpoint
        model = SiglipModel.from_pretrained("google/medsiglip-448")

        # Optional: apply LoRA on vision encoder
        if args and getattr(args, "usage", None) == "medsiglip-img-lora":
            lora_cfg = LoraConfig(target_modules=["q_proj", "k_proj", "v_proj"])
            # Freeze all parameters before applying LoRA
            for _, p in model.named_parameters():
                p.requires_grad = False
            model.vision_model = get_peft_model(model.vision_model, lora_cfg)
        
        # Inherit from the correct SiglipBase
        super().__init__(
            text=text,
            num_classes=num_classes,
            model=model,
            args=args,
            **kwargs,
        )

        # Processor components
        processor = SiglipProcessor.from_pretrained("google/medsiglip-448")
        self.tokenizer: SiglipTokenizer = processor.tokenizer
        
        # The ImageProcessorCallable handles the output correctly without a custom transform
        self.image_processor = ImageProcessorCallable(processor.image_processor)
        self.image_processor_evaluation = self.image_processor

        # DO NOT add a new logit_scale. SiglipBase will use the one from the model.
        
        self.initialize_prototypes()

    @torch.no_grad()
    def encode_text(self, text):
        assert len(text) == self.num_classes
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        return self.model.get_text_features(**inputs)

    def encode_image(self, images):
        return self.model.get_image_features(images)


class MedSigLIPLPForDiagnosis(LPModel):
    def __init__(self, args, text, num_classes) -> None:
        super().__init__(text=text, num_classes=num_classes, model=SiglipModel.from_pretrained("google/medsiglip-448"), args=args)
        
        image_processor_hf = SiglipImageProcessor.from_pretrained("google/medsiglip-448")
        self.image_processor = ImageProcessorCallable(image_processor_hf)
        self.image_processor_evaluation = self.image_processor
    
    def setup_encoders(self):
        self.vision_model = self.model.vision_model
        self.text_model = self.model.text_model
        self.text_embed_dim = 1152
        self.vision_embed_dim = 1152


class MedSigLIPLoRALPForDiagnosis(LoRALPModel):
    def __init__(self, args, *kargs, **kwargs):
        model = SiglipModel.from_pretrained("google/medsiglip-448")
        vision_model = model.vision_model
        vision_model.feat_dim = getattr(vision_model.config, "hidden_size", 1024)

        lora_cfg = LoraConfig(target_modules=["q_proj", "k_proj", "v_proj"])
        
        super().__init__(
            args=args,
            lora_config=lora_cfg,
            encoder=vision_model,
            num_classes=kwargs["num_classes"],
        )

        img_proc_hf = SiglipImageProcessor.from_pretrained("google/medsiglip-448")
        self.image_processor = ImageProcessorCallable(image_processor=img_proc_hf)
        self.image_processor_evaluation = self.image_processor

    def extract_features(self, images):
        outputs = self.encoder(images)
        return outputs["pooler_output"]