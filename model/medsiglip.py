import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import (
    SiglipModel,
    SiglipProcessor,
    SiglipImageProcessor,
    SiglipTokenizer,
)
from model.clip_base import CLIPBase, ImageProcessorCallable
from model.lp_base import LPModel
from model.lora_base import LoRALPModel


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
            for _, p in model.named_parameters():
                p.requires_grad = False
            model.vision_model = get_peft_model(model.vision_model, lora_cfg)
        breakpoint()
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

        # MedSigLIP expects 448×448 inputs; the image processor handles this
        transform_func = lambda out: torch.tensor(out["pixel_values"][0])
        self.image_processor = ImageProcessorCallable(
            processor.image_processor,
            transform_func=transform_func,
        )
        self.image_processor_evaluation = self.image_processor

        # Keep a learnable log-temperature like CLIP
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(100.0)))

        self.initialize_prototypes()

    # ----------------------------- Encoders ----------------------------- #
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
    def __init__(self, *args, **kwargs):
        model = SiglipModel.from_pretrained("google/medsiglip-448")
        vision_model = model.vision_model
        # Hidden size of the ViT encoder
        vision_model.feat_dim = getattr(vision_model.config, "hidden_size", 1024)

        super().__init__(encoder=vision_model, *args, **kwargs)

        img_proc_hf = SiglipImageProcessor.from_pretrained("google/medsiglip-448")
        transform_func = lambda out: torch.tensor(out["pixel_values"][0])
        self.image_processor = ImageProcessorCallable(
            img_proc_hf,
            transform_func=transform_func,
        )
        self.image_processor_evaluation = self.image_processor

    def extract_features(self, images):
        return self.encoder(images)["last_hidden_state"].mean(dim=1)


class MedSigLIPLoRALPForDiagnosis(LoRALPModel):
    def __init__(self, args, *kargs, **kwargs):
        model = SiglipModel.from_pretrained("google/medsiglip-448")
        vision_model = model.vision_model
        vision_model.feat_dim = getattr(vision_model.config, "hidden_size", 1024)

        lora_cfg = LoraConfig(target_modules=["q_proj", "k_proj", "v_proj"])
        breakpoint()
        super().__init__(
            args=args,
            lora_config=lora_cfg,
            encoder=vision_model,
            num_classes=kwargs["num_classes"],
        )

        img_proc_hf = SiglipImageProcessor.from_pretrained("google/medsiglip-448")
        transform_func = lambda out: torch.tensor(out["pixel_values"][0])
        self.image_processor = ImageProcessorCallable(
            img_proc_hf,
            transform_func=transform_func,
        )
        self.image_processor_evaluation = self.image_processor

    def extract_features(self, images):
        return self.encoder(images)["last_hidden_state"].mean(dim=1)