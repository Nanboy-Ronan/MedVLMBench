import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel, CLIPProcessor, CLIPFeatureExtractor
from model.clip_base import CLIPBase, ImageProcessorCallable
from model.lp_base import LPModel
from model.lora_base import LoRALPModel


class PLIPForDiagnosis(CLIPBase):
    """
    Full-model fine-tuning or LoRA-tuning wrapper around the
    ViNID/PLIP CLIP checkpoint for zero-/few-shot diagnosis tasks.
    """
    def __init__(self, text, num_classes, args=None, *kargs, **kwargs):
        # Load base PLIP model
        model = CLIPModel.from_pretrained("vinid/plip")

        # Optional LoRA on the vision encoder (matching CLIPForDiagnosis logic)
        if args and getattr(args, "usage", None) == "plip-img-lora":
            lora_config = LoraConfig(target_modules=["k_proj", "q_proj", "v_proj"])
            for _, p in model.named_parameters():
                p.requires_grad = False
            model.vision_model = get_peft_model(model.vision_model, lora_config)
        breakpoint()
        super().__init__(text=text,
                         num_classes=num_classes,
                         model=model,
                         args=args,
                         **kwargs)

        # Shared tokenizer / image processor from CLIPProcessor
        processor = CLIPProcessor.from_pretrained("vinid/plip")
        self.tokenizer = processor.tokenizer

        transform_func = lambda out: torch.tensor(out["pixel_values"][0])
        self.image_processor = ImageProcessorCallable(
            processor.image_processor,
            transform_func=transform_func,
        )
        self.image_processor_evaluation = self.image_processor

        # PLIP keeps the same fixed log-temperature as OpenAI CLIP
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(100.0)))

        self.initialize_prototypes()

    # ------------------------------------------------------------------ #
    # Encoding helpers
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode_text(self, text):
        assert len(text) == self.num_classes
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        return self.model.get_text_features(**inputs)

    def encode_image(self, images):
        return self.model.get_image_features(images)


# ---------------------------------------------------------------------- #
# Linear-probe variant (frozen PLIP vision encoder + small classifier)
# ---------------------------------------------------------------------- #
class PLIPLPForDiagnosis(LPModel):
    def __init__(self, *args, **kwargs):
        model = CLIPModel.from_pretrained("vinid/plip")
        vision_model = model.vision_model
        vision_model.feat_dim = getattr(vision_model, "feat_dim", 768)

        super().__init__(encoder=vision_model, *args, **kwargs)

        image_processor_hf = CLIPFeatureExtractor.from_pretrained("vinid/plip")
        transform_func = lambda out: torch.tensor(out["pixel_values"][0])
        self.image_processor = ImageProcessorCallable(
            image_processor_hf,
            transform_func=transform_func,
        )
        self.image_processor_evaluation = self.image_processor

    def extract_features(self, images):
        # CLS token at index 0
        return self.encoder(images)["last_hidden_state"][:, 0, :]


# ---------------------------------------------------------------------- #
# LoRA-adapted linear-probe variant
# ---------------------------------------------------------------------- #
class PLIPLoRALPForDiagnosis(LoRALPModel):
    def __init__(self, args, *kargs, **kwargs):
        model = CLIPModel.from_pretrained("vinid/plip")
        vision_model = model.vision_model
        vision_model.feat_dim = getattr(vision_model, "feat_dim", 768)

        lora_config = LoraConfig(target_modules=["k_proj", "q_proj", "v_proj"])
        breakpoint()
        super().__init__(args=args,
                         lora_config=lora_config,
                         encoder=vision_model,
                         num_classes=kwargs["num_classes"])

        image_processor_hf = CLIPFeatureExtractor.from_pretrained("vinid/plip")
        transform_func = lambda out: torch.tensor(out["pixel_values"][0])
        self.image_processor = ImageProcessorCallable(
            image_processor_hf,
            transform_func=transform_func,
        )
        self.image_processor_evaluation = self.image_processor

    def extract_features(self, images):
        return self.encoder(images)["last_hidden_state"][:, 0, :]
