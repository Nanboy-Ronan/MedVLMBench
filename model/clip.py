import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.lp_base import LPModel
from torchvision.transforms.functional import to_pil_image


class ImageProcessorLPCallable:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image):
        image_batch_pil = [to_pil_image(img_tensor) for img_tensor in image]
        image = [self.image_processor(pil_image) for pil_image in image_batch_pil]
        image = torch.stack(image)
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

    def load_for_training(self, path):
        pass
    
    def load_from_pretrained(self, model_path, device, **kwargs):
        model_ckpt = torch.load(model_path)
        self.model.load_state_dict(model_ckpt)
        self.model.to(device)