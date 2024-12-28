import torch
from PIL import Image
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from transformers import BlipProcessor, BlipConfig, BlipForConditionalGeneration, BlipForQuestionAnswering, BlipModel

from model.base import BaseModel
from model.chat import ChatMetaModel
from model.clip_base import CLIPModel


def visualize_tensor_image(tensor, unnormalize=True):
    """
    Visualizes a single image tensor.

    Args:
        tensor (torch.Tensor): Image tensor of shape [1, 3, H, W].
        unnormalize (bool): Whether to unnormalize the tensor.
    """
    if tensor.dim() != 4 or tensor.size(0) != 1 or tensor.size(1) != 3:
        raise ValueError("Input tensor must have shape [1, 3, H, W]")

    img = tensor.squeeze(0)

    if unnormalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean

    img = torch.clamp(img, 0, 1)

    img_np = img.permute(1, 2, 0).cpu().numpy()

    plt.imshow(img_np)
    plt.axis("off")
    plt.savefig("./demo.png")


class ImageProcessorCallable:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image):
        return self.image_processor(image)["pixel_values"]


class BLIPForQA(ChatMetaModel):
    def __init__(self, args=None):
        super().__init__(args)
        # if mode == "vqa":
        self.name = "BLIP"
        self.model_type = "general"
        self.model_name = "Salesforce/blip-vqa-base"
        # elif mode == "caption":
        #     self.model_name = "Salesforce/blip-image-captioning-base"
        # else:
        #     raise NotImplementedError()
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        # self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.args.device) # for captioning
        self.model = BlipForQuestionAnswering.from_pretrained(self.model_name).to(self.args.device)
        self.image_processor_callable = ImageProcessorCallable(self.processor.image_processor)
        self.tokenizer = self.processor.tokenizer


    def infer_vision_language(self, image, qs, image_size=None):
        text_inputs = self.tokenizer(qs, return_tensors="pt", padding=True, truncation=True)

        # inputs = self.processor(images=image, text=qs, return_tensors="pt")
        inputs = {
            "input_ids": text_inputs["input_ids"].to(self.args.device),
            "attention_mask": text_inputs["attention_mask"].to(self.args.device),
            "pixel_values": image.to(self.args.device),
        }

        outputs = self.model.generate(**inputs, max_length=768)
        answer = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        return answer


class BLIPForDiagnosis(CLIPModel):
    def __init__(self, backbone="ViT-B/32", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.blip_config = BlipConfig()
        self.model = BlipModel(self.blip_config)
        self.vision_model = self.model.vision_model
        self.vision_model.feat_dim = 768

    def forward_clip(self, images, text_features):
        sample = {"image": images, "text_input": None}
        image_features = self.model.extract_features(sample, mode="image").image_embeds_proj[:, 0]

        text_features = F.normalize(text_features, dim=-1)

        logits = (image_features @ text_features.T) / self.model.temp

        return logits

    def encode_text(self, text):
        sample = {"image": None, "text_input": text}

        text_features = self.model.extract_features(sample, mode="text").text_embeds_proj[:, 0, :]
        return text_features


    # def forward(self, images):
        # sample = {"image": images, "text_input": None}
        # return self.vision_model(images).image_embeds[:, 0, :]

    def load_for_training(self, model_name_or_path):
        if "lp" in self.args.usage:
            from wrappers import LinearProbeWrapper
            self.model = LinearProbeWrapper(self.vision_model)


# if __name__ == "__main__":
#     # blip_caption = BLIP(mode="caption")
#     blip_vqa = BLIP(args=edict(device="cuda"))

#     image_path = "/fast/rjin02/DataSets/CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg"
#     # image_path = "/fast/rjin02/DataSets/COCO/2014/val2014/COCO_val2014_000000000042.jpg"
#     image_path = "/fast/rjin02/DataSets/mimic_cxr_all/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"

#     # caption = blip_caption.caption(image_path)
#     # print("Generated Caption:", caption)

#     image = Image.open(image_path).convert("RGB")
#     image = blip_vqa.image_processor_callable(image)[0]
#     image = torch.tensor(image).unsqueeze(0)

#     question = "the gender of this patient is?"
#     answer = blip_vqa.infer_vision_language(image, question, image_size=None)
#     print("VQA Answer:", answer)
