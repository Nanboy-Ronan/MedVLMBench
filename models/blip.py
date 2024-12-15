import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from models.base import BaseModel


class BLIP(ChatMetaModel):
    def __init__(self, mode=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        if mode == "vqa":
            self.model_name = "Salesforce/blip-vqa-base"
        elif mode == "caption":
            self.model_name = "Salesforce/blip-image-captioning-base"
        else:
            raise NotImplementedError()

        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

    def caption(self, image_path):
        raw_image = Image.open(image_path).convert("RGB")
        inputs = self.processor(raw_image, return_tensors="pt").to(self.device)

        outputs = self.model.generate(**inputs, max_length=50)
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption

    def vqa(self, image_path, question):
        raw_image = Image.open(image_path).convert("RGB")
        inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device)

        outputs = self.model.generate(**inputs, max_length=50)
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        return answer


if __name__ == "__main__":
    blip_caption = BLIP(mode="caption")
    blip_vqa = BLIP(mode="vqa")

    image_path = "/fast/rjin02/DataSets/CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg"

    caption = blip_caption.caption(image_path)
    print("Generated Caption:", caption)

    question = "What is in the image?"
    answer = blip_vqa.vqa(image_path, question)
    print("VQA Answer:", answer)
