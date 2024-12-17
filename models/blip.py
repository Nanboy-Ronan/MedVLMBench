import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from models.base import BaseModel
from models.chat import ChatMetaModel


class BLIP(ChatMetaModel):
    def __init__(self, args=None):
        super().__init__(args)
        # if mode == "vqa":
        self.model_name = "Salesforce/blip-vqa-base"
        # elif mode == "caption":
        #     self.model_name = "Salesforce/blip-image-captioning-base"
        # else:
        #     raise NotImplementedError()
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.args.device)
        self.image_processor = self.processor.image_processor
        self.tokenizer = self.processor.tokenizer

    # def caption(self, image_path):
    #     raw_image = Image.open(image_path).convert("RGB")
    #     inputs = self.processor(raw_image, return_tensors="pt").to(self.device)

    #     outputs = self.model.generate(**inputs, max_length=50)
    #     caption = self.processor.decode(outputs[0], skip_special_tokens=True)
    #     return caption

    # def infer_vision_language(self, image_path, question):
    def infer_vision_language(self, image, qs, image_size):
        # Tokenize the question
        text_inputs = self.tokenizer(qs, return_tensors="pt", padding=True, truncation=True)

        inputs = {
            "input_ids": text_inputs["input_ids"].to(self.args.device),
            "attention_mask": text_inputs["attention_mask"].to(self.args.device),
            "pixel_values": image.to(self.args.device),
        }

        outputs = self.model.generate(**inputs, max_length=50)
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        return answer


if __name__ == "__main__":
    # blip_caption = BLIP(mode="caption")
    blip_vqa = BLIP(mode="vqa")

    image_path = "/fast/rjin02/DataSets/CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg"

    # caption = blip_caption.caption(image_path)
    # print("Generated Caption:", caption)

    question = "What is in the image?"
    answer = blip_vqa.infer_vision_language(image_path, question)
    print("VQA Answer:", answer)
