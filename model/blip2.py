import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from model.base import BaseModel
from model.chat import ChatMetaModel


class BLIP2(ChatMetaModel):
    def __init__(self, args=None):
        super().__init__(args)
        self.name = "BLIP2"
        self.model_name = "Salesforce/blip2-opt-2.7b"
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_name).to(self.args.device)
        self.image_processor = self.processor.image_processor
        self.tokenizer = self.processor.tokenizer

    def infer_vision_language(self, image, qs, image_size=None):
        """
        Generates answers based on input image and text prompt.
        :param image: The image tensor (preprocessed)
        :param qs: The input question/prompt as a string
        :param image_size: Optional parameter for image size
        :return: Generated text output
        """
        # Preprocess inputs
        inputs = self.processor(images=image, text=qs, return_tensors="pt", padding=True, truncation=True).to(self.args.device)

        # Generate response
        outputs = self.model.generate(**inputs, max_new_tokens=50)
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
