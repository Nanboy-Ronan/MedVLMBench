import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from model.base import BaseModel
from model.chat import ChatMetaModel

class ImageProcessorCallable:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image):
        # TODO: check for batch > 1
        return self.image_processor(image)["pixel_values"][0]


class BLIP2(ChatMetaModel):
    def __init__(self, args=None):
        super().__init__(args)
        self.name = "BLIP2-2.7b"
        self.model_name = "Salesforce/blip2-opt-2.7b"
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_name).to(self.args.device)
        # self.image_processor_callable = ImageProcessorCallable(self.processor.image_processor)
        self.tokenizer = self.processor.tokenizer

    def infer_vision_language(self, image, qs, image_size=None):
        """
        Generates answers based on input image and text prompt.
        :param image: The image tensor (preprocessed)
        :param qs: The input question/prompt as a string
        :param image_size: Optional parameter for image size
        :return: Generated text output
        """
        inputs = self.processor(images=image, text=qs, return_tensors="pt", padding=True, truncation=True).to(self.args.device)

        text_inputs = self.tokenizer(qs, return_tensors="pt", padding=True, truncation=True)
        breakpoint()
        # inputs = {
        #     "input_ids": text_inputs["input_ids"].to(self.args.device),
        #     "attention_mask": text_inputs["attention_mask"].to(self.args.device),
        #     "pixel_values": image.to(self.args.device),
        # }
        outputs = self.model.generate(**inputs, max_length=50)
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        print(answer)
        return answer
