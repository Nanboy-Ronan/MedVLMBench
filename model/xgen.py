import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria

from model.base import BaseModel
from model.chat import ChatMetaModel

class ImageProcessorCallable:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image):
        # TODO: check for batch > 1
        breakpoint()
        self.image_processor(image)
        return self.image_processor(image)["pixel_values"][0]


class XGenMiniV1(ChatMetaModel):
    def __init__(self, args=None):
        super().__init__(args)
        self.name = "XGenMiniV1"
        self.model_name = "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"
        
        self.hf_path = "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"
        self.model = AutoModelForVision2Seq.from_pretrained(self.hf_path, trust_remote_code=True).to(self.args.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_path, trust_remote_code=True, use_fast=False, legacy=False)
        self.img_processor = AutoImageProcessor.from_pretrained(self.hf_path, trust_remote_code=True)
        self.image_processor_callable = ImageProcessorCallable(self.img_processor)
        self.tokenizer = self.model.update_special_tokens(self.tokenizer)

    def infer_vision_language(self, image, qs, image_size=None):
        """
        Generates an answer based on input image and text prompt.
        
        :param image: The image in PIL format or a tensor convertible to PIL.
        :param qs: The input question/prompt as a string.
        :param image_size: Optional parameter for image size if resizing is needed.
        
        :return: Generated text output.
        """
        # Preprocess the image
        if not isinstance(image, Image.Image):
            pass
        # breakpoint()
        # pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"].to(self.args.device)
        
        text_inputs = self.tokenizer(qs, return_tensors="pt")

        outputs = self.model.generate(
            pixel_values=image.to(self.args.device),
            input_ids=text_inputs["input_ids"].to(self.args.device), 
            attention_mask=text_inputs["attention_mask"].to(self.args.device), 
            max_new_tokens=50
        )
        
        # Decode the model output to text
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
