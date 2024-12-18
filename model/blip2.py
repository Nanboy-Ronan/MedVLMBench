import torch
from PIL import Image
from easydict import EasyDict as edict
from transformers import Blip2ForConditionalGeneration, Blip2Processor, Blip2Model, BatchEncoding
from transformers.tokenization_utils import AddedToken

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
        # self.model = Blip2Model.from_pretrained(self.model_name).to(self.args.device)
        # self.image_processor_callable = ImageProcessorCallable(self.processor.image_processor)
        # self.tokenizer = self.processor.tokenizer

    def infer_vision_language(self, image, qs, image_size=None):
        """
        Generates answers based on input image and text prompt.
        :param image: The image tensor (preprocessed)
        :param qs: The input question/prompt as a string
        :param image_size: Optional parameter for image size
        :return: Generated text output
        """
        qs = "Question: {} Answer:".format(qs)
        inputs = self.processor(images=image, text=qs, return_tensors="pt", padding=True, truncation=True).to(self.args.device)
        
        # text_inputs = self.tokenizer(qs, return_tensors="pt", padding=True, truncation=True)
        # inputs = {
        #     "input_ids": text_inputs["input_ids"].to(self.args.device),
        #     "attention_mask": text_inputs["attention_mask"].to(self.args.device),
        #     "pixel_values": image.to(self.args.device),
        # }
        # breakpoint()
        outputs = self.model.generate(**inputs, max_new_tokens=768)
        answer = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        return answer

if __name__ == "__main__":
    # blip_vqa = BLIP2(args=edict(device="cuda"))

    # image_path = "/fast/rjin02/DataSets/CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg"
    # # image_path = "/fast/rjin02/DataSets/COCO/2014/val2014/COCO_val2014_000000000042.jpg"

    # image = Image.open(image_path).convert("RGB")

    # question = "What is in the image?"
    # answer = blip_vqa.infer_vision_language(image, question, image_size=None)
    # print("VQA Answer:", answer)


    # Official example
    from PIL import Image
    import requests
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    ).to(device)

    image_path = "/fast/rjin02/DataSets/CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg"
    # image_path = "/fast/rjin02/DataSets/COCO/2014/val2014/COCO_val2014_000000000042.jpg"

    image = Image.open(image_path).convert("RGB")
    # prompt = "Question: how many cats are there? Answer:"
    prompt = "Question: What's in the image? Answer:"
    breakpoint()
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

    image = processor.image_processor(image)["pixel_values"]

    tokenizer_args = {'add_special_tokens': True, 'padding': False, 'stride': 0, 'return_overflowing_tokens': False, 'return_special_tokens_mask': False, 'return_offsets_mapping': False, 'return_token_type_ids': False, 'return_length': False, 'verbose': True}
    text_inputs = processor.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {
        "input_ids": text_inputs["input_ids"].to(device),
        "attention_mask": text_inputs["attention_mask"].to(device),
        "pixel_values": torch.tensor(image).unsqueeze(0).to(device),
    }
    
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(generated_text)