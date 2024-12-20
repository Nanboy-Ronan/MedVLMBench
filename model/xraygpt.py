import torch
from PIL import Image
from easydict import EasyDict as edict
from model.release.xraygpt.models.mini_gpt4 import MiniGPT4

from model.base import BaseModel
from model.chat import ChatMetaModel


class XrayGPT(ChatMetaModel):
    def __init__(self, args=None):
        super().__init__(args)
        self.name = "XrayGPT-mini"
        self.model_type = "medical"
        # self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = MiniGPT4(
            vit_model="eva_clip_g",
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            freeze_qformer=True,
            num_query_token=32,
            llama_model="./Vicuna_Radiology_fp16/",
            prompt_path="prompts/alignment.txt",
            prompt_template="###Patient: {} ###Doctor: ",
            max_txt_len=160,
            low_resource=True,
            end_sym="###",
        )

    def infer_vision_language(self, image, qs, image_size=None):
        """
        Generates answers based on input image and text prompt.
        :param image: The image tensor (preprocessed)
        :param qs: The input question/prompt as a string
        :param image_size: Optional parameter for image size
        :return: Generated text output
        """
        qs = "Question: {} Answer:".format(qs)
        inputs = self.processor(images=image, text=qs, return_tensors="pt", padding=True, truncation=True).to(
            self.args.device
        )

        outputs = self.model.generate(**inputs, max_new_tokens=768)
        answer = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        return answer


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(
        device
    )

    image_path = "/fast/rjin02/DataSets/CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg"
    # image_path = "/fast/rjin02/DataSets/COCO/2014/val2014/COCO_val2014_000000000042.jpg"

    image = Image.open(image_path).convert("RGB")
    # prompt = "Question: how many cats are there? Answer:"
    prompt = "Question: What's in the image? Answer:"
    breakpoint()
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

    image = processor.image_processor(image)["pixel_values"]

    tokenizer_args = {
        "add_special_tokens": True,
        "padding": False,
        "stride": 0,
        "return_overflowing_tokens": False,
        "return_special_tokens_mask": False,
        "return_offsets_mapping": False,
        "return_token_type_ids": False,
        "return_length": False,
        "verbose": True,
    }
    text_inputs = processor.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {
        "input_ids": text_inputs["input_ids"].to(device),
        "attention_mask": text_inputs["attention_mask"].to(device),
        "pixel_values": torch.tensor(image).unsqueeze(0).to(device),
    }

    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(generated_text)
