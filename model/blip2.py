import torch
from PIL import Image
from easydict import EasyDict as edict
from transformers import Blip2ForConditionalGeneration, Blip2Processor, Blip2Model, BatchEncoding, Blip2ForImageTextRetrieval
from transformers.tokenization_utils import AddedToken
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from peft import LoftQConfig, LoraConfig, get_peft_model

from model.base import BaseModel
from model.chat import ChatMetaModel
from model.lp_base import LPModel
from model.clip_base import CLIPBase


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
        self.model_type = "general"
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
        inputs = self.processor(images=image, text=qs, return_tensors="pt", padding=True, truncation=True).to(
            self.args.device
        )

        outputs = self.model.generate(**inputs, max_new_tokens=768)
        answer = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        return answer


class ImageProcessorCallable:
    def __init__(self, image_processor):
        """
        Wrapper around the Blip2Processor for image preprocessing.
        Converts input images to the format required by Blip2.
        """
        self.image_processor = image_processor

    def __call__(self, images):
        """
        Processes a batch of images and returns pixel values.
        
        Args:
            images (torch.Tensor): Batch of image tensors (C, H, W).
        
        Returns:
            torch.Tensor: Processed image tensors.
        """
        device = images.device
        image_batch_pil = [to_pil_image(img_tensor) for img_tensor in images]
        processed_images = [
            torch.tensor(self.image_processor(pil_image)["pixel_values"][0])
            for pil_image in image_batch_pil
        ]
        processed_images = torch.stack(processed_images).to(device)
        return processed_images


class BLIP2ForDiagnosis(CLIPBase):
    def __init__(self, text, num_classes, model_name="Salesforce/blip2-itm-vit-g"):
        """
        Wrapper around Blip2ForImageTextRetrieval to simplify usage.
        
        Args:
            text (list): List of text prototypes for each class.
            num_classes (int): Number of classes.
            model_name (str): Name of the pre-trained model.
        """
        model = Blip2ForImageTextRetrieval.from_pretrained(model_name)
        super().__init__(text=text, num_classes=num_classes, model=model)
        self.tokenizer = Blip2Processor.from_pretrained(model_name).tokenizer
        self.image_processor = Blip2Processor.from_pretrained(model_name).image_processor
        self.image_processor = ImageProcessorCallable(self.image_processor)
        self.image_processor_evaluation = self.image_processor
        self.num_classes = num_classes
        self.prototype = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    @torch.no_grad()
    def encode_text(self, text):
        assert len(text) == self.num_classes
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_outputs = self.model.text_model(**inputs)
        return F.normalize(text_outputs.last_hidden_state[:, 0, :], dim=-1)

    def forward(self, images):
        device = images.device
        self.model.to(device)
        outputs = self.model(input_ids=self.prototype["input_ids"].to(device), attention_mask=self.prototype["attention_mask"].to(device), pixel_values=images)
        image_features, text_features = outputs["image_embeds"], outputs["text_embeds"]
        logits = outputs["logits_per_image"]
        return logits


class BLIP2LoRAForDiagnosis(CLIPBase):
    def __init__(self, args, text, num_classes, model_name="Salesforce/blip2-itm-vit-g"):
        """
        Wrapper around Blip2ForImageTextRetrieval to simplify usage.
        
        Args:
            text (list): List of text prototypes for each class.
            num_classes (int): Number of classes.
            model_name (str): Name of the pre-trained model.
        """
        model = Blip2ForImageTextRetrieval.from_pretrained(model_name)

        if args.usage == "clip-img-lora":
            lora_config = LoraConfig(target_modules=["qkv"])
            for name, para in model.named_parameters():
                para.requires_grad = False
            model.vision_model = get_peft_model(model.vision_model, lora_config)
        else:
            raise NotImplementedError()

        super().__init__(text=text, num_classes=num_classes, model=model)
        self.tokenizer = Blip2Processor.from_pretrained(model_name).tokenizer
        self.image_processor = Blip2Processor.from_pretrained(model_name).image_processor
        self.image_processor = ImageProcessorCallable(self.image_processor)
        self.image_processor_evaluation = self.image_processor
        self.num_classes = num_classes
        self.prototype = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    @torch.no_grad()
    def encode_text(self, text):
        assert len(text) == self.num_classes
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_outputs = self.model.text_model(**inputs)
        return F.normalize(text_outputs.last_hidden_state[:, 0, :], dim=-1)

    def forward(self, images):
        device = images.device
        self.model.to(device)
        outputs = self.model(input_ids=self.prototype["input_ids"].to(device), attention_mask=self.prototype["attention_mask"].to(device), pixel_values=images)
        image_features, text_features = outputs["image_embeds"], outputs["text_embeds"]
        logits = outputs["logits_per_image"]
        return logits

class BLIP2LPForDiagnosis(LPModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.blip_config = BlipConfig()
        self.image_processor = BlipImageProcessor()
        self.model = BlipModel(self.blip_config)
        self.vision_model = self.model.vision_model
        self.vision_model.feat_dim = 768
        if "lp" in self.args.usage:
            from wrappers import LinearProbeWrapper
            self.model = LinearProbeWrapper(self.vision_model)
    
    def load_for_training(self, model_path):
        pass
        
    def load_from_pretrained(self, model_path, device, **kwargs):
        model_ckpt = torch.load(model_path)
        self.model.load_state_dict(model_ckpt)
        self.model.to(device)
    
    def forward(self, x):
        return self.model.head(self.model.encoder(x)["last_hidden_state"][:, 0, :])


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
