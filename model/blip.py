import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from transformers import AutoTokenizer
from transformers import BlipProcessor, BlipImageProcessor, BlipConfig, BlipForConditionalGeneration, BlipForQuestionAnswering, BlipModel
from peft import LoraConfig, get_peft_model

from model.base import BaseModel
from model.chat import ChatMetaModel
from model.clip_base import CLIPBase
from model.lp_base import LPModel
from model.lora_base import LoRALPModel


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


class BLIPForDiagnosis(CLIPBase):
    def __init__(self, text, num_classes, *args, **kwargs) -> None:
        blip_config = BlipConfig()
        model = BlipModel(blip_config)
        super().__init__(text=text, num_classes=num_classes, model=model)
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-vqa-base")
        self.image_processor = BlipImageProcessor()
        self.image_processor = ImageProcessorLPCallable(self.image_processor)
        self.image_processor_evaluation = self.image_processor
        self.prototype = self.encode_text(self.prototype)
    
    @torch.no_grad()
    def encode_text(self, text):
        assert len(text) == self.num_classes
        text_inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        text_outputs = self.model.text_model(input_ids=text_inputs.input_ids, attention_mask=text_inputs.attention_mask)
        return F.normalize(text_outputs.last_hidden_state[:, 0, :], dim=-1)
    
    def forward(self, images):    
        image_features = self.model.vision_model(images)
        image_features = F.normalize(image_features.last_hidden_state[:, 0, :], dim=-1)  # Normalize the image features

        text_features = self.prototype.to(images.device)

        logits = 100.0 * image_features @ text_features.T

        return logits
    

class BLIPLoRAForDiagnosis(CLIPBase):
    def __init__(self, args, text, num_classes) -> None:
        blip_config = BlipConfig()
        model = BlipModel(blip_config)

        if args.usage == "clip-img-lora":
            lora_config = LoraConfig(target_modules=["qkv"])
            for name, para in model.named_parameters():
                para.requires_grad = False
            model.vision_model = get_peft_model(model.vision_model, lora_config)
        else:
            raise NotImplementedError()
        
        super().__init__(text=text, num_classes=num_classes, model=model)
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-vqa-base")
        self.image_processor = BlipImageProcessor()
        self.image_processor = ImageProcessorLPCallable(self.image_processor)
        self.image_processor_evaluation = self.image_processor
        self.prototype = self.encode_text(self.prototype).to(args.device)
    
    @torch.no_grad()
    def encode_text(self, text):
        assert len(text) == self.num_classes
        text_inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        text_outputs = self.model.text_model(input_ids=text_inputs.input_ids, attention_mask=text_inputs.attention_mask)
        return F.normalize(text_outputs.last_hidden_state[:, 0, :], dim=-1)
    
    def forward(self, images):
        image_features = self.model.vision_model(images)
        image_features = F.normalize(image_features.last_hidden_state[:, 0, :], dim=-1)  # Normalize the image features

        text_features = self.prototype.to(images.device)

        logits = 100.0 * image_features @ text_features.T

        return logits


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


class ImageProcessorLPCallable:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image):
        device = image.device
        return torch.tensor(self.image_processor(image)["pixel_values"]).to(device)


class BLIPLPForDiagnosis(LPModel):
    def __init__(self, *args, **kwargs) -> None: # We choose this implemention as the generative model and CLIP-based model are initialized differently.
        blip_config = BlipConfig()
        model = BlipModel(blip_config)
        vision_model = model.vision_model
        vision_model.feat_dim = 768

        super().__init__(encoder=vision_model, *args, **kwargs)
        self.image_processor = BlipImageProcessor()
        self.image_processor = ImageProcessorLPCallable(self.image_processor)
        self.image_processor_evaluation = self.image_processor
    
    def forward(self, x):
        return self.model.head(self.model.encoder(x)["last_hidden_state"][:, 0, :])


class BLIPLoRALPForDiagnosis(LoRALPModel):
    def __init__(self, *args, **kwargs) -> None:
        # TODO: refactor LP to be the following implementation, where lp is collectively added to base model. discard wrapper.
        self.blip_config = BlipConfig()
        model = BlipModel(self.blip_config)
        vision_model = model.vision_model
        vision_model.feat_dim = 768
        lora_config = LoraConfig(target_modules=["qkv"])
        super().__init__(args=args, lora_config=lora_config, encoder=vision_model, num_classes=kwargs['num_classes'])
        
        self.image_processor = BlipImageProcessor()
        self.image_processor_evaluation = ImageProcessorLPCallable(self.image_processor)
    
    def forward(self, x):
        return self.model.head(self.model.encoder(x)["last_hidden_state"][:, 0, :])

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
