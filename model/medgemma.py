import os
import shutil
import warnings
import torch
from torchvision.transforms.functional import to_pil_image
import transformers
from transformers import AutoProcessor, AutoModelForImageTextToText

from model.chat import ChatMetaModel


class MedGemma(ChatMetaModel):
    def __init__(self, args):
        super().__init__(args)

        self.name = "MedGemma"
        self.model_type = "medical"

    def load_from_pretrained(self, model_path, **kwargs):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def infer_vision_language(self, image, qs, image_size=None):
        image = to_pil_image(image)

        # prepare messages
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert on understanding medical images."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": qs},
                    {"type": "image", "image": image},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)

        print(decoded)

        return decoded.strip()
