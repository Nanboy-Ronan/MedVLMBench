import torch
import torch.nn as nn
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from lavis.models import load_model_and_preprocess

from base import BaseModel

class BLIP:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Load the BLIP model and preprocessors for image captioning
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name="blip_caption", model_type="base", is_eval=True, device=self.device
        )

    def caption(self, image_path):
        # Load and preprocess the image
        raw_image = Image.open(image_path).convert("RGB")
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        # Generate caption
        caption = self.model.generate({"image": image})
        return caption[0]

    def vqa(self, image_path, question):
        # Load and preprocess the image
        raw_image = Image.open(image_path).convert("RGB")
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        # Preprocess the question
        question = self.txt_processors["eval"](question)
        # Generate answer
        answer = self.model.generate({"image": image, "text_input": question})
        return answer[0]

# Example usage
if __name__ == "__main__":
    blip = BLIP()

    # Image path
    image_path = "path_to_your_image.jpg"

    # Generate caption
    caption = blip.caption(image_path)
    print("Generated Caption:", caption)

    # Visual Question Answering
    question = "What is in the image?"
    answer = blip.vqa(image_path, question)
    print("VQA Answer:", answer)