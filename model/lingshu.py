from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from model.chat import ChatMetaModel

class Lingshu(ChatMetaModel):

    def __init__(self, args):
        super().__init__(args)
        self.name = "Lingshu"
        self.model_type = "medical"
        self.processor = None
        self.context_len = 2048

    def load_from_pretrained(
        self,
        # model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        model_path: str = "lingshu-medical-mllm/Lingshu-7B",
        device_map: str = "auto",
        torch_dtype: str | torch.dtype = "auto",
        **kwargs,
    ):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            # **hf_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)


    def _build_messages(self, image: Image.Image | str | None, text: str):
        """
        Build the messages list in the format expected by
        `processor.apply_chat_template`.
        """
        if image is None:

            return [{"role": "user", "content": text}]
        else:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text": text},
                    ],
                }
            ]

    @torch.inference_mode()
    def infer_vision_language(self, image, qs, temperature: float = 0.2, **gen_kwargs):
        """
        Single-image VL-QA / captioning.  Accepts PIL.Image, filepath or URL (the
        processor handles URLs transparently).
        """
        image = self._to_pil(image)
        messages = self._build_messages(image, qs)

        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=gen_kwargs.get("max_new_tokens", 256),
            temperature=temperature,
        )

        answer_ids = gen_ids[:, inputs.input_ids.shape[-1] :]
        answer = self.processor.batch_decode(
            answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return answer.strip()

    @torch.inference_mode()
    def infer_language(self, qs, temperature: float = 0.7, **gen_kwargs):
        """
        Text-only inference.  We still let Qwen format the prompt so that system /
        assistant roles work in the same way as multimodal chats.
        """
        messages = self._build_messages(None, qs)
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[chat_text], padding=True, return_tensors="pt"
        ).to(self.model.device)

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=gen_kwargs.get("max_new_tokens", 512),
            temperature=temperature,
        )
        answer_ids = gen_ids[:, inputs.input_ids.shape[-1] :]
        return self.processor.batch_decode(
            answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def save(self, output_folder, trainer=None):
        self.model.save_pretrained(output_folder)
        self.processor.save_pretrained(output_folder)
    
    def _to_pil(self, img: Image.Image | torch.Tensor | str):
        """
        Make sure the vision object is a PIL.Image or a path/URL.
        """
        if isinstance(img, torch.Tensor):
            # If the tensor is float in [0, 1] convert to uint8 first
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).to(torch.uint8)
            img = to_pil_image(img.cpu())
        return img
