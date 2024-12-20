import os
import shutil
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

from model.release.llava_med.model import LlavaMistralForCausalLM
from model.release.llava_med.conversation import conv_templates
from model.release.llava_med.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from model.llava import LLaVA
from PIL import Image

import warnings


class ImageProcessorCallable:
    def __init__(self, image_processor, model_cfg):
        self.image_processor = image_processor
        self.model_cfg = model_cfg

    def __call__(self, image):
        return process_images([image], self.image_processor, self.model_cfg)[0]


class LLaVAMed(LLaVA):
    def __init__(self, args):
        super().__init__(args)

        self.conv_mode = "mistral_instruct"
        self.name = "LLaVA-Med"
        self.model_type = "medical"

    def load_from_pretrained(
        self,
        model_path,
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
        device="cuda",
        use_flash_attn=False,
        **kwargs,
    ):
        model_base = self.args.model_base
        model_name = get_model_name_from_path(model_path)

        kwargs = {}

        if device != "cuda":
            kwargs["device_map"] = {"": device}

        if load_8bit:
            kwargs["load_in_8bit"] = True
        elif load_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = torch.float16

        if "llava" in model_name.lower():
            # Load LLaVA model
            if "mistral" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=False, use_flash_attention_2=False, **kwargs
                )
        else:
            # Load language model
            if model_base is not None:
                # PEFT model
                from peft import PeftModel

                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
                print(f"Loading LoRA weights from {model_path}")
                model = PeftModel.from_pretrained(model, model_path)
                print(f"Merging weights")
                model = model.merge_and_unload()
                print("Convert to FP16...")
                model.to(torch.float16)
            else:
                use_fast = False
                if "mpt" in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
                    )
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

        image_processor = None

        if "llava" in model_name.lower():  # or 'mistral' in model_name.lower():
            mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
            mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
            if mm_use_im_patch_token:
                tokenizer.add_tokens([self.constants.DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            if mm_use_im_start_end:
                tokenizer.add_tokens(
                    [self.constants.DEFAULT_IM_START_TOKEN, self.constants.DEFAULT_IM_END_TOKEN], special_tokens=True
                )
            model.resize_token_embeddings(len(tokenizer))

            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(device=device, dtype=torch.float16)
            model.model.mm_projector.to(device=device, dtype=torch.float16)
            model.to(device=device, dtype=torch.float16)
            image_processor = vision_tower.image_processor

        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048

        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_processor_callable = ImageProcessorCallable(image_processor, model.config)
        self.context_len = context_len

        return tokenizer, model, image_processor, context_len

    def infer_vision_language(self, image, qs, temperature=0, image_size=None):
        # Model inference for vision-language tasks
        warnings.filterwarnings("ignore")

        qs = qs.replace(self.constants.DEFAULT_IMAGE_TOKEN, "").strip()
        if self.model.config.mm_use_im_start_end:
            qs = (
                self.constants.DEFAULT_IM_START_TOKEN
                + self.constants.DEFAULT_IMAGE_TOKEN
                + self.constants.DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = self.constants.DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, self.constants.IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.model.device)
        )

        if type(image) is Image.Image:
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]
            image_size = image.size
        else:
            image_tensor = image

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image_size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=None,
                num_beams=1,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
