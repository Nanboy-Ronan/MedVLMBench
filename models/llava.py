import os
import shutil
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from .release.llava.model import LlavaLlamaForCausalLM, LlavaMptForCausalLM, LlavaMistralForCausalLM
from .release.llava.conversation import conv_templates, default_conversation
from .release.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from .chat import ChatMetaModel


class LLaVA(ChatMetaModel):
    def __init__(self, model_args):
        super().__init__(model_args)

        self.conv_mode = "vicuna_v1"

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
        # Load models from pretrained weights. For inference only.
        model_base = self.model_args.model_base

        model_name = get_model_name_from_path(model_path)

        kwargs = {"device_map": device_map, **kwargs}

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

        if use_flash_attn:
            kwargs["attn_implementation"] = "flash_attention_2"

        if "llava" in model_name.lower():
            # Load LLaVA model

            if "lora" in model_name.lower() and model_base is None:
                warnings.warn(
                    "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
                )
            if "lora" in model_name.lower() and model_base is not None:
                from llava.model.language_model.llava_llama import LlavaConfig

                lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                print("Loading LLaVA from base model...")
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
                )
                token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
                if model.lm_head.weight.shape[0] != token_num:
                    model.lm_head.weight = torch.nn.Parameter(
                        torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
                    )
                    model.model.embed_tokens.weight = torch.nn.Parameter(
                        torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
                    )

                print("Loading additional LLaVA weights...")
                if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                    non_lora_trainables = torch.load(
                        os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu"
                    )
                else:
                    # this is probably from HF Hub
                    from huggingface_hub import hf_hub_download

                    def load_from_hf(repo_id, filename, subfolder=None):
                        cache_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
                        return torch.load(cache_file, map_location="cpu")

                    non_lora_trainables = load_from_hf(model_path, "non_lora_trainables.bin")
                non_lora_trainables = {
                    (k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()
                }
                if any(k.startswith("model.model.") for k in non_lora_trainables):
                    non_lora_trainables = {
                        (k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()
                    }
                model.load_state_dict(non_lora_trainables, strict=False)

                from peft import PeftModel

                print("Loading LoRA weights...")
                model = PeftModel.from_pretrained(model, model_path)
                print("Merging LoRA weights...")
                model = model.merge_and_unload()
                print("Model is loaded...")
            elif model_base is not None:
                # this may be mm projector only
                print("Loading LLaVA from base model...")
                if "mpt" in model_name.lower():
                    if not os.path.isfile(os.path.join(model_path, "configuration_mpt.py")):
                        shutil.copyfile(
                            os.path.join(model_base, "configuration_mpt.py"),
                            os.path.join(model_path, "configuration_mpt.py"),
                        )
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                    cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                    model = LlavaMptForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                    )
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                    cfg_pretrained = AutoConfig.from_pretrained(model_path)
                    model = LlavaLlamaForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                    )

                mm_projector_weights = torch.load(os.path.join(model_path, "mm_projector.bin"), map_location="cpu")
                mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
                model.load_state_dict(mm_projector_weights, strict=False)
            else:
                if "mpt" in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                    model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                elif "mistral" in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = LlavaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
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

        if "llava" in model_name.lower():
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
                vision_tower.load_model(device_map=device_map)
            if device_map != "auto":
                vision_tower.to(device=device_map, dtype=torch.float16)
            image_processor = vision_tower.image_processor

        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048

        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.context_len = context_len

    def infer_vision_language(self, image, qs, temperature=0):
        # Model inference for vision-language tasks
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
            .cuda()
        )

        image_tensor = process_images([image], self.image_processor, self.model.config)[0]

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
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

    def infer_language(self, qs, temperature=0):
        # model inference for language only tasks
        conv = default_conversation.copy()
        conv.append_message(conv.roles[0], qs)
        prompt = conv.get_prompt()
        inputs = self.tokenizer([prompt])

        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        output_ids = self.model.generate(
            input_ids,
            do_sample=True,
            use_cache=True,
            temperature=0.7,
            max_new_tokens=1024,
        )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        try:
            index = outputs.index(conv.sep, len(prompt))
        except ValueError:
            outputs += conv.sep
            index = outputs.index(conv.sep, len(prompt))

        outputs = outputs[len(prompt) + len(conv.roles[1]) + 2 : index].strip()

        return outputs


if __name__ == "__main__":
    from easydict import EasyDict as edict
    from PIL import Image

    # model download command: git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
    model_path = "/mnt/hdd/weights/llava-v1.5-7b"

    prompt = """
    Please caption the image with findings for medical report.
    """

    image_file = "/media/yesindeed/DATADRIVE1/mount/remote_cse/datasets/LLaVA-Med/data/images/34630837_F2.jpg"
    img = Image.open(image_file).convert("RGB")

    """
    GT report:
    Abdominopelvic CT scan in axial view indicates significant distension of the stomach and intestines with marked luminal dilatation observed in the oesophagus, stomach, small, and large bowel, accompanied by faecal loading. Notably, the distended large bowel is positioned anterior to the liver, causing medial displacement of the liver, which suggests a possible chronic underlying condition. This constellation of findings points to a long-standing obstructive process in the gastrointestinal tract, necessitating further clinical correlation and potential intervention.
    """

    llava_model = LLaVA(model_args=edict(model_path=model_path, model_base=None))
    llava_model.load_from_pretrained(model_path=model_path)

    output = llava_model.infer_vision_language(img, prompt)
    print(output)
