import os
import shutil
import warnings
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, LlamaForCausalLM
from torchvision import transforms
from model.release.vila.model import LlavaLlamaModel, LlavaTopDownLlamaModel
from model.release.vila.model.utils import is_mm_model
from model.release.vila.model.builder import prepare_config_for_eval
from model.release.vila.conversation import conv_templates, default_conversation
from model.release.vila.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from model.chat import ChatMetaModel
from utils.utils import maybe_zero_3
from PIL import Image


class ImageProcessorCallable:
    def __init__(self, image_processor, model_cfg):
        self.image_processor = image_processor
        self.model_cfg = model_cfg
        self.to_pil = transforms.ToPILImage()

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = self.to_pil(image)
        return image
    

class VILA(ChatMetaModel):
    def __init__(self, args):
        super().__init__(args)

        self.name = args.model_path
        self.model_type = "medical"

    def load_for_training(self, model_name_or_path):
        raise NotImplementedError

    def load_from_pretrained(
        self,
        model_path,
        model_base=None,
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
        device="cuda",
        use_flash_attn=False,
        **kwargs,
    ):
        if "NVILA-8B" in model_path:
            model_name = "NVILA-8B"
        else:
            raise NotImplementedError
        
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
            # kwargs["torch_dtype"] = torch.bfloat16

        if is_mm_model(model_path):
            # Load LLaVA model
            ## TODO @yunhao: mind fixing lora
            if "lora" in model_name.lower() and model_base is None:
                warnings.warn(
                    "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
                )
            if ("lora" in model_name.lower() or "dora" in model_name.lower()) and model_base is not None:
                lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
                print(lora_cfg_pretrained)
                print("Loading LLaVA from base model...")
                config = AutoConfig.from_pretrained(model_base)
                prepare_config_for_eval(config, kwargs)
                model = LlavaLlamaModel.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)
                tokenizer = model.tokenizer
                token_num, tokem_dim = model.llm.lm_head.out_features, model.llm.lm_head.in_features
                if model.llm.lm_head.weight.shape[0] != token_num:
                    model.llm.lm_head.weight = torch.nn.Parameter(
                        torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
                    )
                    model.llm.embed_tokens.weight = torch.nn.Parameter(
                        torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
                    )

                print("Loading additional LLaVA weights...")
                if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                    non_lora_trainables = torch.load(
                        os.path.join(model_path, "non_lora_trainables.bin"),
                        map_location="cpu",
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
            else:
                config = AutoConfig.from_pretrained(model_path)
                config.resume_path = model_path
                prepare_config_for_eval(config, kwargs)
                if "topdown" in config.model_type.lower():
                    model = LlavaTopDownLlamaModel(config=config, low_cpu_mem_usage=True, **kwargs)
                else:
                    model = LlavaLlamaModel(config=config, low_cpu_mem_usage=True, **kwargs)
                tokenizer = model.tokenizer
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
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        model.eval()
        image_processor = None
        if is_mm_model(model_path):
            model.resize_token_embeddings(len(tokenizer))
            vision_tower = model.get_vision_tower()
            if vision_tower is None:
                raise ValueError("Vision tower failed to load!")
            vision_tower.to(device=device, dtype=torch.float16)
            # vision_tower.to(device=device, dtype=torch.bfloat16)
            mm_projector = model.get_mm_projector()
            mm_projector.to(device=device, dtype=torch.float16)
            # mm_projector.to(device=device, dtype=torch.bfloat16)
            image_processor = vision_tower.image_processor

        if hasattr(model.llm.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048
        
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len

    def infer_vision_language(self, image, qs, temperature=0, image_size=None):
        # Model inference for vision-language tasks
        # TODO: Make it work for a batch
        image = transforms.ToPILImage()(image)
        prompt = [image, qs]
        answer_generated = self.model.generate_content(prompt)
        return answer_generated



    def infer_language(self, qs, temperature=0):
        raise NotImplementedError

    def save(self, output_folder, trainer=None):
        self.model.config.use_cache = True
        if self.args.lora_enable:
            state_dict = get_peft_state_maybe_zero_3(self.model.named_parameters(), self.args.lora_bias)
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters())
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_folder)
                self.model.save_pretrained(output_folder, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(output_folder, "non_lora_trainables.bin"))
        else:
            safe_save_model_for_hf_trainer(trainer=trainer, output_dir=output_folder)

if __name__ == "__main__":
    from easydict import EasyDict as edict
    from PIL import Image

    # run this command to set absolute import path before testing "export PYTHONPATH=/bigdata/rjin02/MedVLMBench:$PYTHONPATH"

    model_path = "Efficient-Large-Model/NVILA-8B"
    model_name = "NVILA-8B"

    prompt = """
    Please caption the image with findings for medical report.
    """

    # image_file = "/media/yesindeed/DATADRIVE1/mount/remote_cse/datasets/LLaVA-Med/data/images/34630837_F2.jpg"
    image_file = "/bigdata/rjin02/MedVLMBench/data/SLAKE/imgs/xmlab0/source.jpg"
    img = Image.open(image_file).convert("RGB")

    """
    GT report:
    Abdominopelvic CT scan in axial view indicates significant distension of the stomach and intestines with marked luminal dilatation observed in the oesophagus, stomach, small, and large bowel, accompanied by faecal loading. Notably, the distended large bowel is positioned anterior to the liver, causing medial displacement of the liver, which suggests a possible chronic underlying condition. This constellation of findings points to a long-standing obstructive process in the gastrointestinal tract, necessitating further clinical correlation and potential intervention.
    """

    llava_model = VILA(args=edict(model_path=model_path, model_name=model_name, model_base=None))
    llava_model.load_from_pretrained(model_path=model_path, model_name=model_name, model_base=None)
    
    output = llava_model.infer_vision_language(img, prompt)
    print(output)