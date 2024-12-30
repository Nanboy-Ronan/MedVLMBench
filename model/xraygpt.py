import torch
from PIL import Image
from easydict import EasyDict as edict
from torchvision.transforms.functional import to_pil_image
from model.release.xraygpt.models.mini_gpt4 import MiniGPT4
from model.release.xraygpt.processors.blip_processors import Blip2ImageEvalProcessor


from model.base import BaseModel
from model.chat import ChatMetaModel
from model.lp_base import LPModel


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
            vit_precision="fp32",
            freeze_vit=True,
            freeze_qformer=True,
            num_query_token=32,
            llama_model='./pretrained_models/Vicuna_Radiology_fp16/',
            prompt_path='./model/release/xraygpt/prompts/alignment.txt',
            prompt_template='###Patient: {} ###Doctor: ',
            max_txt_len=160,
            low_resource=True,
            end_sym="###",
        )

        ckpt = torch.load("./pretrained_models/xraygpt_pretrained1.pth", map_location="cpu")
        msg = self.model.load_state_dict(ckpt['model'], strict=False)
        all_ckpt_keys = set(ckpt['model'].keys())
        missing_keys = set(msg.missing_keys)
        unexpected_keys = set(msg.unexpected_keys)
        loaded_keys = all_ckpt_keys - unexpected_keys  # keys from checkpoint that aren't unexpected
        loaded_keys = loaded_keys - missing_keys
        self.model.to(self.args.device)

        print(f"Number of keys successfully loaded: {len(loaded_keys)}")
        assert len(loaded_keys) == len(all_ckpt_keys)

        self.processor = Blip2ImageEvalProcessor() # TODO: wrap it. Tried wrap it on 1220 but get size issue due to loader.


    def infer_vision_language(self, image, qs, image_size=None):
        """
        Generates answers based on input image and text prompt.
        :param image: The image tensor (preprocessed)
        :param qs: The input question/prompt as a string
        :param image_size: Optional parameter for image size
        :return: Generated text output
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        else:
            assert image.dim() == 4

        image = torch.stack([self.processor(to_pil_image(img_tensor)) for img_tensor in image], dim=0).to(self.args.device)
        img_embeds, atts_img = self.model.encode_img(image)
        
        prompt = f"###Patient: <Img><ImageHere></Img> {qs}###Doctor:"

        # Wrap the image embeddings with the prompt text
        img_embeds, atts_img = self.model.prompt_wrap(img_embeds, atts_img, prompt)

        # Prepare the model inputs for generation
        # Add a BOS token at the beginning
        bos = torch.ones((img_embeds.size(0), 1), dtype=torch.long, device=self.args.device) * self.model.llama_tokenizer.bos_token_id
        bos_embeds = self.model.llama_model.model.embed_tokens(bos)

        # Concatenate BOS embedding with image+prompt embeddings
        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=self.args.device)

        # Generate the output from the model
        # You can adjust parameters like max_new_tokens, top_p, temperature, etc. as needed.
        outputs = self.model.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=300,
            num_beams=1,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=1.0,
            eos_token_id=self.model.llama_tokenizer.eos_token_id,
            pad_token_id=self.model.llama_tokenizer.eos_token_id
        )
        
        output_token = outputs[0]
        if output_token[0] == 0:  # Remove initial <unk> token if present
            output_token = output_token[1:]
        if output_token[0] == 1:  # Remove <s> token if present
            output_token = output_token[1:]
        
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        # The prompt uses '###' as a separator and 'Doctor:' to initiate the response.
        # We remove any trailing segments after '###' and strip extra whitespace.
        output_text = output_text.split('###')[0]
        answer = output_text.split('Doctor:')[-1].strip()
        return answer

class XGenGPTLPForDiagnosis(LPModel):
    def __init__(self, args=None) -> None:
        super().__init__(args)
        self.name = "XrayGPT-mini"
        self.model_type = "medical"
        self.model = MiniGPT4(
            vit_model="eva_clip_g",
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp32",
            freeze_vit=True,
            freeze_qformer=True,
            num_query_token=32,
            llama_model='./pretrained_models/Vicuna_Radiology_fp16/',
            prompt_path='./model/release/xraygpt/prompts/alignment.txt',
            prompt_template='###Patient: {} ###Doctor: ',
            max_txt_len=160,
            low_resource=True,
            end_sym="###",
        )

        ckpt = torch.load("./pretrained_models/xraygpt_pretrained1.pth", map_location="cpu")
        msg = self.model.load_state_dict(ckpt['model'], strict=False)
        all_ckpt_keys = set(ckpt['model'].keys())
        missing_keys = set(msg.missing_keys)
        unexpected_keys = set(msg.unexpected_keys)
        loaded_keys = all_ckpt_keys - unexpected_keys  # keys from checkpoint that aren't unexpected
        loaded_keys = loaded_keys - missing_keys

        self.model.to(self.args.device)

        self.vision_model = self.model.visual_encoder
        self.vision_model.feat_dim = 1408
        
        if "lp" in self.args.usage:
            from wrappers import LinearProbeWrapper
            self.model = LinearProbeWrapper(self.vision_model)
            # self.image_processor_callable = ImageProcessorCallable(self.image_processor)
        
        self.image_processor = Blip2ImageEvalProcessor()
    
    def load_for_training(self, model_path):
        pass
        
    def load_from_pretrained(self, model_path, device, **kwargs):
        model_ckpt = torch.load(model_path)
        self.model.load_state_dict(model_ckpt)
        self.model.to(device)
    
    def forward(self, x):
        return self.model.head(self.model.encoder(x)[:, 0, :])