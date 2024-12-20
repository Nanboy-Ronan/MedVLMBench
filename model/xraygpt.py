import torch
from PIL import Image
from easydict import EasyDict as edict
from model.release.xraygpt.models.mini_gpt4 import MiniGPT4
from model.release.xraygpt.conversation.conversation import Conversation

from model.base import BaseModel
from model.chat import ChatMetaModel


class XrayGPT(ChatMetaModel):
    def __init__(self, args=None):
        super().__init__(args)
        self.name = "XrayGPT-mini"
        # self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = MiniGPT4(
            vit_model='eva_clip_g',
            q_former_model='https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth',
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision='fp16',
            freeze_vit=True,
            freeze_qformer=True,
            num_query_token=32,
            llama_model='./pretrained_models/Vicuna_Radiology_fp16/',
            prompt_path='./model/release/xraygpt/prompts/alignment.txt',
            prompt_template='###Patient: {} ###Doctor: ',
            max_txt_len=160,
            low_resource=True,
            end_sym='###'
        )

        ckpt = torch.load("./pretrained_models/xraygpt_pretrained1.pth", map_location="cpu")
        msg = self.model.load_state_dict(ckpt['model'], strict=False)
        all_ckpt_keys = set(ckpt['model'].keys())
        missing_keys = set(msg.missing_keys)
        unexpected_keys = set(msg.unexpected_keys)
        loaded_keys = all_ckpt_keys - unexpected_keys  # keys from checkpoint that aren't unexpected
        loaded_keys = loaded_keys - missing_keys

        print(f"Number of keys successfully loaded: {len(loaded_keys)}")
        assert len(loaded_keys) == len(all_ckpt_keys)


    def get_context_emb(self, conv, img_list):
        breakpoint()
        prompt = conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs
    
    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        breakpoint()
        conv.append_message(conv.roles[1], None)
        embs = self.get_context_emb(conv, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Doctor:')[-1].strip()
        conv.messages[-1][1] = output_text
        output_text = output_text.replace("ChatDoctor", "XrayGPT") ### additionally added
        output_text = output_text.replace("Chat Doctor", "XrayGPT") ### additionally added
        return output_text, output_token.cpu().numpy()


    def infer_vision_language(self, image, qs, image_size=None):
        """
        Generates answers based on input image and text prompt.
        :param image: The image tensor (preprocessed)
        :param qs: The input question/prompt as a string
        :param image_size: Optional parameter for image size
        :return: Generated text output
        """
        img_embeds, atts_img = self.model.encode_img(image)
        
        prompt = f"###Patient: <Img><ImageHere></Img> {qs}###Doctor:"

        # Wrap the image embeddings with the prompt text
        img_embeds, atts_img = self.model.prompt_wrap(img_embeds, atts_img, prompt)

        # Prepare the model inputs for generation
        # Add a BOS token at the beginning
        bos = torch.ones((img_embeds.size(0), 1), dtype=torch.long, device=device) * self.model.llama_tokenizer.bos_token_id
        bos_embeds = self.model.llama_model.model.embed_tokens(bos)

        # Concatenate BOS embedding with image+prompt embeddings
        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=device)

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
            eos_token_id=self.llama_tokenizer.eos_token_id,
            pad_token_id=self.llama_tokenizer.eos_token_id
        )

        output_token = outputs[0]
        if output_token[0] == 0:  # Remove initial <unk> token if present
            output_token = output_token[1:]
        if output_token[0] == 1:  # Remove <s> token if present
            output_token = output_token[1:]
        
        # Decode the output tokens
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        # The prompt uses '###' as a separator and 'Doctor:' to initiate the response.
        # We remove any trailing segments after '###' and strip extra whitespace.
        output_text = output_text.split('###')[0]
        answer = output_text.split('Doctor:')[-1].strip()
        return answer

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    ).to(device)

    image_path = "/fast/rjin02/DataSets/CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg"
    # image_path = "/fast/rjin02/DataSets/COCO/2014/val2014/COCO_val2014_000000000042.jpg"

    image = Image.open(image_path).convert("RGB")
    # prompt = "Question: how many cats are there? Answer:"
    prompt = "Question: What's in the image? Answer:"
    breakpoint()
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

    image = processor.image_processor(image)["pixel_values"]

    tokenizer_args = {'add_special_tokens': True, 'padding': False, 'stride': 0, 'return_overflowing_tokens': False, 'return_special_tokens_mask': False, 'return_offsets_mapping': False, 'return_token_type_ids': False, 'return_length': False, 'verbose': True}
    text_inputs = processor.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {
        "input_ids": text_inputs["input_ids"].to(device),
        "attention_mask": text_inputs["attention_mask"].to(device),
        "pixel_values": torch.tensor(image).unsqueeze(0).to(device),
    }
    
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(generated_text)