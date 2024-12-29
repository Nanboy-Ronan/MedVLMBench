import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import Trainer

class ContrastiveDataset(Dataset):
    # TODO
    def __init__(self, data, tokenizer, transform, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        image = self.transform(item['image'])

        text = item['text']
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'pixel_values': image
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

def make_contrastive_data_module(args, dataset, tokenizer, image_processor, model_constants):
    # TODO
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_constants['image_mean'], std=model_constants['image_std']),
    ])

    train_data = dataset['train']
    val_data = dataset['val']

    train_dataset = ContrastiveDataset(train_data, tokenizer, transform, max_length=args.max_length)
    val_dataset = ContrastiveDataset(val_data, tokenizer, transform, max_length=args.max_length)

    return train_dataset, val_dataset

class BLIPTrainer(Trainer):
    def __init__(self, 
                 model, 
                 args, 
                 tokenizer, 
                 image_processor, 
                 train_dataset, 
                 eval_dataset, 
                 temperature=0.07, 
                 **kwargs):
        super().__init__(
            model=model,
            args=args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs
        )
        self.image_processor = image_processor
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        image_embeds = outputs.get('image_embeds')
        text_embeds = outputs.get('text_embeds')

        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        logits = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
        labels = torch.arange(logits.size(0)).to(logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        loss = (loss_i2t + loss_t2i) / 2

        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        return super().get_eval_dataloader(eval_dataset)