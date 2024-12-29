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
            'pixel_values': image,
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

class CLIPLPTrainer(Trainer):
    def __init__(self, model, args, image_processor, train_dataset, eval_dataset, **kwargs):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs
        )
        self.image_processor = image_processor

    def compute_loss(self, model, inputs, return_outputs=False):
        pixel_values = inputs.pop("pixel_values")
        labels = inputs.pop("labels")

        logits = model(pixel_values=pixel_values)
        
        loss = F.cross_entropy(logits, labels)
        
        return (loss, logits) if return_outputs else loss

    def get_labels(self, eval_preds):
        """
        Extracts labels from the evaluation predictions.

        Args:
            eval_preds: The evaluation predictions.

        Returns:
            torch.Tensor: The ground truth labels.
        """
        logits, labels = eval_preds
        return labels

class LinearProbingDataset(Dataset):
    def __init__(self, data, tokenizer, transform, max_length=128):
        """
        Initializes the dataset.

        Args:
            data (list of dict): Each dict should have 'image' and 'label' keys.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for text (unused here but kept for compatibility).
            transform (torchvision.transforms.Compose): Transformations for images.
            max_length (int, optional): Max token length for texts (unused here but kept for compatibility). Defaults to 128.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        image = self.transform(item['image'])

        label = item['label']
        
        return {
            'pixel_values': image,
            'labels': label
        }

def make_lp_data_module(args, dataset, image_processor):
    from dataset.diagnosis import PneumoniaMNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=model_constants['image_mean'], std=model_constants['image_std']),
    ])

    # TODO: check transform
    train_dataset = PneumoniaMNIST(
        data_args=args,
        split="train", # 'train', 'val' or 'test'
        transform=image_processor,
        target_transform=None,
        download=True,
        as_rgb=True,
        size=224,
        mmap_mode=None,
    )

    val_dataset = PneumoniaMNIST(
        data_args=args,
        split="val", # 'train', 'val' or 'test'
        transform=image_processor,
        target_transform=None,
        download=True,
        as_rgb=True,
        size=224,
        mmap_mode=None,
    )

    return train_dataset, val_dataset


class CLIPTrainer(Trainer):
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