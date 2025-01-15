import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import Trainer, TrainerCallback
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Optional
from eval import get_eval_engine
from dataset import get_dataset
from dataset.utils import LinearProbingDataCollator


class CustomCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        This method is called at the end of each epoch.
        """
        if self.trainer.current_epoch % 10 == 0:
            print(f"Running evaluation at epoch {self.trainer.current_epoch}...")
            metrics = self.trainer.eval_engine.evaluate(args=self.trainer.args, model=self.trainer.model)
            print(f"Evaluation metrics at epoch {self.trainer.current_epoch}: {metrics}")
        self.trainer.current_epoch += 1
    
    def on_train_end(self, args, state, control, **kwargs):
        """
        This method is called after the full training is complete.
        """
        print("Training complete. Running final evaluation...")
        metrics = self.trainer.eval_engine.evaluate(args=self.trainer.args, model=self.trainer.model)
        print(f"Final evaluation metrics: {metrics}")


class CLIPLPTrainer(Trainer):
    def __init__(self, model, args, image_processor, train_dataset, eval_dataset, **kwargs):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs
        )
        dataset = get_dataset(args=args, split="test")
        self.eval_engine = get_eval_engine(args=args, dataset=dataset)
        self.image_processor = image_processor
        self.current_epoch = 0
        self.add_callback(CustomCallback(self))

    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        device = inputs["pixel_values"].device
        pixel_values = inputs["pixel_values"]
        labels = inputs["labels"]

        if self.image_processor is not None and pixel_values.dtype == torch.uint8: # torch.uinit8 means the image is processed with PILToImage only
            pixel_values = self.image_processor(pixel_values)

        logits = model(pixel_values)
        loss = F.cross_entropy(logits, labels)
        
        return (loss, logits) if return_outputs else loss

    def get_labels(self, eval_preds):
        logits, labels = eval_preds
        return labels

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)
        
        model_to_save = self.model.model

        if hasattr(model_to_save, 'module'):
            model_to_save = model_to_save.module

        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
    
    def on_epoch_end(self):
        """
        Called at the end of each epoch.
        Check if the current epoch is a multiple of 10, and if so, run evaluation.
        """
        if self.current_epoch % 10 == 0:  # Every 10 epochs
            self.log(f"Running evaluation at epoch {self.current_epoch}...")
            self.eval_engine.evaluate(args=args, model=model_wrapped)
            self.log(f"Evaluation metrics at epoch {self.current_epoch}: {metrics}")
        self.current_epoch += 1



def make_lp_data_module(args, dataset, image_processor):
    from dataset.diagnosis import PneumoniaMNIST
    if image_processor is None:
        transform = transforms.Compose([
            transforms.PILToTensor(),
            # transforms.Normalize(mean=model_constants['image_mean'], std=model_constants['image_std']),
        ])
    else:
        transform = image_processor

    data_collator = LinearProbingDataCollator()
    # TODO: check transform
    train_dataset = PneumoniaMNIST(
        data_args=args,
        split="train", # 'train', 'val' or 'test'
        transform=transform,
        target_transform=None,
        download=True,
        as_rgb=True,
        size=224,
        mmap_mode=None,
    )

    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

# TODO
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