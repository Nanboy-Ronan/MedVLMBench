import os
import pandas as pd

from PIL import Image

from datasets import load_dataset, concatenate_datasets
from dataset.base import BaseDataset


class VQADataset(BaseDataset):
    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split)

        self.transform = transform


class SLAKE(VQADataset):
    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split, transform)

        self.name = "SLAKE"
        self.modality = "medical"

        if split == "all":
            df_train = load_dataset("BoKelvin/SLAKE", split="train")
            df_val = load_dataset("BoKelvin/SLAKE", split="validation")
            df_test = load_dataset("BoKelvin/SLAKE", split="test")

            self.ds = concatenate_datasets([df_train, df_val, df_test])
        else:
            self.ds = load_dataset("BoKelvin/SLAKE", split=split)
            # self.ds = load_dataset("BoKelvin/SLAKE", split=split).select(range(200))  # for debug only

        self.ds = self.ds.filter(lambda x: x["q_lang"].startswith("en"))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_args.image_path, self.ds[index]["img_name"])
        qs = self.ds[index]["question"]
        answer = self.ds[index]["answer"]
        is_open = self.ds[index]["answer_type"] == "OPEN"

        image = Image.open(image_path).convert("RGB")
        image_size = image.size

        if self.transform is not None:
            image = self.transform(image)

        return image, qs, answer, is_open, image_size, image_path


class PathVQA(VQADataset):
    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split, transform)

        self.name = "PathVQA"
        self.modality = "medical"

        if split == "all":
            df_train = load_dataset("flaviagiammarino/path-vqa", split="train")
            df_val = load_dataset("flaviagiammarino/path-vqa", split="validation")
            df_test = load_dataset("flaviagiammarino/path-vqa", split="test")

            self.ds = concatenate_datasets([df_train, df_val, df_test])
        else:
            self.ds = load_dataset("flaviagiammarino/path-vqa", split=split)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        qs = self.ds[index]["question"]
        answer = self.ds[index]["answer"]

        is_open = self.ds[index]["answer"] in ["yes", "no"]
        image_path = "No exist in this implementation"

        image = self.ds[index]["image"]
        image_size = image.size

        if self.transform is not None:
            image = self.transform(image)

        return image, qs, answer, is_open, image_size, image_path



class VQARAD(VQADataset):
    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split, transform)

        self.name = "VQA-RAD"
        self.modality = "medical"

        if split == "all":
            df_train = load_dataset("flaviagiammarino/vqa-rad", split="train")
            df_test = load_dataset("flaviagiammarino/vqa-rad", split="test")
            self.ds = concatenate_datasets([df_train, df_val, df_test])
        else:
            self.ds = load_dataset("flaviagiammarino/vqa-rad", split=split)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        qs = self.ds[index]["question"]
        answer = self.ds[index]["answer"]
        image = self.ds[index]["image"]

        is_open = answer in ["yes", "no"]

        image_size = image.size if hasattr(image, 'size') else (None, None)

        if self.transform is not None:
            image = self.transform(image)

        image_path = "No exist in this implementation"

        return image, qs, answer, is_open, image_size, image_path


if __name__ == "__main__":
    import torch
    from torchvision.transforms import PILToTensor

    from easydict import EasyDict as edict
    from torch.utils.data import DataLoader

    # dataset = SLAKE(edict(image_path="/mnt/hdd/data/SLAKE/imgs"), split="test", transform=PILToTensor())
    # dataset = SLAKE(edict(image_path="./data/SLAKE/imgs"), split="test", transform=PILToTensor())
    # dataset = PathVQA(edict(image_path="./data/PathVQA/imgs"), split="test", transform=PILToTensor())
    dataset = VQARAD(edict(image_path="./data/VQARAD/imgs"), split="test", transform=PILToTensor())

    image, qs, answer, is_open, image_size, image_path = dataset[0]

    dataloader = DataLoader(dataset, batch_size=2)

    for batch in dataloader:
        break

    print(batch[0])
    print(batch[1])
