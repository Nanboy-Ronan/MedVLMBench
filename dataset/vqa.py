import os
import pandas as pd

from PIL import Image

from dataset import load_dataset, concatenate_datasets
from dataset.base import BaseDataset


class VQADataset(BaseDataset):
    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split)

        self.transform = transform


class SLAKE(VQADataset):
    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split, transform)

        self.name = "SLAKE"
        self.modality = "general"

        if split == "all":
            df_train = load_dataset("BoKelvin/SLAKE", split="train")
            df_val = load_dataset("BoKelvin/SLAKE", split="validation")
            df_test = load_dataset("BoKelvin/SLAKE", split="test")

            self.ds = concatenate_datasets([df_train, df_val, df_test])
        else:
            self.ds = load_dataset("BoKelvin/SLAKE", split=split)

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


if __name__ == "__main__":
    import torch
    from torchvision.transforms import PILToTensor

    from easydict import EasyDict as edict
    from torch.utils.data import DataLoader

    dataset = SLAKE(edict(image_path="/mnt/hdd/data/SLAKE/imgs"), split="test", transform=PILToTensor())
    # dataset = SLAKE(edict(image_path="./data/SLAKE/imgs"), split="test", transform=PILToTensor())

    image, qs, answer, image_path, is_open = dataset[0]

    dataloader = DataLoader(dataset, batch_size=2)

    for batch in dataloader:
        break

    print(batch[0])
    print(batch[1])
