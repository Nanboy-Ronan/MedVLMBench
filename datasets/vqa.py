import os
import pandas as pd

from PIL import Image

from datasets import load_dataset
from base import BaseDataset


class VQADataset(BaseDataset):
    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split)

        self.transform = transform


class SLAKE(VQADataset):
    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split, transform)

        self.ds = load_dataset("BoKelvin/SLAKE", split=split)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_args.image_path, self.ds[index]["img_name"])
        qs = self.ds[index]["question"]
        answer = self.ds[index]["answer"]
        is_open = self.ds[index]["answer_type"] == "OPEN"

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, qs, answer, image_path, is_open


if __name__ == "__main__":
    import torch
    from torchvision.transforms import PILToTensor

    from easydict import EasyDict as edict
    from torch.utils.data import DataLoader

    # dataset = SLAKE(edict(image_path="/mnt/hdd/data/SLAKE/imgs"), split="test", transform=PILToTensor())
    dataset = SLAKE(edict(image_path="./data/SLAKE/imgs"), split="test", transform=PILToTensor())

    image, qs, answer, image_path, is_open = dataset[0]

    dataloader = DataLoader(dataset, batch_size=2)

    for batch in dataloader:
        break

    print(batch[0])
    print(batch[1])
