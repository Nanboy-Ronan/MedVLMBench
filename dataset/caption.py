import os
import pandas as pd

from PIL import Image

from datasets import load_dataset, concatenate_datasets
from dataset.base import BaseDataset


class CaptionDataset(BaseDataset):
    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split)

        self.transform = transform
