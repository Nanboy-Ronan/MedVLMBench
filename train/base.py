import csv
import json
import logging
import os
import string
import json

import numpy as np
import pandas as pd
import requests
import transformers
import torch
from torch.utils.data import DataLoader


class TrainEngine:
    def __init__(self, args, dataset, logger):
        """Initialize the benchmark.

        Args:
            logger: A logger object.
        """
        self.args = args
        self.task: str = "None"
        self.prompt_template = "{}"
        self.dataset = dataset

        self.hf_trainer = None
        self.lora_enable = False
        self.logger = logger

    def train(self, args, model):
        pass

    def train_batch(self, subject, model):
        pass

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get an item from the dataset.

        Args:
            idx: The index of the item to get.

        Returns:
            The item from the dataset.
        """
        return {"idx": idx, "sample": self.dataset[idx]}

    def save(self, path):
        pass
