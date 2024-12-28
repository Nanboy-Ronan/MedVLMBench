import csv
import json
import logging
import os
import string
import json
import pathlib

import numpy as np
import pandas as pd
import requests
import transformers
import torch
from torch.utils.data import DataLoader


class TrainEngine:
    def __init__(self, args, dataset, model_wrapped, logger, hf_trainer=None):
        """Initialize the benchmark.

        Args:
            logger: A logger object.
        """
        self.args = args
        self.task: str = "None"
        self.dataset = dataset
        self.model_wrapped = model_wrapped

        self.hf_trainer = hf_trainer
        self.logger = logger

    def train(self):
        if self.hf_trainer is not None:
            if list(pathlib.Path(self.args.output_dir).glob("checkpoint-*")):
                self.hf_trainer.train(resume_from_checkpoint=True)
            else:
                self.hf_trainer.train()
        else:
            # TODO: implementation of not using HF trainer
            pass

        self.save()

    def train_batch(self, subject, model):
        pass

    def save(self):
        if self.hf_trainer is not None:
            self.hf_trainer.save_state()

            self.model_wrapped.save(self.args.output_dir, self.hf_trainer)
        else:
            self.model_wrapped.save(self.args.output_dir)
