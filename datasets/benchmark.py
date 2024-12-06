import csv
import json
import logging
import os
import re
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Literal

import numpy as np
import requests
import torch
from datasets import Dataset
from tqdm import tqdm
from PIL.Image import Image
from nibabel.spatialimages import SpatialImage

class Benchmark(ABC):
    """Abstract class for benchmarks."""

    def __init__(self, engine, logger) -> None:
        """Initialize the benchmark.

        Args:
            engine: Reference to the engine class.
            logger: A logger object.
        """
        self.task_name: str = "None"
        self.engine: "MultiMedEval" = engine
        self.modality: str = "None"
        self.task: str = "None"
        self._prompt = None
        self.train_dataset = None
        self.dataset: Optional[Dataset] = None
        self.logger: logging.Logger = logger

    def get_prompt(self):
        """Get the fewshot prompt."""
        if not self.train_dataset:
            return None

        if self._prompt is None:
            batcher_input = BatcherInput()
            for i in range(5):
                index = int(i / 5 * len(self.train_dataset))
                single_turn_input = self.format_question(
                    self.train_dataset[index],
                    prompt=True,
                )
                batcher_input = batcher_input + single_turn_input

            self._prompt = batcher_input
        return self._prompt

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.dataset)

    @abstractmethod
    def format_question(self, sample, prompt=False):
        """Format the question in a Huggingface format."""

    @abstractmethod
    def setup(self):
        """Setup the benchmark and download the dataset."""

    def __getitem__(self, idx):
        """Get an item from the dataset.

        Args:
            idx: The index of the item to get.

        Returns:
            The item from the dataset.
        """
        return {"idx": idx, "sample": self.dataset[idx]}

    @abstractmethod
    def evaluate(self, predictions: List[Dict[str, Union[int, BatcherOutput]]]):
        """Runs the evaluation on the predictions."""