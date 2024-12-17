import abc
import torch.nn as nn
from easydict import EasyDict as edict


class BaseModel:
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.name = ""

        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.image_processor_callable = None

        self.constants = edict()

    @abc.abstractmethod
    def load_from_pretrained(self):
        pass

    @abc.abstractmethod
    def init_for_training(self):
        pass

    @abc.abstractmethod
    def save(self, output_folder):
        pass
