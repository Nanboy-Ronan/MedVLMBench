import abc
import torch.nn as nn
from easydict import EasyDict as edict


class BaseModel(abc.ABC):
    def __init__(self, model_args):
        super().__init__()

        self.model = None
        self.tokenizer = None
        self.model_args = model_args
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

    # @abc.abstractmethod
    # def caption(self, images):
    #     """Generate captions for the given images."""
    #     pass

    # @abc.abstractmethod
    # def vqa(self, images, questions):
    #     """Answer visual questions for the given images and questions."""
    #     pass
