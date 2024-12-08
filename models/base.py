import abc
import torch.nn as nn

class BaseModel(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def caption(self, images):
        """Generate captions for the given images."""
        pass

    @abc.abstractmethod
    def vqa(self, images, questions):
        """Answer visual questions for the given images and questions."""
        pass
