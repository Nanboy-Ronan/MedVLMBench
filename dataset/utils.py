import torchvision.transforms as transforms

from dataset.diagnosis import INFO
from torch.utils.data import Dataset, WeightedRandomSampler


def get_transform(args):
    transform = transforms.Compose(
        [
            # transforms.Resize((256, 256)),
            transforms.PILToTensor(),
        ]
    )
    return transform

def get_prototype(args):
    text_classes = list(INFO[args.dataset.lower()]["label"].values())
    return text_classes