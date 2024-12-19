import torchvision.transforms as transforms

from torch.utils.data import Dataset, WeightedRandomSampler


def get_transform(args):
    transform = transforms.Compose(
        [
            # transforms.Resize((256, 256)),
            transforms.PILToTensor(),
        ]
    )
    return transform
