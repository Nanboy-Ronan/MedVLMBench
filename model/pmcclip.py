from collections import OrderedDict
from typing import Optional, Sequence, Tuple
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, CenterCrop
from torchvision.transforms.functional import to_pil_image
from peft import LoraConfig, get_peft_model

from transformers import AutoTokenizer, AutoModel
from model.clip_base import CLIPBase
from model.lp_base import LPModel

from PIL import Image
import matplotlib.pyplot as plt

# Downloading link
# wget https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_encoder.pth
# wget "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/image_encoder(resnet50).pth"
# wget https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_projection_layer.pth

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        # FIXME support for non-transformer
        pass

    def stem(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        visual_output = dict.fromkeys(["image_features", "mim_loss"], None)
        visual_output.update({
            'image_features': x,
        })

        return visual_output

def _convert_to_rgb(image):
    return image.convert('RGB')

def image_transform(
        image_size: int,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        fill_color: int = 0,
):
    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    mean = mean or (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
    std = std or (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std
    normalize = Normalize(mean=mean, std=std)

    transforms = [
        Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_size),
    ]
    transforms.extend([
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])
    return Compose(transforms)

class CLIPModel(nn.Module):
    def __init__(self, image_encoder, text_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def forward(self, images, text_inputs):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(**text_inputs).last_hidden_state[:, 0, :]
        return image_features, text_features


class ImageProcessorLPCallable:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image):
        device = image.device
        image_batch_pil = [to_pil_image(img_tensor) for img_tensor in image]
        image = [torch.tensor(self.image_processor(pil_image)) for pil_image in image_batch_pil]
        image = torch.stack(image).to(device)
        return image

class PMCCLIPForDiagnosis(CLIPBase):
    def __init__(self, text, num_classes, *args, **kwargs) -> None:
        # Initialize encoders and projection layers
        image_encoder = ModifiedResNet(layers=[3, 4, 6, 3], output_dim=768, heads=8, image_size=224, width=64)
        image_encoder.load_state_dict(torch.load('./pretrained_models/pmcclip/image_encoder(resnet50).pth'))
        
        tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        text_encoder = AutoModel.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        text_encoder.load_state_dict(torch.load('./pretrained_models/pmcclip/text_encoder.pth'))

        text_projection_layer = torch.load('./pretrained_models/pmcclip/text_projection_layer.pth')
        text_projection_layer = nn.Parameter(text_projection_layer)

        logit_scale = 4.4292  # Initialize logit scaling factor
        
        model = CLIPModel(image_encoder, text_encoder)

        # Call the parent class initializer
        super().__init__(text=text, num_classes=num_classes, model=model)

        self.image_encoder = self.model.image_encoder
        self.text_encoder = self.model.text_encoder
        self.text_projection_layer = text_projection_layer
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale))
        self.tokenizer = tokenizer
        
        self.prototype = self.encode_text(self.prototype)
        self.image_processor = ImageProcessorLPCallable(self.image_transform(image_size=224))
        self.image_processor_evaluation = self.image_processor

    @staticmethod
    def image_transform(
        image_size: int,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        fill_color: int = 0,
    ):
        if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
            # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
            image_size = image_size[0]

        mean = mean or (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
        std = std or (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std
        normalize = Normalize(mean=mean, std=std)

        transforms = [
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
        ]
        transforms.extend([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)

    @torch.no_grad()
    def encode_text(self, text):
        """
        Encodes text descriptions into feature vectors using the text encoder and applies the text projection layer.
        """
        assert len(text) == self.num_classes
        
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        self.text_encoder.to(next(self.text_encoder.parameters()).device)
        text_outputs = self.text_encoder(**text_inputs).last_hidden_state[:, 0, :]  # Extract [CLS] token outputs
        projected_text_features = F.normalize(text_outputs @ self.text_projection_layer.to(next(self.text_encoder.parameters()).device), dim=-1)

        return projected_text_features

    def forward(self, images):
        """
        Forward pass to compute logits based on image and text features.
        """
        image_features = self.image_encoder(images)["image_features"]
        image_features = F.normalize(image_features, dim=-1)

        text_features = self.prototype.to(images.device)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T

        return logits


class PMCCLIPLoRAForDiagnosis(CLIPBase):
    def __init__(self, args, text, num_classes, **kwargs) -> None:
        # Initialize encoders and projection layers
        image_encoder = ModifiedResNet(layers=[3, 4, 6, 3], output_dim=768, heads=8, image_size=224, width=64)
        image_encoder.load_state_dict(torch.load('./pretrained_models/pmcclip/image_encoder(resnet50).pth'))
        
        tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        text_encoder = AutoModel.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        text_encoder.load_state_dict(torch.load('./pretrained_models/pmcclip/text_encoder.pth'))

        text_projection_layer = torch.load('./pretrained_models/pmcclip/text_projection_layer.pth')
        text_projection_layer = nn.Parameter(text_projection_layer)

        logit_scale = 4.4292  # Initialize logit scaling factor
        
        model = CLIPModel(image_encoder, text_encoder)

        if args.usage == "clip-img-lora":
            raise RuntimeError("Image encoder is resnet")
        elif args.usage == "clip-txt_lora":
            lora_config = LoraConfig(target_modules=["query", "key", "value"])
            for name, para in model.named_parameters():
                para.requires_grad = False
            model.text_encoder = get_peft_model(model.text_encoder, lora_config)
        else:
            raise NotImplementedError()

        # Call the parent class initializer
        super().__init__(text=text, num_classes=num_classes, model=model)

        self.image_encoder = self.model.image_encoder
        self.text_encoder = self.model.text_encoder
        self.text_projection_layer = text_projection_layer
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale))
        self.tokenizer = tokenizer
        
        self.prototype = self.encode_text(self.prototype)
        self.image_processor = ImageProcessorLPCallable(self.image_transform(image_size=224))
        self.image_processor_evaluation = self.image_processor

    @staticmethod
    def image_transform(
        image_size: int,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        fill_color: int = 0,
    ):
        if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
            # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
            image_size = image_size[0]

        mean = mean or (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
        std = std or (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std
        normalize = Normalize(mean=mean, std=std)

        transforms = [
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
        ]
        transforms.extend([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)

    @torch.no_grad()
    def encode_text(self, text):
        """
        Encodes text descriptions into feature vectors using the text encoder and applies the text projection layer.
        """
        assert len(text) == self.num_classes
        
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        self.text_encoder.to(next(self.text_encoder.parameters()).device)
        text_outputs = self.text_encoder(**text_inputs).last_hidden_state[:, 0, :]  # Extract [CLS] token outputs
        projected_text_features = F.normalize(text_outputs @ self.text_projection_layer.to(next(self.text_encoder.parameters()).device), dim=-1)

        return projected_text_features

    def forward(self, images):
        """
        Forward pass to compute logits based on image and text features.
        """
        image_features = self.image_encoder(images)["image_features"]
        image_features = F.normalize(image_features, dim=-1)

        text_features = self.prototype.to(images.device)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T

        return logits


class PMCCLIPLPForDiagnosis(LPModel):
    def __init__(self, backbone="ViT-B/32", *args, **kwargs) -> None:
        # Initialize encoders and projection layers
        image_encoder = ModifiedResNet(layers=[3, 4, 6, 3], output_dim=768, heads=8, image_size=224, width=64)
        image_encoder.load_state_dict(torch.load('./pretrained_models/pmcclip/image_encoder(resnet50).pth'))
        image_encoder.feat_dim = 768

        logit_scale = 4.4292  # Initialize logit scaling factor

        # Call the parent class initializer
        super().__init__(encoder=image_encoder, *args, **kwargs)
        
        self.image_processor = ImageProcessorLPCallable(self.image_transform(image_size=224))
        self.image_processor_evaluation = self.image_processor
    
    @staticmethod
    def image_transform(
        image_size: int,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        fill_color: int = 0,
    ):
        if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
            # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
            image_size = image_size[0]

        mean = mean or (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
        std = std or (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std
        normalize = Normalize(mean=mean, std=std)

        transforms = [
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
        ]
        transforms.extend([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)
    
    def forward(self, images):
        return self.head(self.encoder(images)['image_features'])