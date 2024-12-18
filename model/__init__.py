from model.blip import BLIP
from model.llava import LLaVA
from model.blip2 import BLIP2

from easydict import EasyDict as edict


def get_model(args, **kwargs):
    if args.model == "BLIP":
        model = BLIP(args=args)
    elif args.model == "LLaVA-1.5":
        model = LLaVA(args=edict(model_path=args.model_path, model_base=None))
    elif args.model == "BLIP2-2.7b":
        model = BLIP2(args=args)
    else:
        raise NotImplementedError()

    model.load_from_pretrained(model_path=args.model_path, **kwargs)

    return model
