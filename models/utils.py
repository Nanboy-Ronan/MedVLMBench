import models
from easydict import EasyDict as edict

def get_model(args):
    if args.model == "BLIP":
        model = models.BLIP(args=args)
    elif args.model == "LLaVA-1.5":
        model_path = "./pretrained_models/llava-v1.5-7b"
        model = models.LLaVA(args=edict(model_path=model_path, model_base=None))
        model.load_from_pretrained(model_path=model_path)
    else:
        raise NotImplementedError()

    return model