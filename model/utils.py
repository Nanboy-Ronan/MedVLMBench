import model
from easydict import EasyDict as edict


def get_model(args, **kwargs):
    if args.model == "BLIP":
        model = model.BLIP(model_args=args)
    elif args.model == "LLaVa-1.5":
        model = model.LLaVA(model_args=edict(model_path=args.model_path, model_base=None))

    else:
        raise NotImplementedError()

    model.load_from_pretrained(model_path=args.model_path, **kwargs)

    return model
