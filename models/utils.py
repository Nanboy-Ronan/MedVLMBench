import models

def get_model(args):
    if args.model == "BLIP":
        model = models.BLIP(mode=args.task)
    elif args.model == "LLaVa-1.5":
        model = models.LLaVA()
    else:
        raise NotImplementedError()

    return model