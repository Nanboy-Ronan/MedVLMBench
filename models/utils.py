import models

def get_model(args):
    if args.model == "BLIP":
        model = models.BLIP(mode=args.task)
    else:
        raise NotImplementedError()

    return model