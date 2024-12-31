from model.blip import BLIPForQA, BLIPForDiagnosis, BLIPLPForDiagnosis
from model.llava import LLaVA
from model.blip2 import BLIP2
from model.llava_med import LLaVAMed
from model.xgen import XGenMiniV1
from model.xraygpt import XrayGPT, XGenGPTLPForDiagnosis
from model.biomedclip import BioMedCLIPLPForDiagnosis
from dataset.diagnosis import INFO


from easydict import EasyDict as edict


def get_model(args, **kwargs):
    if args.task == "vqa":
        if args.model == "BLIP":
            model = BLIPForQA(args=args)
        elif args.model == "LLaVA-1.5":
            model = LLaVA(args=args)
        elif args.model == "BLIP2-2.7b":
            model = BLIP2(args=args)
        elif args.model == "LLaVA-Med":
            model = LLaVAMed(args=args)
        elif args.model == "XGenMiniV1":
            model = XGenMiniV1(args=args)
        elif args.model == "XrayGPT":
            from model.xraygpt import XrayGPT
            model = XrayGPT(args=args)
        else:
            raise NotImplementedError()
    elif args.task == "caption":
        if args.model == "BLIP":
            model = BLIPForQA(args=args)
        elif args.model == "LLaVA-1.5":
            model = LLaVA(args=args)
        elif args.model == "BLIP2-2.7b":
            model = BLIP2(args=args)
        elif args.model == "LLaVA-Med":
            model = LLaVAMed(args=args)
        elif args.model == "XGenMiniV1":
            model = XGenMiniV1(args=args)
        elif args.model == "XrayGPT":
            from model.xraygpt import XrayGPT
            model = XrayGPT(args=args)
        else:
            raise NotImplementedError()
    elif args.task == "diagnosis":
        num_classes = len(INFO[args.dataset.lower()]["label"])
        if args.model == "BLIP":
            model = BLIPLPForDiagnosis(args=args, num_classes=num_classes)
        elif args.model == "XrayGPT":
            model = XGenGPTLPForDiagnosis(args=args, num_classes=num_classes)
        elif args.model == "BioMedCLIP":
            model = BioMedCLIPLPForDiagnosis(args=args, num_classes=num_classes)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    return model
