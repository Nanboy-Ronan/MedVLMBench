from model.blip import BLIPForQA, BLIPLPForDiagnosis, BLIPLoRALPForDiagnosis, BLIPForDiagnosis, BLIPLoRAForDiagnosis
from model.llava import LLaVA
from model.blip2 import BLIP2, BLIP2ForDiagnosis
from model.llava_med import LLaVAMed
from model.xgen import XGenMiniV1
from model.xraygpt import XrayGPT, XGenGPTLPForDiagnosis, XGenGPTLoRALPForDiagnosis
from model.biomedclip import BioMedCLIPLPForDiagnosis, BioMedCLIPLoRALPForDiagnosis, BiomedCLIPForDiagnosis, BiomedCLIPLoRAForDiagnosis
from model.clip import CLIPLPForDiagnosis, CLIPLoRALPForDiagnosis, CLIPForDiagnosis, CLIPLoRAForDiagnosis
from model.medclip import MedCLIPLPForDiagnosis, MedCLIPForDiagnosis, MedCLIPLoRAForDiagnosis
from model.pmcclip import PMCCLIPForDiagnosis
from dataset.diagnosis import INFO

from dataset.utils import get_prototype


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
        if args.usage == "lp":
            if args.model == "BLIP":
                model = BLIPLPForDiagnosis(args=args, num_classes=num_classes)
            elif args.model == "XrayGPT":
                model = XGenGPTLPForDiagnosis(args=args, num_classes=num_classes)
            elif args.model == "BioMedCLIP":
                model = BioMedCLIPLPForDiagnosis(args=args, num_classes=num_classes)
            elif args.model == "CLIP":
                model = CLIPLPForDiagnosis(args=args, num_classes=num_classes)
            elif args.model == "MedCLIP":
                model = MedCLIPLPForDiagnosis(args=args, num_classes=num_classes)
            else:
                raise NotImplementedError()
        elif args.usage == "lora_lp":
            if args.model == "BLIP":
                model = BLIPLoRALPForDiagnosis(args=args, num_classes=num_classes)
            elif args.model == "XrayGPT":
                model = XGenGPTLoRALPForDiagnosis(args=args, num_classes=num_classes)
            elif args.model == "BioMedCLIP":
                model = BioMedCLIPLoRALPForDiagnosis(args=args, num_classes=num_classes)
            elif args.model == "CLIP":
                model = CLIPLoRALPForDiagnosis(args=args, num_classes=num_classes)
            else:
                raise NotImplementedError()
        elif args.usage == "clip-zs":
            text = get_prototype(args)
            if args.model == "BLIP":
                model = BLIPForDiagnosis(text=text, num_classes=num_classes)
            elif args.model == "CLIP":
                model = CLIPForDiagnosis(text=text, num_classes=num_classes)
            elif args.model == "BioMedCLIP":
                model = BiomedCLIPForDiagnosis(text=text, num_classes=num_classes)
            elif args.model == "MedCLIP":
                model = MedCLIPForDiagnosis(text=text, num_classes=num_classes)
            elif args.model == "PMCCLIP":
                model = PMCCLIPForDiagnosis(text=text, num_classes=num_classes)
            elif args.model == "BLIP2-2.7b":
                model = BLIP2ForDiagnosis(text=text, num_classes=num_classes)
            else:
                raise NotImplementedError()
        elif args.model == "clip-lora":
            text = get_prototype(args)
            if args.model == "BLIP":
                model = BLIPLoRAForDiagnosis(text=text, num_classes=num_classes)
            elif args.model == "CLIP":
                model = CLIPLoRAForDiagnosis(text=text, num_classes=num_classes)
            elif args.model == "BioMedCLIP":
                model = BiomedCLIPLoRAForDiagnosis(text=text, num_classes=num_classes)
            elif args.model == "MedCLIP":
                model = MedCLIPLoRAForDiagnosis(text=text, num_classes=num_classes)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    return model
