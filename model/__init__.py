from model.blip import BLIPForQA, BLIPLPForDiagnosis, BLIPLoRALPForDiagnosis, BLIPForDiagnosis
from model.llava import LLaVA
from model.blip2 import BLIP2, BLIP2ForDiagnosis, BLIP2LPForDiagnosis, BLIP2LPLoRAForDiagnosis
from model.llava_med import LLaVAMed
from model.xgen import XGenMiniV1
from model.xraygpt import XrayGPT, XGenGPTLPForDiagnosis, XGenGPTLoRALPForDiagnosis
from model.biomedclip import BioMedCLIPLPForDiagnosis, BioMedCLIPLoRALPForDiagnosis, BiomedCLIPForDiagnosis
from model.clip import CLIPLPForDiagnosis, CLIPLoRALPForDiagnosis, CLIPForDiagnosis
from model.medclip import MedCLIPLPForDiagnosis, MedCLIPForDiagnosis, MedCLIPLoRALPForDiagnosis
from model.pmcclip import PMCCLIPForDiagnosis, PMCCLIPLPForDiagnosis
from model.plip import PLIPForDiagnosis
from model.clip_adapter import CLIPAdapterWrapper
from dataset.diagnosis import INFO

from dataset.utils import get_prototype


from easydict import EasyDict as edict


def get_model(args, **kwargs):
    if args.task == "vqa" or args.task == "caption":
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
            model = XrayGPT(args=args)
        else:
            raise NotImplementedError()
            
    elif args.task == "diagnosis":
        from dataset.diagnosis import INFO

        num_classes = len(INFO[args.dataset.lower()]["label"])

        if args.usage == "lp":
            if args.model == "BLIP":
                model = BLIPLPForDiagnosis(args=args, num_classes=num_classes)
            elif args.model == "XrayGPT":
                model = XGenGPTLPForDiagnosis(args=args, num_classes=num_classes)
            elif args.model == "BioMedCLIP":
                from model.biomedclip import BioMedCLIPLPForDiagnosis

                model = BioMedCLIPLPForDiagnosis(args=args, num_classes=num_classes)
            elif args.model == "CLIP":
                from model.clip import CLIPLPForDiagnosis

                model = CLIPLPForDiagnosis(args=args, num_classes=num_classes)
            elif args.model == "MedCLIP":
                from model.medclip import MedCLIPLPForDiagnosis

                model = MedCLIPLPForDiagnosis(args=args, num_classes=num_classes)
            elif args.model == "BLIP2-2.7b":
                model = BLIP2LPForDiagnosis(args=args, num_classes=num_classes)
            elif args.model == "PMCCLIP":
                model = PMCCLIPLPForDiagnosis(args=args, num_classes=num_classes)
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
            elif args.model == "BLIP2-2.7b":
                model = BLIP2LPLoRAForDiagnosis(args=args, num_classes=num_classes)
            else:
                raise NotImplementedError()
        elif args.usage in ["clip-zs", "clip-img-lora", "clip-txt-lora", "clip-full-lora"]:
            text = get_prototype(args)
            text = ["a photo of {}".format(txt) for txt in text]
            if args.model == "BLIP":
                model = BLIPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "CLIP":
                model = CLIPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "BioMedCLIP":
                model = BiomedCLIPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "MedCLIP":
                model = MedCLIPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "PMCCLIP":
                model = PMCCLIPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "BLIP2-2.7b":
                model = BLIP2ForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "PLIP":
                model = PLIPForDiagnosis(args=args, text=text, num_classes=num_classes)
            else:
                raise NotImplementedError()
        elif args.usage in ["clip-adapter"]:
            text = get_prototype(args)
            text = ["a photo of {}".format(txt) for txt in text]
            if args.model == "CLIP":
                # CLIPAdapterWrapper needs the base CLIP model instance
                clip_model_instance = CLIPForDiagnosis(text=text, num_classes=num_classes)
                model = CLIPAdapterWrapper(text=text, num_classes=num_classes, clip_model=clip_model_instance.model)
            else:
                raise NotImplementedError(f"CLIP-Adapter not implemented for model {args.model}")
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    return model