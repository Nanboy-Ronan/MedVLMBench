from dataset.utils import get_prototype


def get_model(args, **kwargs):
    if args.task == "vqa" or args.task == "caption":
        if args.model == "BLIP":
            from model.blip import BLIPForQA
            model = BLIPForQA(args=args)
        elif args.model == "LLaVA-1.5":
            from model.llava import LLaVA
            model = LLaVA(args=args)
        elif args.model == "BLIP2-2.7b":
            from model.blip2 import BLIP2
            model = BLIP2(args=args)
        elif args.model == "LLaVA-Med":
            from model.llava_med import LLaVAMed
            model = LLaVAMed(args=args)
        elif args.model == "XGenMiniV1":
            from model.xgen import XGenMiniV1
            model = XGenMiniV1(args=args)
        elif args.model == "XrayGPT":
            from model.xraygpt import XrayGPT

            model = XrayGPT(args=args)
        elif args.model in ["NVILA", "VILA-M3", "VILA1.5"]:
            from model.vila import VILA
            model = VILA(args=args)
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
        from dataset.diagnosis import INFO

        num_classes = len(INFO[args.dataset.lower()]["label"])

        if args.usage == "lp":
            if args.model == "BLIP":
                from model.blip import BLIPLPForDiagnosis
                model = BLIPLPForDiagnosis(args=args, num_classes=num_classes)
            elif args.model == "XrayGPT":
                from model.xraygpt import XGenGPTLPForDiagnosis
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
                from model.blip2 import BLIP2LPForDiagnosis
                model = BLIP2LPForDiagnosis(args=args, num_classes=num_classes)
            elif args.model == "PMCCLIP":
                from model.pmcclip import PMCCLIPForDiagnosis, PMCCLIPLPForDiagnosis
                model = PMCCLIPLPForDiagnosis(args=args, num_classes=num_classes)
            else:
                raise NotImplementedError()
        elif args.usage == "lora_lp":
            if args.model == "BLIP":
                from model.blip import BLIPLoRALPForDiagnosis
                model = BLIPLoRALPForDiagnosis(args=args, num_classes=num_classes)
            elif args.model == "XrayGPT":
                from model.xraygpt import XGenGPTLoRALPForDiagnosis
                model = XGenGPTLoRALPForDiagnosis(args=args, num_classes=num_classes)
            elif args.model == "BioMedCLIP":
                from model.biomedclip import BioMedCLIPLoRALPForDiagnosis
                model = BioMedCLIPLoRALPForDiagnosis(args=args, num_classes=num_classes)
            elif args.model == "CLIP":
                from model.clip import CLIPLoRALPForDiagnosis
                model = CLIPLoRALPForDiagnosis(args=args, num_classes=num_classes)
            elif args.model == "BLIP2-2.7b":
                from model.blip2 import BLIP2LPLoRAForDiagnosis
                model = BLIP2LPLoRAForDiagnosis(args=args, num_classes=num_classes)
            else:
                raise NotImplementedError()
        elif args.usage in ["clip-zs", "clip-img-lora", "clip-txt-lora", "clip-full-lora"]:
            text = get_prototype(args)
            text = ["a photo of {}".format(txt) for txt in text]
            if args.model == "BLIP":
                from model.blip import BLIPForDiagnosis
                model = BLIPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "CLIP":
                from model.clip import CLIPForDiagnosis
                model = CLIPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "BioMedCLIP":
                from model.biomedclip import BiomedCLIPForDiagnosis
                model = BiomedCLIPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "MedCLIP":
                from model.medclip import MedCLIPForDiagnosis
                model = MedCLIPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "PMCCLIP":
                model = PMCCLIPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "BLIP2-2.7b":
                from model.blip2 import BLIP2ForDiagnosis
                model = BLIP2ForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "PLIP":
                from model.plip import PLIPForDiagnosis
                model = PLIPForDiagnosis(args=args, text=text, num_classes=num_classes)
            else:
                raise NotImplementedError()
        elif args.usage in ["clip-adapter"]:
            text = get_prototype(args)
            text = ["a photo of {}".format(txt) for txt in text]
            if args.model == "CLIP":
                # CLIPAdapterWrapper needs the base CLIP model instance
                from model.clip_adapter import CLIPAdapterWrapper
                clip_model_instance = CLIPForDiagnosis(text=text, num_classes=num_classes)
                model = CLIPAdapterWrapper(text=text, num_classes=num_classes, clip_model=clip_model_instance.model)
            else:
                raise NotImplementedError(f"CLIP-Adapter not implemented for model {args.model}")
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    return model