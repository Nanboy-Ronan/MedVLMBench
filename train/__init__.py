from train.caption import CaptionTrainEngine
from train.vqa import VQATrainEngine
from train.lp import DiagnosisLPTrainEngine
from train.clip_trainer import CLIPLPTrainer, make_lp_data_module
from train.clip_trainer import make_lp_data_module

task_engines = {"vqa": VQATrainEngine, "diagnosis": DiagnosisLPTrainEngine, "caption": CaptionTrainEngine}


def get_trainer(args, model_wrapped, dataset):
    if args.model in ["LLaVA-1.5", "LLaVA-Med"]:
        from model.release.llava.train.llava_trainer import LLaVATrainer
        from train.llava_trainer import make_supervised_data_module

        data_module = make_supervised_data_module(
            args,
            dataset=dataset,
            tokenizer=model_wrapped.tokenizer,
            image_processor=model_wrapped.image_processor,
            model_constants=model_wrapped.constants,
        )
        trainer = LLaVATrainer(model=model_wrapped.model, args=args, tokenizer=model_wrapped.tokenizer, **data_module)

        return trainer
    elif args.model in ["NVILA", "VILA1.5", "VILA-M3"]:
        from model.release.vila.train.llava_trainer import LLaVATrainer
        from model.release.vila.data import make_supervised_data_module

        data_module = make_supervised_data_module(
            args,
            dataset=dataset,
            tokenizer=model_wrapped.tokenizer,
            image_processor=model_wrapped.image_processor,
            model_constants=model_wrapped.constants,
        )
        trainer = LLaVATrainer(model=model_wrapped.model, args=args, tokenizer=model_wrapped.tokenizer, **data_module)

        return trainer
    elif args.model == "BLIP" or args.model == "BLIP2-2.7b":
        if args.usage in ["lp", "lora_lp", "clip-img-lora", "clip-txt-lora", "clip-full-lora"]:
            num_classes = args.num_classes if hasattr(args, 'num_classes') else 10
            data_module = make_lp_data_module(
                args,
                dataset=dataset,
                image_processor=None,
            )

            trainer = CLIPLPTrainer(
                model=model_wrapped,
                args=args,
                image_processor=model_wrapped.image_processor,
                **data_module
            )

            return trainer
        else:
            raise NotImplementedError()

        return trainer


    elif args.model == "XrayGPT":
        if args.usage in ["lp", "lora_lp", "clip-img-lora", "clip-txt-lora", "clip-full-lora"]:
            num_classes = args.num_classes if hasattr(args, 'num_classes') else 10
            data_module = make_lp_data_module(
                args,
                dataset=dataset,
                image_processor=None,
            )

            trainer = CLIPLPTrainer(
                model=model_wrapped,
                args=args,
                image_processor=model_wrapped.image_processor,
                **data_module
            )

            return trainer

        else:
            raise NotImplementedError()
    
    elif args.model == "BioMedCLIP":        
        if args.usage in ["lp", "lora_lp", "clip-img-lora", "clip-txt-lora", "clip-full-lora"]:
            num_classes = args.num_classes if hasattr(args, 'num_classes') else 10
            data_module = make_lp_data_module(
                args,
                dataset=dataset,
                image_processor=None,
            )

            trainer = CLIPLPTrainer(
                model=model_wrapped,
                args=args,
                image_processor=model_wrapped.image_processor,
                **data_module
            )

            return trainer
        else:
            raise NotImplementedError()
            
    elif args.model == "CLIP" or args.model == "MedCLIP" or args.model == "PMCCLIP":        
        if args.usage in ["lp", "lora_lp", "clip-img-lora", "clip-txt-lora", "clip-full-lora"]:
            num_classes = args.num_classes if hasattr(args, 'num_classes') else 10
            data_module = make_lp_data_module(
                args,
                dataset=dataset,
                image_processor=None,
            )

            trainer = CLIPLPTrainer(
                model=model_wrapped,
                args=args,
                image_processor=model_wrapped.image_processor,
                **data_module
            )

            return trainer
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError("Trainer not supported for {}".format(args.model))


def get_train_engine(args, model_wrapped, dataset):
    engine = task_engines[args.task](
        args=args,
        dataset=dataset,
        model_wrapped=model_wrapped,
        logger=args.logger,
        hf_trainer=get_trainer(args, model_wrapped=model_wrapped, dataset=dataset),
    )

    return engine
