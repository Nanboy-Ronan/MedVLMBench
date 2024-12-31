from train.vqa import VQATrainEngine
from train.lp import DiagnosisLPTrainEngine

task_engines = {"vqa": VQATrainEngine, "diagnosis": DiagnosisLPTrainEngine}


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
    elif args.model == "BLIP":
        if args.usage == "lp":
            from train.clip_trainer import CLIPLPTrainer
            from train.clip_trainer import make_lp_data_module

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


        elif args.usage == "clip-zs":
            from train.blip_trainer import BLIPTrainer
            from train.blip_data import make_contrastive_data_module

            train_dataset, eval_dataset = make_contrastive_data_module(
                args,
                dataset=dataset,
                tokenizer=model_wrapped.tokenizer,
                image_processor=model_wrapped.image_processor,
                model_constants=model_wrapped.constants,
            )

            trainer = BLIPTrainer(
                model=model_wrapped.model,
                args=training_args,
                tokenizer=model_wrapped.tokenizer,
                image_processor=model_wrapped.image_processor,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                temperature=args.temperature if hasattr(args, 'temperature') else 0.07,
            )

        else:
            raise NotImplementedError()

        return trainer


    elif args.model == "XrayGPT":
        if args.usage == "lp":
            from train.clip_trainer import XrayGPTLPTrainer
            from train.clip_trainer import make_lp_data_module

            num_classes = args.num_classes if hasattr(args, 'num_classes') else 10
            data_module = make_lp_data_module(
                args,
                dataset=dataset,
                image_processor=model_wrapped.image_processor,
            )

            trainer = XrayGPTLPTrainer(
                model=model_wrapped,
                args=args,
                image_processor=model_wrapped.image_processor,
                **data_module
            )

            return trainer
        else:
            raise NotImplementedError()
    
    elif args.model == "BioMedCLIP":        
        if args.usage == "lp":
            from train.clip_trainer import BioMedCLIPLPTrainer
            from train.clip_trainer import make_lp_data_module

            num_classes = args.num_classes if hasattr(args, 'num_classes') else 10
            data_module = make_lp_data_module(
                args,
                dataset=dataset,
                image_processor=model_wrapped.image_processor,
            )

            trainer = BioMedCLIPLPTrainer(
                model=model_wrapped,
                args=args,
                image_processor=model_wrapped.image_processor,
                **data_module
            )

            return trainer
        else:
            raise NotImplementedError()
    elif args.model == "CLIP":        
        if args.usage == "lp":
            from train.clip_trainer import CLIPLPTrainer
            from train.clip_trainer import make_lp_data_module

            num_classes = args.num_classes if hasattr(args, 'num_classes') else 10
            data_module = make_lp_data_module(
                args,
                dataset=dataset,
                image_processor=model_wrapped.image_processor,
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
