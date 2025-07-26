from train.caption import CaptionTrainEngine
from train.vqa import VQATrainEngine
from train.lp import DiagnosisLPTrainEngine
from train.clip_trainer import CLIPLPTrainer, make_diagnosis_data_module
from train.clip_trainer import make_diagnosis_data_module

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
    elif args.model in ["Lingshu"]:
        from transformers import Trainer
        from model.lingshu import LingshuDataset, LingshuCollator

        ds_train   = LingshuDataset(args, dataset, model_wrapped.processor)

        collator   = LingshuCollator(
            pad_token_id = model_wrapped.processor.tokenizer.pad_token_id,
            ignore_index  = model_wrapped.tokenizer.pad_token_id
        )
        trainer = Trainer(model=model_wrapped.model, args=args, tokenizer=model_wrapped.tokenizer, train_dataset=ds_train, data_collator=collator)

        return trainer
    
    elif args.model in ["CLIP", "MedCLIP", "PMCCLIP", "PLIP", "MedSigLIP", "XrayGPT", "BioMedCLIP", "BLIP", "BLIP2-2.7b", "PubMedCLIP", "SigLIP"]:        
        if args.usage in ["lp", "img-lora-lp", "clip-img-lora"]:
            data_module = make_diagnosis_data_module(
                train_dataset=dataset,
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
