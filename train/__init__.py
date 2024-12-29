from train.vqa import VQATrainEngine
from train.lp import DiagnosisLPTrainEngine

task_engines = {"vqa": VQATrainEngine, "diagnosis": DiagnosisLPTrainEngine}


def get_trainer(args, model_wrapped, dataset):
    if args.model == "LLaVA-1.5":
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
            from train.clip_trainer import make_lp_data_module, LinearProbingDataset

            num_classes = args.num_classes if hasattr(args, 'num_classes') else 10
            train_dataset, eval_dataset = make_lp_data_module(
                args,
                dataset=dataset,
                image_processor=model_wrapped.image_processor,
            )


            # training_args = TrainingArguments(
            #     output_dir=args.output_dir if hasattr(args, 'output_dir') else './results',
            #     num_train_epochs=args.num_epochs if hasattr(args, 'num_epochs') else 10,  # More epochs for linear probing
            #     per_device_train_batch_size=args.batch_size,
            #     per_device_eval_batch_size=args.batch_size,
            #     learning_rate=args.learning_rate,
            #     logging_dir=args.logging_dir if hasattr(args, 'logging_dir') else './logs',
            #     logging_steps=args.logging_steps if hasattr(args, 'logging_steps') else 100,
            #     evaluation_strategy='steps' if args.evaluate_during_training else 'no',
            #     save_strategy='steps' if args.save_steps else 'no',
            #     save_steps=args.save_steps if hasattr(args, 'save_steps') else 500,
            #     load_best_model_at_end=True if args.load_best_model else False,
            #     metric_for_best_model=args.metric_for_best_model if hasattr(args, 'metric_for_best_model') else None,
            #     greater_is_better=True if hasattr(args, 'greater_is_better') and args.greater_is_better else False,
            # )

            trainer = CLIPLPTrainer(
                model=model_wrapped,
                args=args,
                image_processor=model_wrapped.image_processor,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
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

            training_args = TrainingArguments(
                output_dir=args.output_dir if hasattr(args, 'output_dir') else './results',
                num_train_epochs=args.num_epochs if hasattr(args, 'num_epochs') else 3,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                logging_dir=args.logging_dir if hasattr(args, 'logging_dir') else './logs',
                logging_steps=args.logging_steps if hasattr(args, 'logging_steps') else 500,
                evaluation_strategy='steps' if args.evaluate_during_training else 'no',
                save_strategy='steps' if args.save_steps else 'no',
                save_steps=args.save_steps if hasattr(args, 'save_steps') else 1000,
                load_best_model_at_end=True if args.load_best_model else False,
                metric_for_best_model=args.metric_for_best_model if hasattr(args, 'metric_for_best_model') else None,
                greater_is_better=True if hasattr(args, 'greater_is_better') and args.greater_is_better else False,
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
