from eval.vqa import VQAEvalEngine


task_engines = {"vqa": VQAEvalEngine}


def get_eval_engine(args, dataset):
    engine = task_engines[args.task](args=args, dataset=dataset, logger=args.logger)
    return engine
