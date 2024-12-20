from eval.vqa import VQAEvalEngine
from eval.caption import CaptionEvalEngine


task_engines = {"vqa": VQAEvalEngine, "caption": CaptionEvalEngine}


def get_eval_engine(args, dataset):
    engine = task_engines[args.task](args=args, dataset=dataset, logger=args.logger)
    return engine
