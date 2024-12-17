from .vqa import VQAEvalEngine


def get_benchmark(args, dataset=None):
    benchmark = VQAEvalEngine(args=args, dataset=dataset, logger=args.logger)
    return benchmark