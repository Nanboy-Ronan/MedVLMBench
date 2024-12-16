<<<<<<< HEAD:eval/vqa.py
from .base import EvalEngine
=======
from evaluation.benchmark import Benchmark
>>>>>>> ruinan:evaluation/vqa.py
from torchvision.transforms.functional import to_pil_image

from torchmetrics.functional.text import bleu_score, rouge_score

<<<<<<< HEAD:eval/vqa.py

class VQAEvalEngine(EvalEngine):
    def __init__(self, dataset, logger):
        super().__init__(dataset, logger)
=======
class VQABenchmark(Benchmark):
    def __init__(self, args, dataset, logger):
        super().__init__(args, dataset, logger)
>>>>>>> ruinan:evaluation/vqa.py

        self.task = "VQA"

        # 0 for closed question, 1 for open question
        self.prompt_template = [
            "Answer the following question about the image with yes or no. {}",
            "{}",
        ]

    def evaluate_batch(self, batch, model):
        image, qs, answer, image_path, is_open = batch
        device = self.args.device
        
        image = image.to(device, non_blocking=True)

        prompt = [self.prompt_template[int(_is_open)].format(_qs) for _qs, _is_open in zip(qs, is_open)]

        output = model.infer_vision_language(image, prompt)

        batch_size = len(qs)

        # if is_open:
        #     # evaluation of open questions
        #     pass
        # else:
        #     # evaluation of closed questions
        #     pass

        self.metric_logger.meters["xxx"].update(len(prompt), n=batch_size)
