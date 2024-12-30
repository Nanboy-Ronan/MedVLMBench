import os
import warnings
import numpy as np
import pandas as pd
import torch
from collections import namedtuple
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from eval.base import EvalEngine

Metrics = namedtuple("Metrics", ["AUC", "ACC"])


class DiagnosisEvalEngine(EvalEngine):
    # TODO: support for multi-class
    def __init__(self, args, dataset, logger, task="binary-class", device="cpu"):
        super().__init__(args, dataset, logger)

        self.task = task  # e.g., "multi-class", "binary-class", "multi-label"
        self.device = device 

        self.metrics = {
            "accuracy": 0.0,
            "auc": 0.0,
        }

    def evaluate(self, args, model):
        """Run evaluation on the classification dataset."""
        data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        
        self.init_metric_logger()

        with torch.no_grad():
            for batch in self.metric_logger.log_every(data_loader, self.args.eval_print_freq, header="Test:"):
                subject = {k: v[0] for k, v in batch.items()}
                self.evaluate_subject(subject, model)

        self.metric_logger.synchronize_between_processes()

        results = {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}
        self.logger.info("\nEvaluation results:\n" + "\n".join(f"{k} {v:.3f}" for k, v in results.items()))

        return results

    def evaluate_subject(self, subject, model):
        """Evaluate a single subject (image and label)."""
        image = subject["pixel_values"]
        true_label = subject["label"]

        image = image.to(self.device, non_blocking=True)

        output = model.forward(image)
        
        if output.dim() > 1:
            output = torch.softmax(output, dim=-1)
        
        pred_label = output.argmax(dim=-1) if output.dim() > 1 else (output > 0.5).int()

        acc = self.compute_accuracy(true_label, pred_label)
        auc = self.compute_auc(true_label, output)

        self.metric_logger.meters["accuracy"].update(acc, n=1)
        self.metric_logger.meters["auc"].update(auc, n=1)

        if self.args.save_pred:
            self.records.append({
                "image_path": image_path,
                "true_label": true_label,
                "pred_label": pred_label.item(),
                "pred_score": output.max().item() if output.dim() > 1 else output.item(),
            })

    def compute_accuracy(self, true_label, pred_label):
        """Compute accuracy for classification."""
        if true_label.ndimension() > 1:
            true_label = true_label.argmax(dim=-1)
        return (true_label == pred_label).float().mean().item()

    def compute_auc(self, true_label, output):
        """Compute AUC (Area Under the Curve) for multi-class or binary classification."""
        # For multi-class, we calculate the AUC per class
        if self.task == "multi-class" or self.task == "binary-class":
            return roc_auc_score(true_label.cpu().numpy(), output.cpu().numpy(), multi_class='ovr' if output.shape[1] > 2 else 'raise')
        else:
            # Handle other cases (e.g., multi-label classification)
            raise NotImplementedError("AUC computation for multi-label classification is not yet implemented.")
    
    def save(self, path, model):
        """Save evaluation results to CSV and predictions to JSON."""
        info = {
            "model": [model.name],
            "task": [self.task],
            "dataset": [self.dataset.name],
            "model_type": [model.model_type],
            "modality": [self.dataset.modality],
            "size": [len(self.dataset)],
        }
        info = {**info, **{k: [meter.global_avg] for k, meter in self.metric_logger.meters.items()}}

        df = pd.DataFrame(info)
        df.to_csv(os.path.join(path, "results.csv"), index=False)

        if self.args.save_pred:
            with open(os.path.join(path, "predictions.json"), "w") as fp:
                json.dump(self.records, fp, indent=4)
