import os
import warnings
import numpy as np
import pandas as pd
import torch
from collections import namedtuple
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from train.clip_trainer import LinearProbingDataCollator
from torchvision.transforms.functional import to_pil_image
from eval.base import EvalEngine

Metrics = namedtuple("Metrics", ["AUC", "ACC"])


class DiagnosisEvalEngine(EvalEngine):
    # TODO: support for multi-class
    def __init__(self, args, dataset, logger, task="binary-class", device="cuda"):
        super().__init__(args, dataset, logger)

        self.task = task  # e.g., "multi-class", "binary-class", "multi-label"
        self.device = device 

        self.metrics = {
            "accuracy": 0.0,
            "auc": 0.0,
        }

    def evaluate(self, args, model):
        """Run evaluation on the classification dataset."""
        data_loader = DataLoader(self.dataset, batch_size=64, collate_fn=LinearProbingDataCollator(), shuffle=False)
        
        self.init_metric_logger()

        with torch.no_grad():
            for batch in self.metric_logger.log_every(data_loader, self.args.eval_print_freq, header="Test:"):
                # batch = {k: v[0] for k, v in batch.items()}
                self.evaluate_subject(batch, model)

        self.metric_logger.synchronize_between_processes()

        results = {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}
        self.logger.info("\nEvaluation results:\n" + "\n".join(f"{k} {v:.3f}" for k, v in results.items()))

        return results

    def evaluate_subject(self, batch, model):
        """Evaluate a single batch (image and label)."""
        image = batch["pixel_values"]
        true_label = batch["labels"]
        # breakpoint()
        image = model.image_processor_evaluation(image)
        # image = torch.tensor(model.image_processor(image)["pixel_values"]) # BLIP

        # XGPT
        # image_batch_pil = [to_pil_image(img_tensor) for img_tensor in image]
        # image = [model.image_processor(pil_image) for pil_image in image_batch_pil]
        # image = torch.stack(image)

        image = image.to(self.device, non_blocking=True)
        true_label = true_label.to(self.device)

        output = model(image)
        
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
        true_label_np = true_label.cpu().numpy()
        output_np = output.cpu().numpy()

        if self.task == "binary-class":
            # Ensure true_label is 1D
            if true_label_np.ndim > 1 and true_label_np.shape[1] == 2:
                # If labels are one-hot encoded, convert to single class labels
                true_label_np = true_label_np.argmax(axis=1)
            elif true_label_np.ndim > 1:
                raise ValueError("For binary classification, true_label should be 1D or one-hot encoded with 2 classes.")

            # For binary classification, use the probability of the positive class
            y_score = output_np[:, 1]
            return roc_auc_score(true_label_np, y_score)
        elif self.task == "multi-class":
            # Ensure that y_true is a 1D array of class indices
            if true_label_np.ndim > 1 and true_label_np.shape[1] > 1:
                # If labels are one-hot encoded, convert to single class labels
                true_label_np = true_label_np.argmax(axis=1)
            elif true_label_np.ndim > 1:
                raise ValueError("For multi-class classification, true_label should be 1D or one-hot encoded.")

            return roc_auc_score(true_label_np, output_np, multi_class='ovr')
        else:
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
