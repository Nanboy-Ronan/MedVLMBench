import os
import warnings
import numpy as np
import pandas as pd
import torch
from collections import namedtuple
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from eval.base import EvalEngine
from dataset.utils import LinearProbingDataCollator

Metrics = namedtuple("Metrics", ["AUC", "ACC"])


class DiagnosisEvalEngine(EvalEngine):
    # TODO: support for multi-class
    def __init__(self, args, dataset, logger, task="binary-class", device="cuda"):
        super().__init__(args, dataset, logger)

        # self.task = task  # e.g., "multi-class", "binary-class", "multi-label"
        self.task = "multi-class"  # e.g., "multi-class", "binary-class", "multi-label"
        self.device = device 

        self.metrics = {
            "accuracy": 0.0,
            "auc": 0.0,
        }

    def evaluate(self, args, model):
        """Run evaluation on the classification dataset."""
        args.logger.info("Length of the evaluation dataset: {}".format(len(self.dataset)))
        data_loader = DataLoader(self.dataset, batch_size=64, collate_fn=LinearProbingDataCollator(), shuffle=False)
        self.num_classes = model.num_classes
        self.init_metric_logger()
        self.all_true_labels = []
        self.all_outputs = []

        with torch.no_grad():
            for batch in self.metric_logger.log_every(data_loader, self.args.eval_print_freq, header="Test:"):
                batch_true_labels, batch_outputs, acc = self.evaluate_subject(batch, model)
                self.all_true_labels.append(batch_true_labels)
                self.all_outputs.append(batch_outputs)
                
                self.metric_logger.meters["accuracy"].update(acc, n=1)

        all_true_labels_np = np.concatenate(self.all_true_labels, axis=0)
        all_outputs_np = np.concatenate(self.all_outputs, axis=0)
        auc = self.compute_auc(all_true_labels_np, all_outputs_np)
        self.metric_logger.meters["auc"].update(auc, n=1)

        self.metric_logger.synchronize_between_processes()
        

        results = {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}
        self.logger.info("\nEvaluation results:\n" + "\n".join(f"{k} {v:.8f}" for k, v in results.items()))

        return results

    def evaluate_subject(self, batch, model):
        """Evaluate a single batch (image and label)."""
        image = batch["pixel_values"]
        true_label = batch["labels"]
        image = model.image_processor_evaluation(image)

        image = image.to(self.device, non_blocking=True)
        model.to(self.device)
        true_label = true_label.to(self.device)

        output = model(image)
        
        if output.dim() > 1:
            output = torch.softmax(output, dim=-1)
        
        pred_label = output.argmax(dim=-1) if output.dim() > 1 else (output > 0.5).int()
        acc = self.compute_accuracy(true_label, pred_label)

        true_label_np = true_label.detach().cpu().numpy()
        output_np = output.detach().cpu().numpy()

        self.metric_logger.meters["accuracy"].update(acc, n=1)

        if self.args.save_pred:
            self.records.append({
                "image_path": image_path,
                "true_label": true_label,
                "pred_label": pred_label.item(),
                "pred_score": output.max().item() if output.dim() > 1 else output.item(),
            })
        
        if self.task == "binary-class" and output_np.ndim == 1:
            output_np = np.vstack([1 - output_np, output_np]).T  # Convert to two columns

        return true_label_np, output_np, acc

    def compute_accuracy(self, true_label, pred_label):
        """Compute accuracy for classification."""
        if true_label.ndimension() > 1:
            true_label = true_label.argmax(dim=-1)
        return (true_label == pred_label).float().mean().item()

    def compute_auc(self, true_label_np, output_np):
        """Compute AUC (Area Under the Curve) for multi-class or binary classification."""
        
        if self.task == "binary-class":
            # Ensure true_label is 1D
            if true_label_np.ndim > 1 and true_label_np.shape[1] == 2:
                # If labels are one-hot encoded, convert to single class labels
                true_label_np = true_label_np.argmax(axis=1)
            elif true_label_np.ndim > 1:
                raise ValueError(
                    "For binary classification, true_label should be 1D or one-hot encoded with 2 classes."
                )

            # For binary classification, use the probability of the positive class
            if output_np.shape[1] != 2:
                raise ValueError(
                    f"For binary classification, output should have 2 columns (found {output_np.shape[1]})."
                )
            y_score = output_np[:, 1]
            return roc_auc_score(true_label_np, y_score)
        
        elif self.task == "multi-class":
            # Convert true labels to one-hot encoding to ensure all classes are represented
            lb = LabelBinarizer()
            lb.fit(range(self.num_classes))  # Ensure all classes are considered
            y_true_binarized = lb.transform(true_label_np)
            
            # Handle binary case in multi-class if necessary
            if y_true_binarized.shape[1] == 1:
                y_true_binarized = np.hstack((1 - y_true_binarized, y_true_binarized))
            
            if y_true_binarized.shape[1] != self.num_classes:
                raise ValueError(
                    f"Number of classes in y_true ({y_true_binarized.shape[1]}) does not match "
                    f"the expected number of classes ({self.num_classes})."
                )
            
            if output_np.shape[1] != self.num_classes:
                raise ValueError(
                    f"Number of classes in y_score ({output_np.shape[1]}) does not match "
                    f"the expected number of classes ({self.num_classes})."
                )

            return roc_auc_score(y_true_binarized, output_np, multi_class='ovr')
        
        else:
            raise NotImplementedError(
                "AUC computation for multi-label classification is not yet implemented."
            )
    
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
