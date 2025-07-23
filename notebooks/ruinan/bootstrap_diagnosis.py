#!/usr/bin/env python
"""
Estimate AUROC and its 95 % bootstrap CI from a predictions.json file.
The JSON is expected to be a list of records, each with:
  - "true_label" : 0 or 1
  - "pred_score" : model‑estimated probability or score for the positive class
"""

import argparse
import json
import numpy as np
from sklearn.metrics import roc_auc_score


def load_predictions(path):
    """Return y_true, y_score from a MedVLMBench-style JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    y_true = np.array([row["true_label"] for row in data], dtype=np.int8)
    y_score = np.array([row["pred_score"] for row in data], dtype=np.float32)
    return y_true, y_score


def bootstrap_ci(y_true, y_score, n_bootstraps=1000, seed=42):
    """
    Bootstrap AUROC.
    Returns: mean_bootstrap_auc, (ci_low, ci_high), bootstrapped_scores
    """
    rng = np.random.default_rng(seed)
    boot_scores = []

    n = len(y_true)
    for _ in range(n_bootstraps):
        indices = rng.integers(0, n, n)          # sample with replacement
        if len(np.unique(y_true[indices])) < 2:  # skip if all same class
            continue
        score = roc_auc_score(y_true[indices], y_score[indices])
        boot_scores.append(score)

    boot_scores = np.array(boot_scores)
    mean_score = boot_scores.mean()
    ci_low, ci_high = np.percentile(boot_scores, [2.5, 97.5])
    return mean_score, (ci_low, ci_high), boot_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", help="Path to predictions.json")
    parser.add_argument("-n", "--n_bootstraps", type=int, default=1000,
                        help="Number of bootstrap resamples (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    y_true, y_score = load_predictions(args.json_path)

    # Point estimate on full test set
    auroc = roc_auc_score(y_true, y_score)

    # Bootstrap
    mean_boot, (ci_low, ci_high), _ = bootstrap_ci(
        y_true, y_score,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed
    )

    print(f"Test AUROC         : {auroc:.4f}")
    print(f"Bootstrap mean     : {mean_boot:.4f}")
    print(f"95% CI (percentile): [{ci_low:.4f}, {ci_high:.4f}]  "
          f"({args.n_bootstraps} resamples)")


if __name__ == "__main__":
    main()