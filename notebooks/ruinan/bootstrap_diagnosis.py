import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

# ---------- configuration ----------
JSON_PATH      = "/bigdata/rjin02/MedVLMBench/log/diagnosis/Camelyon17/CLIP/train_lp_seed42/predictions.json"
N_BOOTSTRAPS   = 1_000
RANDOM_SEED    = 42
# -----------------------------------

def load_predictions(json_path: Path):
    """Returns two flat numpy arrays: y_true, y_score."""
    with open(json_path) as f:
        batches = json.load(f)

    y_true  = np.concatenate([np.asarray(b["true_label"])  for b in batches])
    y_score = np.concatenate([np.asarray(b["pred_score"])  for b in batches])
    return y_true, y_score


def bootstrap_auc_ci(y_true, y_score, *, n_iterations=1000, seed=None, ci=0.95):
    """
    Bootstraps AUROC and returns (lower, upper) CI bounds.
    Skips any resample that contains only one class.
    """
    rng  = np.random.default_rng(seed)
    n    = len(y_true)
    aucs = []

    for _ in range(n_iterations):
        idx = rng.integers(0, n, n)       # sample with replacement
        if np.unique(y_true[idx]).size < 2:
            continue                      # AUROC undefined if only one class present
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))

    lower = np.percentile(aucs, (1 - ci) / 2 * 100)
    upper = np.percentile(aucs, (1 + ci) / 2 * 100)
    return np.asarray(aucs), (lower, upper)


if __name__ == "__main__":
    y_true, y_score = load_predictions(JSON_PATH)

    # 1. point estimate
    point_auc = roc_auc_score(y_true, y_score)

    # 2. bootstrap CI
    auc_samples, (ci_low, ci_high) = bootstrap_auc_ci(
        y_true, y_score,
        n_iterations=N_BOOTSTRAPS,
        seed=RANDOM_SEED,
    )

    print(f"AUROC: {point_auc:.4f}")
    print(f"{(0.95*100):.0f}% bootstrap CI: [{ci_low:.4f}, {ci_high:.4f}]")
