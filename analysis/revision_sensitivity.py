#!/usr/bin/env python3
"""Standalone sensitivity analyses requested during major revision.

This script intentionally operates on the published summary-result CSVs. It does
not describe the item bootstrap as an LMM bootstrap. Item bootstrapping is used
for per-model/dataset metric uncertainty; LMM Wald intervals and P values are
computed from the fitted mixed model itself.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]


def load_results(diagnosis_csv: Path, vqa_csv: Path) -> pd.DataFrame:
    diagnosis = pd.read_csv(diagnosis_csv)
    diagnosis = diagnosis.rename(columns={"strategy": "strategy_raw", "auroc": "outcome"})
    diagnosis["task"] = "diagnosis"
    diagnosis["family"] = diagnosis["model_family"]
    diagnosis["adaptation"] = np.where(diagnosis["strategy_raw"].eq("ZS"), "off_the_shelf", "adapted")

    vqa = pd.read_csv(vqa_csv)
    vqa = vqa.rename(columns={"trainable_module": "strategy_raw", "acc": "outcome"})
    vqa["task"] = "vqa"
    vqa["family"] = vqa["model_family"]
    vqa["adaptation"] = np.where(vqa["strategy_raw"].eq("ZS"), "off_the_shelf", "adapted")

    common = ["task", "dataset", "model", "model_type", "family", "strategy_raw", "adaptation", "outcome"]
    result = pd.concat([diagnosis[common], vqa[common]], ignore_index=True)
    result["model_type"] = result["model_type"].replace({"general": "generalist", "medical": "medical"})
    return result.dropna(subset=["outcome"]).reset_index(drop=True)


def fit_lmm(frame: pd.DataFrame, formula: str):
    data = frame.copy()
    data["dataset"] = data["dataset"].astype(str)
    data["family"] = data["family"].astype(str)
    data["model"] = data["model"].astype(str)
    specifications = [
        (
            "dataset + family + model nested in family",
            lambda: smf.mixedlm(
                formula,
                data,
                groups=np.ones(len(data)),
                re_formula="0",
                vc_formula={
                    "dataset": "0 + C(dataset)",
                    "family": "0 + C(family)",
                    "model_in_family": "0 + C(family):C(model)",
                },
            ),
        ),
        (
            "dataset + model",
            lambda: smf.mixedlm(
                formula,
                data,
                groups=np.ones(len(data)),
                re_formula="0",
                vc_formula={"dataset": "0 + C(dataset)", "model": "0 + C(model)"},
            ),
        ),
        ("dataset random intercept", lambda: smf.mixedlm(formula, data, groups=data["dataset"])),
    ]
    attempts = []
    for random_effects, build_model in specifications:
        for method in ("lbfgs", "powell", "cg"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    result = build_model().fit(reml=True, method=method, maxiter=2000)
                except Exception as exc:
                    attempts.append(f"{random_effects}/{method}: {type(exc).__name__}: {exc}")
                    continue
            fit_warnings = [str(item.message) for item in caught]
            if result.converged:
                return result, random_effects, method, fit_warnings, attempts
            attempts.append(f"{random_effects}/{method}: did not converge")
    raise RuntimeError("No mixed-effects specification converged: " + " | ".join(attempts))


def tidy_fit(name: str, task: str, data: pd.DataFrame, formula: str) -> tuple[pd.DataFrame, dict]:
    result, random_effects, optimizer, fit_warnings, failed_attempts = fit_lmm(data, formula)
    intervals = result.conf_int()
    rows = []
    for term, estimate in result.fe_params.items():
        rows.append(
            {
                "analysis": name,
                "task": task,
                "term": term,
                "estimate": estimate,
                "ci_low": intervals.loc[term, 0],
                "ci_high": intervals.loc[term, 1],
                "p_value": result.pvalues[term],
                "n_observations": len(data),
                "n_datasets": data["dataset"].nunique(),
                "n_families": data["family"].nunique(),
                "n_models": data["model"].nunique(),
            }
        )
    details = {
        "analysis": name,
        "task": task,
        "formula": formula,
        "random_effects": random_effects,
        "optimizer": optimizer,
        "converged": bool(result.converged),
        "reml": True,
        "wald_intervals_and_p_values": True,
        "variance_components": [float(value) for value in np.atleast_1d(result.vcomp)],
        "fit_warnings": fit_warnings,
        "failed_attempts": failed_attempts,
    }
    return pd.DataFrame(rows), details


def strict_models(pair_file: Path, task: str) -> set[str]:
    pairs = pd.read_csv(pair_file)
    pairs = pairs[(pairs["task"] == task) & (pairs["strict_match"] == "yes")]
    return set(pairs["general_model"]) | set(pairs["medical_model"])


def strict_pair_differences(data: pd.DataFrame, pair_file: Path, task: str) -> pd.DataFrame:
    pairs = pd.read_csv(pair_file)
    pairs = pairs[(pairs["task"] == task) & (pairs["strict_match"] == "yes")]
    rows = []
    for pair in pairs.itertuples(index=False):
        general = data[data["model"] == pair.general_model][["dataset", "outcome"]]
        medical = data[data["model"] == pair.medical_model][["dataset", "outcome"]]
        matched = general.merge(medical, on="dataset", suffixes=("_general", "_medical"))
        for row in matched.itertuples(index=False):
            rows.append(
                {
                    "task": task,
                    "dataset": row.dataset,
                    "family": pair.pair_id,
                    "model": pair.pair_id,
                    "outcome": row.outcome_general - row.outcome_medical,
                }
            )
    return pd.DataFrame(rows)


def cross_domain_drops(results: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    medical = results[(results["model_type"] == "medical") & (results["adaptation"] == "adapted")]
    certain = exposure[exposure["exposure_status"].isin({"documented_direct", "modality_level", "no_reported"})]
    merged = medical.merge(
        certain[["task", "model", "dataset", "exposure_status"]],
        on=["task", "model", "dataset"],
        how="inner",
    )
    merged["domain_group"] = np.where(
        merged["exposure_status"].eq("no_reported"), "cross_domain", "in_domain"
    )
    rows = []
    for (task, model, strategy), group in merged.groupby(["task", "model", "strategy_raw"]):
        means = group.groupby("domain_group")["outcome"].agg(["mean", "count"])
        if {"in_domain", "cross_domain"}.issubset(means.index):
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "adaptation_strategy": strategy,
                    "in_domain_mean": means.loc["in_domain", "mean"],
                    "cross_domain_mean": means.loc["cross_domain", "mean"],
                    "cross_minus_in_domain": means.loc["cross_domain", "mean"]
                    - means.loc["in_domain", "mean"],
                    "n_in_domain": int(means.loc["in_domain", "count"]),
                    "n_cross_domain": int(means.loc["cross_domain", "count"]),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "task",
            "model",
            "adaptation_strategy",
            "in_domain_mean",
            "cross_domain_mean",
            "cross_minus_in_domain",
            "n_in_domain",
            "n_cross_domain",
        ],
    )


def run_analyses(results, pair_file, exposure_file):
    proprietary = {"o3", "gemini-2.5-pro"}
    results = results[~results["model"].isin(proprietary)].copy()
    exposure = pd.read_csv(exposure_file)
    tables, diagnostics = [], []

    model_type = "C(model_type, Treatment('medical'))"
    adaptation = "C(adaptation, Treatment('off_the_shelf'))"

    for task in ("diagnosis", "vqa"):
        task_data = results[results["task"] == task].copy()

        symmetric = task_data[task_data["strategy_raw"].isin({"ZS", "FT-LP", "ML"})]
        table, detail = tidy_fit(
            "symmetric_model_type_and_adaptation",
            task,
            symmetric,
            f"outcome ~ {model_type} * {adaptation}",
        )
        tables.append(table)
        diagnostics.append(detail)

        adapted = symmetric[symmetric["adaptation"] == "adapted"]
        table, detail = tidy_fit(
            "rq2_symmetric_adapted_models",
            task,
            adapted,
            f"outcome ~ {model_type}",
        )
        tables.append(table)
        diagnostics.append(detail)

        ots = task_data[task_data["adaptation"] == "off_the_shelf"].copy()
        table, detail = tidy_fit("rq1_all_available_models", task, ots, f"outcome ~ {model_type}")
        tables.append(table)
        diagnostics.append(detail)

        strict = strict_pair_differences(ots, pair_file, task)
        table, detail = tidy_fit("rq1_strict_pair_general_minus_medical", task, strict, "outcome ~ 1")
        tables.append(table)
        diagnostics.append(detail)

        direct = exposure[
            (exposure["task"] == task) & (exposure["exclude_from_no_direct_overlap"] == "yes")
        ][["model", "dataset"]]
        no_direct = ots.merge(direct.assign(_direct=True), on=["model", "dataset"], how="left")
        no_direct = no_direct[no_direct["_direct"].isna()].drop(columns="_direct")
        table, detail = tidy_fit("rq1_excluding_documented_direct_exposure", task, no_direct, f"outcome ~ {model_type}")
        tables.append(table)
        diagnostics.append(detail)

    return pd.concat(tables, ignore_index=True), diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnosis-csv",
        type=Path,
        default=ROOT / "notebooks/yuan/plot_v2/results/boot_results_diagnosis.csv",
    )
    parser.add_argument(
        "--vqa-csv", type=Path, default=ROOT / "notebooks/yuan/plot_v2/results/boot_results_vqa.csv"
    )
    parser.add_argument("--model-pairs", type=Path, default=ROOT / "metadata/model_pairs.csv")
    parser.add_argument(
        "--exposure-matrix", type=Path, default=ROOT / "metadata/model_dataset_exposure.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/revision_outputs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(args.diagnosis_csv, args.vqa_csv)
    estimates, diagnostics = run_analyses(results, args.model_pairs, args.exposure_matrix)
    estimates.to_csv(args.output_dir / "sensitivity_estimates.csv", index=False)
    exposure = pd.read_csv(args.exposure_matrix)
    cross_domain_drops(results, exposure).to_csv(args.output_dir / "cross_domain_drops.csv", index=False)
    with open(args.output_dir / "model_diagnostics.json", "w") as stream:
        json.dump(diagnostics, stream, indent=2)
    print(estimates.to_string(index=False))


if __name__ == "__main__":
    main()
