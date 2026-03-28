from pathlib import Path
import sys

sys.path.append("../../../")

import os
import pandas as pd
from eval.utils import normalize_word
from eval.metrics import (
    calculate_exactmatch,
    calculate_f1score,
    calculate_appearance_with_normalization,
    calculate_bertscore,
    calculate_meteor,
)
from torchmetrics.functional.text import bleu_score, rouge_score

import tqdm
import json
import numpy as np


open_metrics = [
    "bleu1",
    "bleu2",
    "bleu3",
    "bleu4",
    "rouge1",
    "rouge2",
    "rougeL",
    "exact_match",
    "recall",
    "precision",
    "f1_score",
    "accuracy",
    "bertscore",
    "meteor",
    "gpt_score",
]


closed_metrics = [
    "exact_match",
    "recall",
    "precision",
    "f1_score",
    "accuracy",
]

overall_metrics = ["exact_match", "recall", "precision", "f1_score", "gpt_score"]


all_metrics = (
    [f"{x}_open" for x in open_metrics]
    + [f"{x}_closed" for x in closed_metrics]
    + [f"{x}_overall" for x in overall_metrics]
)


def process_tokens(text):
    tokenized_text = set(text.split())
    tokenized_text.discard("")
    return tokenized_text


if __name__ == "__main__":
    exp_root = "/media/yesindeed/DATADRIVE1/mount/remote_cse/experiments/med_vlm_benchmark/merged"

    datasets = {
        "SLAKE": "/research/d5/gds/yzhong22/datasets/SLAKE/imgs",
        "PathVQA": "None",
        "VQA-RAD": "None",
        "Harvard-FairVLMed10k": "/research/d5/gds/yzhong22/datasets/Harvard-FairVLMed10k",
    }

    df_index = pd.read_csv(os.path.join(exp_root, "exp_status.csv"))

    for i in tqdm.tqdm(range(len(df_index))):
        item = df_index.iloc[i]

        path = item["path"]
        save_file = os.path.join(exp_root, path, "each_example_result.csv")

        if os.path.exists(save_file):
            continue

        if not item["have_prediction"]:
            continue

        all_results = []

        with open(os.path.join(exp_root, path, "predictions.json"), "r") as file:
            # Load the JSON data from the file
            predictions = json.load(file)

        if item["have_gpt_score"]:
            with open(os.path.join(exp_root, path, "deekseep_review.json"), "r") as file:
                # Load the JSON data from the file
                gpt_scores = json.load(file)

            assert len(predictions) == len(gpt_scores)
        else:
            gpt_scores = None

        for i_case, pred in tqdm.tqdm(enumerate(predictions)):
            output = pred["prediction"]
            answer = pred["answer"]
            question_type = pred["question_type"]

            output_l, answer_l = output.lower(), answer.lower()

            output_normed = normalize_word(output_l)
            answer_normed = normalize_word(answer_l)

            case_dict = {"question_type": question_type}
            for k in all_metrics:
                case_dict[k] = np.nan

            f1_score, precision, recall = calculate_f1score(output_l, answer_l)
            exact_match = calculate_exactmatch(output_l, answer_l)

            if question_type == "open":
                bleu1 = bleu_score([output_normed], [[answer_normed]], n_gram=1).item()
                bleu2 = bleu_score([output_normed], [[answer_normed]], n_gram=2).item()
                bleu3 = bleu_score([output_normed], [[answer_normed]], n_gram=3).item()
                bleu4 = bleu_score([output_normed], [[answer_normed]], n_gram=4).item()
                rouge_scores = rouge_score(output_normed, answer_normed)
                rouge1, rouge2, rougeL = (
                    rouge_scores["rouge1_fmeasure"].item(),
                    rouge_scores["rouge2_fmeasure"].item(),
                    rouge_scores["rougeL_fmeasure"].item(),
                )
                # accuracy = calculate_appearance_with_normalization(output_l, answer_l)
                accuracy = float(recall >= 0.75)
                bertscore = calculate_bertscore(output_normed, answer_normed)
                meteor = calculate_meteor(output_normed, answer_normed)

                if gpt_scores is not None:
                    gpt_score = gpt_scores[i_case]["score"]
                else:
                    gpt_score = np.nan

                for m in open_metrics:
                    case_dict[f"{m}_open"] = globals()[m]
            elif question_type == "closed":
                accuracy = 1 if answer_l in output_l else 0
                for m in closed_metrics:
                    case_dict[f"{m}_closed"] = globals()[m]
            else:
                raise Exception(f"Unknown question type {question_type}")

            for m in overall_metrics:
                case_dict[f"{m}_overall"] = globals()[m]

            if question_type == "open":
                for m in closed_metrics:
                    assert np.isnan(case_dict[f"{m}_closed"])
            else:
                for m in open_metrics:
                    assert np.isnan(case_dict[f"{m}_open"])
                for m in closed_metrics:
                    assert ~np.isnan(case_dict[f"{m}_closed"])

            all_results.append(case_dict)

        df_case_result = pd.DataFrame(all_results)
        # print(df_case_result)
        break
