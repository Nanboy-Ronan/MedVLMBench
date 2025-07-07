import argparse
import tqdm
import numpy as np
import os
import requests
import json
import pandas as pd
import time

prompt_rule = "We would like to request your feedback on the performance of the AI assistant in response to the user question displayed above, with reference to the provided ground truth answer. Please rate the helpfulness, relevance, accuracy, and level of detail of the assistant's response. Assign an overall score on a scale of 1 to 100, where a higher score indicates better overall performance. Please first output a single line containing only the score (a single numeric value). In the subsequent line, please provide a comprehensive explanation of your evaluation, referencing the ground truth answer to justify your score. Ensure your judgment is unbiased and objective."

NUM_SECONDS_TO_SLEEP = 3
MAX_REQUEST_NUM = 10


def get_gpt4_response(dialogue_history, instruction=""):
    prompt = [
        {
            "role": "system",
            "content": instruction,
        }
    ]

    for entry in dialogue_history:
        prompt.append({"role": entry["role"], "content": entry["content"]})

    try_ = 0
    while True:
        if try_ >= MAX_REQUEST_NUM:
            return None
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": "Bearer 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",
                },
                data=json.dumps({"model": "openai/gpt-4o-mini", "messages": prompt}),  # Optional
            )
            assert response.status_code == 200
            score = response.json()["choices"][0]["message"]["content"].split("\n")[0]
            score = float(score)

            break
        except Exception as e:
            print(e)
        try_ += 1
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response.json()["choices"][0]["message"]["content"]


def get_ds_response(dialogue_history, instruction=""):
    while True:
        try:
            prompt = [
                {
                    "role": "system",
                    "content": instruction,
                }
            ]

            for entry in dialogue_history:
                prompt.append({"role": entry["role"], "content": entry["content"]})

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": "Bearer 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",
                },
                data=json.dumps({"model": "deepseek/deepseek-r1:free", "messages": prompt}),  # Optional
            )

            break
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek-based VQA evaluation.")
    parser.add_argument("-f", "--folder")
    args = parser.parse_args()

    with open(os.path.join(args.folder, "predictions.json"), "r") as file:
        data = json.load(file)

    scores = []
    new_data = []

    for i, item in tqdm.tqdm(enumerate(data)):
        item["index"] = i
        if item["question_type"] != "open":
            pass

        else:

            qs = item["qs"]
            answer = item["answer"]
            pred = item["prediction"]

            if pred == answer:
                score, reason = "100", "exect match"
            else:
                query = (
                    f"[Question]\n{qs}\n\n"
                    f"[True Answer]\n{answer}\n\n[End of True Answer]\n\n"
                    f"[Prediction]\n{pred}\n\n[End of prediction]\n\n"
                    f"[System]\n{prompt_rule}\n\n"
                )

                instruction = "You are a helpful and precise assistant for checking the quality of the answer of medical VQA task"

                dialogue_history = [
                    {
                        "role": "system",
                        "content": query,
                    }
                ]

                response = get_gpt4_response(dialogue_history, instruction)

                if response is None:
                    continue

                score, reason = response.split("\n")[0], response.split("\n")[-1]
            score = score.strip()
            reason = reason.strip()

            item["score"] = score
            item["review"] = reason

            scores.append(float(score))
        new_data.append(item)

        with open(os.path.join(args.folder, "deekseep_review.json"), "w") as fp:
            json.dump(new_data, fp, indent=4)

    avg_score = np.mean(scores)

    print(f"Average score for all open questions: {avg_score}. Folder: {args.folder}")

    df = pd.DataFrame.from_dict({"score": [avg_score]})
    df.to_csv(os.path.join(args.folder, "gpt_score.csv"), index=False)
