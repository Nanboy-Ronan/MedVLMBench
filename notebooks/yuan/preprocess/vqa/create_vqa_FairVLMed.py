import requests
import json
import pandas as pd
import time
import tqdm
import os
import re

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
            assert "Maximum try"
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": "Bearer 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",
                },
                data=json.dumps({"model": "openai/gpt-4o-mini", "messages": prompt}),  # Optional
            )
            assert response.status_code == 200
            content = response.json()["choices"][0]["message"]["content"]
            extracted = extract_questions_and_answers(content)

            assert len(extracted) == 4

            break
        except Exception as e:
            print(e)
        try_ += 1
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response.json()["choices"][0]["message"]["content"]


def extract_questions_and_answers(input_string):
    # Define regex patterns for extracting questions and answers
    patterns = {
        "Open Question": r"\[Open Question\]: (.+)",
        "Open Answer": r"\[Open Answer\]: (.+)",
        "Closed Question": r"\[Closed Question\]: (.+)",
        "Closed Answer": r"\[Closed Answer\]: (.+)",
    }

    # Dictionary to store the extracted contents
    extracted = {}

    # Extract contents using regex
    for key, pattern in patterns.items():
        match = re.search(pattern, input_string)
        if match:
            extracted[key] = match.group(1).strip()
        else:
            extracted[key] = None  # Handle missing entries gracefully

    return extracted


prompt = """You are tasked with generating questions and answers for a Visual Question Answering (VQA) dataset based on medical notes associated with SLO fundus images. For each image paired with medical notes, you will create one open-ended question and one closed (yes/no) question. Follow these instructions carefully:

Use the provided medical notes as context to design both questions.
Ensure the open-ended question requires a descriptive answer based on the notes. The question and answer to it should be consise (within 20 words).
Ensure the closed question requires a simple yes or no answer, clearly derived from the notes.
It is important to remember that you are designing question for VQA. So the questions should be able to answer from SLO fundus image only, but not based on other information such as medical history or results of other tests which may appear in the notes. Remember when people are trying to answer the questions you provide, they do not have access to any other information except for the SLO fundus image.
Your response must strictly follow this format (no additional text):

[Open Question]: <Your open-ended question here>
[Open Answer]: <Your descriptive answer here>
[Closed Question]: <Your yes/no question here>
[Closed Answer]: <Yes or No>"""


instruction = "You are a helpful and precise assistant for generating question and answers for medical VQA datasets based on the medical notes"


if __name__ == "__main__":
    root = "/research/d5/gds/yzhong22/datasets/Harvard-FairVLMed10k"

    df = pd.read_csv(os.path.join(root, "caption_all.csv"))
    writing_freq = 500

    dict_out = []

    for i in tqdm.tqdm(range(len(df))):
        item = df.iloc[i]

        report = item["gpt_summary"]
        file_name = item["filename"]
        img_path = item["image_path"]

        query = f"Now, use the following medical notes to generate questions and answers:\n\nMedical Notes:\n{report}"

        dialogue_history = [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "system",
                "content": query,
            },
        ]

        response = get_gpt4_response(dialogue_history, instruction)
        extracted = extract_questions_and_answers(response)

        sample_open = {
            "gpt_summary": report,
            "filename": file_name,
            "image_path": img_path,
            "question type": "OPEN",
            "question": extracted["Open Question"],
            "answer": extracted["Open Answer"],
        }

        sample_closed = {
            "gpt_summary": report,
            "filename": file_name,
            "image_path": img_path,
            "question type": "CLOSED",
            "question": extracted["Closed Question"],
            "answer": extracted["Closed Answer"],
        }

        dict_out.extend([sample_open, sample_closed])

        if i % writing_freq == 0:
            df_out = pd.DataFrame.from_dict(dict_out)
            df_out.to_csv(os.path.join(root, "vqa_all.csv"), index=False)

    df_out = pd.DataFrame.from_dict(dict_out)
    df_out.to_csv(os.path.join(root, "vqa_all.csv"), index=False)
