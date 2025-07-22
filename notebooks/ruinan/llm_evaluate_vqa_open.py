import argparse
import tqdm
import numpy as np
import os
import requests
import json
import pandas as pd
import time
import concurrent.futures

# --- Configuration ---
# System prompt for the AI model
prompt_rule = "We would like to request your feedback on the performance of the AI assistant in response to the user question displayed above, with reference to the provided ground truth answer. Please rate the helpfulness, relevance, accuracy, and level of detail of the assistant's response. Assign an overall score on a scale of 1 to 100, where a higher score indicates better overall performance. Please first output a single line containing only the score (a single numeric value). In the subsequent line, please provide a comprehensive explanation of your evaluation, referencing the ground truth answer to justify your score. Ensure your judgment is unbiased and objective."

# Constants for API requests
NUM_SECONDS_TO_SLEEP = 3
MAX_REQUEST_NUM = 10
MAX_WORKERS = 10

# --- API Interaction Functions ---

def get_gpt4_response(dialogue_history, instruction=""):
    """
    Sends a request to the GPT-4 model via OpenRouter and returns the response.

    Args:
        dialogue_history (list): A list of message dictionaries representing the conversation.
        instruction (str): A system-level instruction for the AI model.

    Returns:
        str or None: The content of the model's response, or None if the request fails.
    """
    prompt = [{"role": "system", "content": instruction}]
    prompt.extend(dialogue_history)

    for try_count in range(MAX_REQUEST_NUM):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": "Bearer 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",
                },
                data=json.dumps({"model": "openai/gpt-4o-mini", "messages": prompt}),
                timeout=30 # Adding a timeout for the request
            )
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            content = response.json()["choices"][0]["message"]["content"]
            # Basic validation of the score format
            float(content.split("\n")[0])
            return content
        except (requests.exceptions.RequestException, KeyError, IndexError, ValueError) as e:
            print(f"Attempt {try_count + 1} failed with error: {e}")
            if try_count < MAX_REQUEST_NUM - 1:
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:
                print("Maximum number of retries reached. Request failed.")
                return None
    return None


# Note: The get_ds_response function was in the original script but not used.
# I am keeping it here in case it's needed for future use.
def get_ds_response(dialogue_history, instruction=""):
    """
    Sends a request to the DeepSeek model via OpenRouter and returns the response.
    This function is currently not used in the main script.
    """
    prompt = [{"role": "system", "content": instruction}]
    prompt.extend(dialogue_history)

    for _ in range(MAX_REQUEST_NUM):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": "Bearer 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",
                },
                data=json.dumps({"model": "deepseek/deepseek-r1:free", "messages": prompt}),
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except (requests.exceptions.RequestException, KeyError, IndexError) as e:
            print(f"Request failed with error: {e}")
            time.sleep(NUM_SECONDS_TO_SLEEP)
    return None

# --- Data Processing Function for Parallel Execution ---

def process_item(item):
    """
    Processes a single item from the dataset to get a score and review.

    Args:
        item (dict): A dictionary containing question, answer, and prediction data.

    Returns:
        dict or None: The updated item dictionary with score and review, or None if processing fails.
    """
    if item.get("question_type") != "open":
        return item

    if item.get("prediction") == item.get("answer"):
        item["score"] = "100"
        item["review"] = "exact match"
        return item

    query = (
        f"[Question]\n{item['qs']}\n\n"
        f"[True Answer]\n{item['answer']}\n\n[End of True Answer]\n\n"
        f"[Prediction]\n{item['prediction']}\n\n[End of prediction]\n\n"
        f"[System]\n{prompt_rule}\n\n"
    )

    instruction = "You are a helpful and precise assistant for checking the quality of the answer of medical VQA task"
    dialogue_history = [{"role": "user", "content": query}]

    response = get_gpt4_response(dialogue_history, instruction)

    if response:
        try:
            parts = response.split("\n")
            score = parts[0].strip()
            reason = parts[-1].strip()
            # Ensure score is a valid number before assigning
            float(score)
            item["score"] = score
            item["review"] = reason
            return item
        except (ValueError, IndexError) as e:
            print(f"Error processing response: '{response}'. Error: {e}")
            return None
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel DeepSeek-based VQA evaluation.")
    parser.add_argument("-f", "--folder", required=True, help="Folder containing predictions.json")
    args = parser.parse_args()

    input_file = os.path.join(args.folder, "predictions.json")
    output_file = os.path.join(args.folder, "deekseep_review.json")
    score_file = os.path.join(args.folder, "gpt_score.csv")

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        exit()

    with open(input_file, "r") as file:
        data = json.load(file)

    # Add an index to each item for tracking
    for i, item in enumerate(data):
        item["index"] = i

    new_data = []
    # Using ThreadPoolExecutor to process items in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Wrap executor.map with tqdm for a progress bar
        future_to_item = {executor.submit(process_item, item): item for item in data}
        results = []
        for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_item), total=len(data), desc="Processing items"):
            result = future.result()
            if result:
                results.append(result)

    # Sort results by the original index to maintain order
    results.sort(key=lambda x: x.get('index', float('inf')))

    # Save the processed data
    with open(output_file, "w") as fp:
        json.dump(results, fp, indent=4)
        print(f"\nSaved detailed reviews to {output_file}")

    # Calculate and report the average score
    scores = [float(item["score"]) for item in results if "score" in item and item.get("question_type") == "open"]
    if scores:
        avg_score = np.mean(scores)
        print(f"\nAverage score for all open questions: {avg_score:.2f}. Folder: {args.folder}")

        df = pd.DataFrame.from_dict({"score": [avg_score]})
        df.to_csv(score_file, index=False)
        print(f"Saved average score to {score_file}")
    else:
        print("\nNo scores were calculated.")

