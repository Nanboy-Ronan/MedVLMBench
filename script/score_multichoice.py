
"""
python script/score_multichoice.py \
    --predictions ./log/vqa/MedXpertQA/Gemma3/eval_seed0/gemma-3-4b-pt/predictions.json \
    --api-key xxxx \
    --output ./scored_predictions.json
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import google.generativeai as genai


PROMPT_TEMPLATE = """You are evaluating multiple-choice medical VQA answers.

Decide which option letter (A/B/C/D/E or more) the model intended to choose, using the
question and the model's raw output. When the model did not clearly choose an
option, pick the best option yourself.

Return ONLY the single letter A, B, C, D, E, or more. Do not include any other text.

Question with options:
{question}

Model raw output:
{raw_output}

Answer:"""


def parse_args():
    parser = argparse.ArgumentParser(description="Score MedXpertQA predictions with Gemini post-processing.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.json from evaluation.")
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY"), help="Gemini API key. Defaults to GEMINI_API_KEY env var.")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for Gemini.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per item on transient failures.")
    parser.add_argument("--retry-wait", type=float, default=2.0, help="Seconds to wait between retries (exponential backoff).")
    parser.add_argument("--output", default=None, help="Where to write scored output JSON. Defaults to <predictions>_scored.json.")
    return parser.parse_args()


def extract_option(text: str) -> Optional[str]:
    """Return a single uppercase option letter if present, else None."""
    match = re.search(r"\b([A-E])\b", text.upper())
    return match.group(1) if match else None


def call_gemini(model, question: str, raw_output: str, temperature: float, max_retries: int, retry_wait: float) -> str:
    prompt = PROMPT_TEMPLATE.format(question=question.strip(), raw_output=raw_output.strip())
    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config={"temperature": temperature},
            )
            candidate = response.text.strip() if response and response.text else ""
            option = extract_option(candidate)
            if option:
                return option
            option = extract_option(raw_output)
            if option:
                return option
            raise ValueError("Gemini returned no valid option.")
        except Exception as exc:
            if attempt == max_retries:
                raise
            time.sleep(retry_wait * attempt)
    raise RuntimeError("Unreachable")


def main():
    args = parse_args()
    if not args.api_key:
        raise ValueError("No Gemini API key provided. Pass --api-key or set GEMINI_API_KEY.")

    genai.configure(api_key=args.api_key)
    model = genai.GenerativeModel(args.model)

    pred_path = Path(args.predictions)
    data = json.loads(pred_path.read_text())
    output_path = Path(args.output) if args.output else pred_path.with_name(pred_path.stem + "_scored.json")

    scored = []
    correct = 0
    total = len(data)

    for idx, item in enumerate(data):
        question = item["qs"]
        raw_prediction = item["prediction"]
        gold = item["answer"].strip().upper()

        option = call_gemini(
            model=model,
            question=question,
            raw_output=raw_prediction,
            temperature=args.temperature,
            max_retries=args.max_retries,
            retry_wait=args.retry_wait,
        )
        is_correct = option == gold
        correct += int(is_correct)

        scored.append(
            {
                **item,
                "llm_choice": option,
                "is_correct": is_correct,
            }
        )

        if (idx + 1) % 20 == 0 or idx == total - 1:
            print(f"Processed {idx + 1}/{total}")

    accuracy = correct / total if total else 0.0
    summary = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": scored,
    }
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    print(f"Wrote detailed results to {output_path}")


if __name__ == "__main__":
    main()
