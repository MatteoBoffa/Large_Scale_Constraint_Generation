import json
import argparse
import time
from core.common.metrics import compute_metrics
from core.common.query_model import (
    decode_answer_true_false,
    extract_candidates_list,
    query_model,
)
import numpy as np
from tqdm import tqdm
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process text data with LLM model.")
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input JSON data file"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the output results"
    )
    parser.add_argument(
        "--task_template",
        type=str,
        help="Template for the query",
    )
    parser.add_argument(
        "--filter_template",
        type=str,
        help="Template for the filter",
    )
    parser.add_argument(
        "--api_endpoint",
        type=str,
        default="http://localhost:30001/v1/chat/completions",
        help="API endpoint for the model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.3-8B-Instruct",
        help="Name of the model to use",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature for the LLM.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import data
    with open(args.input_path) as f:
        data = json.load(f)

    y_true, y_pred, prediction_times = [], [], []
    for sample in tqdm(data, desc="Querying on premise model..."):
        start = time.time()
        sentence = sample["sentence"]
        concepts = "[" + ", ".join(sample["concepts"]) + "]"
        contains_concept = sample["contains_concept"]
        message_filter = (
            f"Task: {args.filter_template}\nSentence: {sentence}\nWords: {concepts}"
        )
        answer_filter = query_model(
            message_filter,
            args.api_endpoint,
            args.model_name,
            temperature=args.temperature,
            max_tokens=2024,
        )
        if answer_filter:
            list_responses = extract_candidates_list(answer_filter)
            if list_responses:
                filtered_concepts = "[" + ", ".join(list_responses) + "]"
                message_task = f"Task: {args.task_template}\nSentence: {sentence}\nWords: {filtered_concepts}"
                answer_task = query_model(
                    message_task,
                    args.api_endpoint,
                    args.model_name,
                    temperature=args.temperature,
                )
                model_choice = decode_answer_true_false(answer_task)
            else:
                model_choice = "invalid"
        else:
            model_choice = "invalid"
        y_true.append(contains_concept)
        y_pred.append(model_choice)
        prediction_times.append(time.time() - start)

    n_valid_predictions = sum(p != "invalid" for p in y_pred)
    print(f"Made {n_valid_predictions:,} valid predictions!")

    # Compute all metrics
    metrics = compute_metrics(y_true, y_pred)
    metrics["avg_prediction_time"] = round(np.mean(prediction_times), 2)
    metrics["n_valid_predictions"] = n_valid_predictions

    # Save the string to a file
    with open(args.output_path, "w+") as file:
        json.dump(metrics, fp=file)

    print(f"Results saved to {args.output_path}")
