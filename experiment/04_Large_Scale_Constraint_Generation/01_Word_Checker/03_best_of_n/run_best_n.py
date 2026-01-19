import json
import argparse
import time
from core.common.metrics import compute_metrics
from core.common.query_model import decode_answer_true_false, query_model
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process text data with LLM model.")
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input JSON data file"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the output results"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="Check if the following sentence contains one of the following set of words. Just answer True or False.\nSentence:",
        help="Template for the query",
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
        "--n_rounds",
        type=int,
        default=3,
        help="How many times shall we ask the model the same question, before drawing a conclusion. Default to 3.",
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
    n_agreements = 0
    for sample in tqdm(data, desc="Querying on premise model..."):
        start = time.time()
        sentence = sample["sentence"]
        concepts = "[" + ", ".join(sample["concepts"]) + "]"
        contains_concept = sample["contains_concept"]
        message_jury_models = (
            f"Task: {args.template}\nSentence: {sentence}\nWords: {concepts}"
        )
        answers, judge_choices = [], []
        for round_answer in range(args.n_rounds):
            answer_judge = query_model(
                message_jury_models,
                args.api_endpoint,
                args.model_name,
                temperature=args.temperature,
            )
            answers.append(answer_judge)
            judge_choice = decode_answer_true_false(answer_judge)
            judge_choices.append(judge_choice)
        if len(set(judge_choices)) == 1:
            n_agreements += 1
        message_critic = f"Give me your final opinion over the verdicts of a jury of {args.n_rounds} LLMs. \n"
        message_critic += f"When prompted the following message: {message_jury_models}\nA jury of LLMs answered:"
        for round_answer in range(args.n_rounds):
            message_critic += f"\nJudge {round_answer}: {answers[round_answer]}"
        message_critic += "\nWhat is your final verdict? Just answer True or False: Ensure to enclude your final answer into <answer></answer>. For instance, if the sentence contains one of the words, answer <answer>True</answer>; <answer>False</answer> otherwise."
        answer_critic = query_model(
            message_critic,
            args.api_endpoint,
            args.model_name,
            temperature=args.temperature,
        )
        model_choice = decode_answer_true_false(answer_critic)
        y_true.append(contains_concept)
        y_pred.append(model_choice)
        prediction_times.append(time.time() - start)

    n_valid_predictions = sum(p != "invalid" for p in y_pred)
    print(f"Made {n_valid_predictions:,} valid predictions!")

    # Compute all metrics
    metrics = compute_metrics(y_true, y_pred)
    metrics["avg_prediction_time"] = round(np.mean(prediction_times), 2)
    metrics["n_valid_predictions"] = n_valid_predictions
    metrics["n_agreements_among_judges"] = n_agreements

    # Save the string to a file
    with open(args.output_path, "w+") as file:
        json.dump(metrics, fp=file, indent=4)

    print(f"Results saved to {args.output_path}")
