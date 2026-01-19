import json
import argparse
import time
from core.common.evaluator import binary_evaluation, evaluate_infraction
from core.common.query_model import decode_answer, query_model
import numpy as np
from tqdm import tqdm

from pathlib import Path

from core.llm_as_a_judge.judge import eval_semantic_equivalence_geval
from core.rule_checker.baseline_NLP_checker import BaselineNLPChecker


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
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature for the LLM.",
    )

    parser.add_argument(
        "--nltk_data_dir",
        type=str,
        default=None,
        help="Default path to nltk libraries. If None, using default.",
    )

    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4.1",
        help="Model for LLM as a judge for semantic evaluation.",
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

    baseline_rule_checker = BaselineNLPChecker(nltk_data_dir=args.nltk_data_dir)
    # Real sentence, new generation, label
    real_sentences, generated_sentences, y_labels = [], [], []
    # Traditional scores
    infraction_detection, infractions, is_new_invalid = [], [], []
    # LLM as a judge
    passed_semantic_test, score_semantic_test, reason_semantic_test = [], [], []
    for sample in tqdm(data, desc="Querying on premise model..."):
        start = time.time()
        sentence = sample["sentence"]
        real_sentences.append(sentence)
        concepts = "[" + ", ".join(sample["concepts"]) + "]"
        contains_concept = sample["contains_concept"]
        y_labels.append(contains_concept)
        message = f"{args.template}\nThis is the input sentence: {sentence}\nThis is the list of forbidden words: {concepts}"
        answer = query_model(
            message, args.api_endpoint, args.model_name, temperature=args.temperature
        )
        model_choice = decode_answer(answer)
        generated_sentences.append(model_choice)
        # First, binary check: did the model realise the original sentence was fine?
        recognize_infaction = binary_evaluation(
            original=sentence,
            prediction=model_choice,
        )
        infraction_detection.append(recognize_infaction)
        # Then, check if the new sentence contains other forbidden words
        new_infractions, generate_new_infraction = evaluate_infraction(
            baseline_rule_checker, model_choice, sample["concepts"]
        )
        infractions.append(new_infractions)
        is_new_invalid.append(generate_new_infraction)
        # Eventually, use an LLM as a judge to check that the semantic is the same
        res = eval_semantic_equivalence_geval(
            gt=sentence,
            generated=model_choice,
            judge_model=args.judge_model,
            threshold=0.8,
        )
        passed, score, reason = res.passed, res.score, res.reason
        passed_semantic_test.append(passed)
        score_semantic_test.append(score)
        reason_semantic_test.append(reason)

    dict_results = {
        "real_sentence": real_sentences,
        "generated_sentence": generated_sentences,
        "real_was_violating": y_labels,
        "infraction_detection": infraction_detection,
        "generate_new_infraction": infractions,
        "is_new_invalid": is_new_invalid,
        "semantically_valid_generation": passed_semantic_test,
        "semantic_scores": score_semantic_test,
        "reason": reason_semantic_test,
    }

    # Save the string to a file
    with open(args.output_path + "/" + "results_x_row.json", "w+") as file:
        json.dump(dict_results, fp=file, indent=4)

    metrics = {
        "n_infraction_detected": sum(infraction_detection),
        "n_faulty_generation": sum(infractions),
        "n_valid_semantically": sum(passed_semantic_test),
        "avg_semantic_score": np.mean(score_semantic_test),
    }

    # Save the string to a file
    with open(args.output_path + "/" + "metrics.json", "w+") as file:
        json.dump(metrics, fp=file, indent=4)

    print(f"Results saved to {args.output_path}")
