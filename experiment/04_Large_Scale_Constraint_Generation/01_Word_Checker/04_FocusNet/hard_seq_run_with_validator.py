import json
import argparse
import os
import time
from core.common.metrics import compute_metrics
from core.common.query_model import decode_answer_true_false, query_model
from core.rule_checker.knn_checker import KNNChecker
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path

from core.contrastive_encoder.classes.rule_encoder import ContrastiveConstrainerModel
from core.contrastive_encoder.functions.prepare_data import tokenize_data
from core.rule_checker.random_forest import RandomForestChecker


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
        "--llm_model",
        type=str,
        default="meta-llama/Meta-Llama-3.3-8B-Instruct",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--n_pools",
        type=int,
        default=5,
        help="Number of chunks we have to divide the pool of answers in. Default to 5.",
    )

    parser.add_argument(
        "--h_out",
        type=int,
        default=128,
        help="Output hidden dimension size",
    )
    parser.add_argument(
        "--h_in",
        type=int,
        default=768,
        help="Input hidden dimension size",
    )
    parser.add_argument(
        "--temperature_focusnet",
        type=float,
        default=0.05,
        help="Temperature parameter for FocusNet inference",
    )
    parser.add_argument(
        "--normalize_embeddings",
        type=str,
        default="True",
        help="Whether to normalize embeddings",
    )
    parser.add_argument(
        "--embeddings_aggr",
        type=str,
        default="mean_pooling",
        help="Method for aggregating embeddings",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Pretrained sentence encoder model to use",
    )
    parser.add_argument("--classifier", type=str, choices=["rf", "knn"], default="rf")
    parser.add_argument(
        "--path_prompt_template",
        type=str,
        default="prompts/prompt_templates.json",
        help="Path to the prompt templates JSON file",
    )
    parser.add_argument(
        "--huggingface_cache",
        type=str,
        default=None,
        help="Path to Hugging Face cache directory",
    )
    parser.add_argument(
        "--best_model",
        type=str,
        required=True,
        help="Path to the best model checkpoint",
    )
    parser.add_argument(
        "--temperature_llm",
        type=float,
        default=0.2,
        help="Temperature parameter for LLM",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=5,
        help="Id of the GPUs we want to use. Default to 5.",
    )
    opts = parser.parse_args()
    opts.normalize_embeddings = opts.normalize_embeddings.lower() == "true"
    return vars(opts)


def validator_call(
    sentence, rules, path_prompt, rule_checker, confidence_threshold, tokenizer, device
):
    # Extract number of rules
    n_rules = len(rules)
    # Tokenize the sentence > will have shape 1x2xMAX_LEN
    tokenized_sentences = tokenize_data(
        tokenizer,
        path_prompt,
        [sentence],  # Notice: have to turn sentence into a list
        input_type="sentences",
    ).to(device)
    # Expand the same sentence > each copy will be compared against a rule
    tokenized_sentences_expanded = tokenized_sentences.expand(n_rules, -1, -1)
    # Now, tokenize the rules > will get a tensor of shape N_rulesx2xMAX_LEN
    tokenized_rules = tokenize_data(
        tokenizer,
        path_prompt,
        rules,
        input_type="rules",
    ).to(device)
    # Pretend N_rules is the batch size > transform multiple rules into single rules N_rulesx1x2xMAX_LEN
    tokenized_rules = tokenized_rules.unsqueeze(1)
    # check compliance wants lists as input > turning tensors into lists
    #   Sentences: list of N_rules elements (all are the same)
    list_of_sentences = list(tokenized_sentences_expanded.unbind(dim=0))
    #   Rules: list of N_rules lists (each of size 1) of tensors >> [[2, MAX_LEN]]
    list_of_rules = list(tokenized_rules.unbind(dim=0))
    list_of_lists_rules = [list(el.unbind(dim=0)) for el in list_of_rules]
    # finally, call compliance function
    compliance_list, _ = rule_checker.check_compliance(
        sentences=list_of_sentences,
        rules=list_of_lists_rules,
        device=device,
        confidence_score=confidence_threshold,
    )
    return compliance_list


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Create output directory if it doesn't exist
    output_dir = Path(args["output_path"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    # Import data
    with open(args["input_path"]) as f:
        data = json.load(f)
    # Load Validator
    device = torch.device(f"cuda:{args['gpu_id']}")
    rule_constrainer = ContrastiveConstrainerModel(
        h_out=args["h_out"],
        h_in=args["h_in"],
        require_encoder=True,  # suppose we always get raw data
        pretrained_model=args["pretrained_model"],
        huggingface_cache=args["huggingface_cache"],
        aggregation=args["embeddings_aggr"],
        normalize=args["normalize_embeddings"],
        temperature=args["temperature_focusnet"],
    )
    rule_constrainer.load_best(path_best_model=args["best_model"])
    rule_constrainer = rule_constrainer.to(device)
    rule_constrainer.eval()
    # Load the rule checker
    if args["classifier"] == "rf":
        rule_checker = RandomForestChecker.load(
            path=args["best_model"], rule_embedder=rule_constrainer
        )
    else:
        rule_checker = KNNChecker.load(
            path=args["best_model"], rule_embedder=rule_constrainer
        )
    best_threshold = rule_checker.get_confidence_threshold()
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args["pretrained_model"],
        cache_dir=args["huggingface_cache"],
    )
    y_true, y_pred, prediction_times, len_alerts = [], [], [], []
    analysis_validator = {
        "correct": {"llm_correct": 0, "llm_incorrect": 0},
        "incorrect": {"llm_correct": 0, "llm_incorrect": 0},
    }
    # Begin verification loop
    for sample in tqdm(data, desc="Querying on premise model..."):
        start = time.time()
        sentence = sample["sentence"]
        list_concepts = sample["concepts"]
        n_rules = len(list_concepts)
        contains_concept = sample["contains_concept"]
        # First use FocusNet
        predictions = validator_call(
            sentence,
            list_concepts,
            args["path_prompt_template"],
            rule_checker,
            best_threshold,
            tokenizer,
            device,
        )
        alert_concepts = [
            word for (word, prediction) in zip(list_concepts, predictions) if prediction
        ]
        intersection_contained_predicted = set(
            sample["concepts_contained"]
        ).intersection(set(alert_concepts))
        len_alerts.append(len(alert_concepts))
        concepts = (
            "[" + ", ".join(alert_concepts) + "]"
            if len(alert_concepts) != 0
            else "[None]"
        )
        # Then query the model
        message = f"Task: {args['template']}\nSentence: {sentence}\nWords: {concepts}"
        answer = query_model(
            message,
            args["api_endpoint"],
            args["llm_model"],
            temperature=args["temperature_llm"],
            timeout_connection=30,
            timeout_read=180,
        )
        model_choice = decode_answer_true_false(answer)

        # Stats about the rule_checker given the choice of the model
        if model_choice == contains_concept:
            if len(intersection_contained_predicted) > 0 or (
                len(sample["concepts_contained"]) == 0 and len(alert_concepts) == 0
            ):
                analysis_validator["correct"]["llm_correct"] += 1
            else:
                analysis_validator["incorrect"]["llm_correct"] += 1
        elif model_choice != "invalid":
            if len(intersection_contained_predicted) > 0 or (
                len(sample["concepts_contained"]) == 0 and len(alert_concepts) == 0
            ):
                analysis_validator["correct"]["llm_incorrect"] += 1
            else:
                analysis_validator["incorrect"]["llm_incorrect"] += 1

        y_true.append(contains_concept)
        y_pred.append(model_choice)
        prediction_times.append(time.time() - start)

    n_valid_predictions = sum(p != "invalid" for p in y_pred)
    print(f"Made {n_valid_predictions:,} valid predictions!")

    # Compute all metrics
    metrics = compute_metrics(y_true, y_pred)
    metrics["avg_prediction_time"] = round(np.mean(prediction_times), 2)
    metrics["n_valid_predictions"] = n_valid_predictions
    metrics["avg_len_predictions"] = round(np.mean(len_alerts), 2)

    # Save the string to a file
    with open(args["output_path"], "w+") as file:
        json.dump(metrics, fp=file, indent=4)

    path = os.path.dirname(args["output_path"])
    with open(os.path.join(path, f"validator_performance_{n_rules}.json"), "w+") as f:
        json.dump(analysis_validator, f, indent=4)

    print(f"Results saved to {args['output_path']}")
