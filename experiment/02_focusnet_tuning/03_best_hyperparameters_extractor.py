import argparse
import os
import json
import numpy as np

from core.common.extract_tensorboard_metrics import extract_metric


def extract_best_parameters(log_path, metric_analysis):
    folds_paths = os.listdir(log_path)
    results_x_parameter = {}
    for fold_path in folds_paths:
        if "fold" in fold_path:
            fold_path = os.path.join(log_path, fold_path)
            seed_paths = os.listdir(fold_path)
            for seed_path in seed_paths:
                seed_path = os.path.join(fold_path, seed_path)
                parameters_paths = os.listdir(seed_path)
                for parameters in parameters_paths:
                    parameters_path = os.path.join(seed_path, parameters)
                    training_results = extract_metric(parameters_path, metric_analysis)
                    if parameters not in results_x_parameter:
                        results_x_parameter[parameters] = []
                    results_x_parameter[parameters].append(
                        [el.value for el in training_results]
                    )

    aggregated_results = {}
    for parameters, results in results_x_parameter.items():
        aggregation = np.array(results).mean(axis=0)
        best_epoch = np.argmax(aggregation)
        best_metric = aggregation[best_epoch]
        aggregated_results[parameters] = {
            "best_epoch": best_epoch,
            "best_metric": best_metric,
        }

    # Get the key with highest best_MRR using max()
    best_parameters = max(
        aggregated_results, key=lambda k: float(aggregated_results[k]["best_metric"])
    )
    # Get the corresponding best epoch
    best_epoch = int(aggregated_results[best_parameters]["best_epoch"])
    best_parameters = best_parameters.split("_")
    return {
        "epoch": best_epoch,
        "ns": int(best_parameters[1]),
        "lr": float(best_parameters[3]),
        "h_out": int(best_parameters[5]),
        "scheduler": best_parameters[7],
        "temperature": float(best_parameters[-1]),
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parser to extract the parameters to get the best hyperparameters after training."
    )
    ################################ Experiments Metadata
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="Path where to look for the results of a previous training.",
    )
    parser.add_argument(
        "--metric_analysis",
        type=str,
        required=True,
        help="The metric we'll need to use to select the best set of hyperparameters.",
    )
    args = parser.parse_args()
    best_hyperparameters = extract_best_parameters(**vars(args))
    print(json.dumps(best_hyperparameters))
