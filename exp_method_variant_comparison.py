from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiment_utils import (
    DEFAULT_METHOD_COMBOS,
    ensure_output_dir,
    get_device,
    load_graph_data,
    set_seed,
    train_and_test_node_classifier,
)
from structure import (
    adamic_adar_prob_matrix,
    graph_sampling_with_ratio,
    jaccard_probability,
    symmetric_randomized_response_perturbation,
)

DATASET_NAME = "dblp"
EPSILON = 1.0
MODEL_NAME = "gat"
SEED = 42
METHOD_COMBOS = DEFAULT_METHOD_COMBOS


def build_processed_adj(original_adj: np.ndarray, epsilon: float, feature_method: str, prob_method: str) -> np.ndarray:
    edge_num_origin = int(np.sum(original_adj != 0) / 2)
    noisy_adj = symmetric_randomized_response_perturbation(original_adj.copy(), epsilon)
    if feature_method == "jaccard":
        prob_matrix = jaccard_probability(noisy_adj.copy(), method=prob_method)
    elif feature_method == "adamic_adar":
        prob_matrix = adamic_adar_prob_matrix(noisy_adj.copy())
    else:
        raise ValueError(f"unknown feature method: {feature_method}")
    return graph_sampling_with_ratio(noisy_adj, prob_matrix, epsilon, int(edge_num_origin * 0.01))


def main() -> None:
    device = get_device()
    output_dir = ensure_output_dir("exp_method_variant_comparison")
    _, _, original_adj = load_graph_data(DATASET_NAME, show_details=False)
    results = []

    for feature_method, prob_method in METHOD_COMBOS:
        set_seed(SEED)
        method_name = f"{feature_method}+{prob_method}"
        print(f"Running {method_name}")
        start_time = time.perf_counter()
        processed_adj = build_processed_adj(original_adj, EPSILON, feature_method, prob_method)
        accuracy = train_and_test_node_classifier(
            dataset_name=DATASET_NAME,
            processed_adj=processed_adj,
            model_type=MODEL_NAME,
            seed=SEED,
            use_weights=False,
            device=device,
        )
        duration = time.perf_counter() - start_time
        results.append(
            {
                "dataset": DATASET_NAME,
                "epsilon": EPSILON,
                "model": MODEL_NAME,
                "feature_method": feature_method,
                "probability_method": prob_method,
                "test_accuracy": accuracy,
                "runtime_seconds": duration,
            }
        )

    result_path = output_dir / "method_variant_results.csv"
    pd.DataFrame(results).to_csv(result_path, index=False)
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
