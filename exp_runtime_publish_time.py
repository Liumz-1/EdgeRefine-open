from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiment_utils import ensure_output_dir, load_graph_data, set_seed
from structure import (
    DPRR,
    LAPGRAPH,
    LDPGen,
    blink_hard_processing,
    generate_degree_sequence,
    graph_sampling_with_ratio,
    jaccard_probability,
    matrix_to_graph,
    symmetric_randomized_response_perturbation,
)

DATASET_NAME = "dblp"
EPSILON_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
METHODS = ["jaccard_sampling", "blink", "lapgraph", "ldpgen", "dprr"]
SEED = 42


def get_publish_time(adj: np.ndarray, method: str, epsilon: float, edge_num_origin: int, degree: np.ndarray) -> float:
    start_time = time.perf_counter()
    if method == "jaccard_sampling":
        noisy_adj = symmetric_randomized_response_perturbation(adj.copy(), epsilon)
        prob_matrix = jaccard_probability(noisy_adj.copy(), method="histogram")
        graph_sampling_with_ratio(noisy_adj, prob_matrix, epsilon, int(edge_num_origin * 0.01))
    elif method == "blink":
        noisy_adj = symmetric_randomized_response_perturbation(adj.copy(), epsilon)
        blink_hard_processing(noisy_adj, degree, epsilon, 0.1)
    elif method == "lapgraph":
        LAPGRAPH(adj.copy(), epsilon)
    elif method == "ldpgen":
        LDPGen(adj.copy(), epsilon, k=2)
    elif method == "dprr":
        DPRR(adj.copy(), epsilon)
    else:
        raise ValueError(f"unknown method: {method}")
    return time.perf_counter() - start_time


def main() -> None:
    output_dir = ensure_output_dir("exp_runtime_publish_time")
    set_seed(SEED)
    _, _, adj = load_graph_data(DATASET_NAME, show_details=False)
    edge_num_origin = int(np.sum(adj != 0) / 2)
    degree = generate_degree_sequence(matrix_to_graph(adj), 0.1)
    results = []

    for method in METHODS:
        for epsilon in EPSILON_VALUES:
            set_seed(SEED)
            duration = get_publish_time(adj, method, epsilon, edge_num_origin, degree)
            print(f"method={method}, epsilon={epsilon}, time={duration:.4f}s")
            results.append({"method": method, "epsilon": epsilon, "publish_time_seconds": duration})

    result_path = output_dir / f"publish_time_{DATASET_NAME}.csv"
    pd.DataFrame(results).to_csv(result_path, index=False)
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
