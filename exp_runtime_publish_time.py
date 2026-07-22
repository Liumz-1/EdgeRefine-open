from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from experiment_utils import (
    DEFAULT_EPSILONS,
    DEFAULT_MODELS,
    ensure_output_dir,
    get_device,
    load_graph_data,
    set_seed,
    train_and_test_node_classifier,
)
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
EPSILON_VALUES = DEFAULT_EPSILONS
METHODS = ["jaccard_sampling", "blink", "lapgraph", "ldpgen", "dprr"]
SEED = 42


def preprocess_method(
    adj: np.ndarray,
    method: str,
    epsilon: float,
    edge_num_origin: int,
    degree: np.ndarray | None,
) -> np.ndarray:
    if method == "jaccard_sampling":
        noisy_adj = symmetric_randomized_response_perturbation(adj.copy(), epsilon)
        prob_matrix = jaccard_probability(noisy_adj.copy(), method="histogram")
        return graph_sampling_with_ratio(noisy_adj, prob_matrix, epsilon, int(edge_num_origin * 0.01))
    if method == "blink":
        if degree is None:
            raise ValueError("degree is required for Blink")
        noisy_adj = symmetric_randomized_response_perturbation(adj.copy(), epsilon)
        return blink_hard_processing(noisy_adj, degree, epsilon, 0.1)
    if method == "lapgraph":
        return LAPGRAPH(adj.copy(), epsilon)
    if method == "ldpgen":
        return LDPGen(adj.copy(), epsilon, k=2)
    if method == "dprr":
        return DPRR(adj.copy(), epsilon)
    raise ValueError(f"unknown method: {method}")


def normalized_density(adj: np.ndarray) -> float:
    n = adj.shape[0]
    possible_edges = n * (n - 1) / 2
    if possible_edges == 0:
        return 0.0
    edge_count = np.count_nonzero(np.triu(adj, k=1))
    return float(edge_count / possible_edges)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure preprocessing time, graph density, and optional GNN training time for Tables 5-6."
    )
    parser.add_argument("--dataset", choices=["dblp", "acm", "cora", "amap"], default=DATASET_NAME)
    parser.add_argument("--epsilons", nargs="+", type=float, default=EPSILON_VALUES)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--model", choices=DEFAULT_MODELS, default="gat")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-epochs", type=int, default=700)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--skip-training", action="store_true", help="Only measure preprocessing and density.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir("exp_runtime_publish_time")
    device = get_device()
    set_seed(args.seed)
    _, _, adj = load_graph_data(args.dataset, show_details=False)
    edge_num_origin = int(np.sum(adj != 0) / 2)
    degree = None
    if "blink" in args.methods:
        degree = generate_degree_sequence(matrix_to_graph(adj), 0.1)

    results = []
    for method in args.methods:
        for epsilon in args.epsilons:
            set_seed(args.seed)
            start_time = time.perf_counter()
            processed_adj = preprocess_method(adj, method, epsilon, edge_num_origin, degree)
            publish_time_seconds = time.perf_counter() - start_time

            row = {
                "method": method,
                "dataset": args.dataset,
                "epsilon": epsilon,
                "model": args.model,
                "seed": args.seed,
                "normalized_density": normalized_density(processed_adj),
                "publish_time_seconds": publish_time_seconds,
            }

            if not args.skip_training:
                details = train_and_test_node_classifier(
                    dataset_name=args.dataset,
                    processed_adj=processed_adj,
                    model_type=args.model,
                    seed=args.seed,
                    use_weights=False,
                    device=device,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    return_details=True,
                )
                row.update(details)

            results.append(row)
            message = (
                f"method={method}, epsilon={epsilon}, density={row['normalized_density']:.8f}, "
                f"publish={publish_time_seconds:.4f}s"
            )
            if not args.skip_training:
                message += f", mean_epoch={row['mean_epoch_time_ms']:.4f}ms"
            print(message)

    result_path = output_dir / f"publish_time_{args.dataset}.csv"
    pd.DataFrame(results).to_csv(result_path, index=False)
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
