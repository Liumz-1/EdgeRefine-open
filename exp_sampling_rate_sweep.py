from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from experiment_utils import (
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    ensure_output_dir,
    get_device,
    load_graph_data,
    set_seed,
    train_and_test_node_classifier,
)
from structure import (
    graph_sampling_with_ratio,
    jaccard_probability,
    symmetric_randomized_response_perturbation,
)

DEFAULT_SAMPLE_RATIOS = [0.005, 0.01, 0.05, 0.1, 0.3, 0.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Table 7 sampling-rate sweep.")
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=["dblp", "cora"])
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--epsilon", type=float, default=2.0)
    parser.add_argument("--sample-ratios", nargs="+", type=float, default=DEFAULT_SAMPLE_RATIOS)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    output_dir = ensure_output_dir("exp_sampling_rate_sweep")
    rows = []

    for dataset_name in args.datasets:
        _, _, original_adj = load_graph_data(dataset_name, show_details=False)
        original_edges = int(np.sum(original_adj != 0) / 2)

        # k only affects the final target edge count.  Reuse the same seeded
        # randomized-response graph and probability matrix for every k.
        set_seed(args.seed)
        noisy_adj = symmetric_randomized_response_perturbation(original_adj.copy(), args.epsilon)
        prob_matrix = jaccard_probability(noisy_adj.copy(), method="histogram")

        for sample_ratio in args.sample_ratios:
            processed_adj = graph_sampling_with_ratio(
                noisy_adj,
                prob_matrix,
                args.epsilon,
                int(original_edges * sample_ratio),
            )
            for model_name in args.models:
                accuracy = train_and_test_node_classifier(
                    dataset_name=dataset_name,
                    processed_adj=processed_adj,
                    model_type=model_name,
                    seed=args.seed,
                    use_weights=False,
                    device=device,
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "epsilon": args.epsilon,
                        "sample_ratio": sample_ratio,
                        "model": model_name,
                        "seed": args.seed,
                        "test_accuracy": accuracy,
                    }
                )
                print(
                    f"dataset={dataset_name}, epsilon={args.epsilon}, k={sample_ratio}, "
                    f"model={model_name}, acc={accuracy:.4f}"
                )

    result_path = output_dir / "sampling_rate_results.csv"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
