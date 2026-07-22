from __future__ import annotations

import argparse

import pandas as pd

from experiment_utils import (
    DEFAULT_DATASETS,
    DEFAULT_EPSILONS,
    DEFAULT_MODELS,
    ensure_output_dir,
    get_device,
    load_graph_data,
    set_seed,
    train_and_test_node_classifier,
)
from structure import symmetric_randomized_response_perturbation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Figure 5 randomized-response-only ablation.")
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=["cora", "dblp"])
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--epsilons", nargs="+", type=float, default=DEFAULT_EPSILONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    output_dir = ensure_output_dir("exp_rr_only_ablation")
    rows = []

    for dataset_name in args.datasets:
        _, _, original_adj = load_graph_data(dataset_name, show_details=False)
        for epsilon in args.epsilons:
            for seed in args.seeds:
                set_seed(seed)
                rr_only_adj = symmetric_randomized_response_perturbation(original_adj.copy(), epsilon)
                for model_name in args.models:
                    accuracy = train_and_test_node_classifier(
                        dataset_name=dataset_name,
                        processed_adj=rr_only_adj,
                        model_type=model_name,
                        seed=seed,
                        use_weights=False,
                        device=device,
                    )
                    rows.append(
                        {
                            "variant": "rr_only",
                            "dataset": dataset_name,
                            "epsilon": epsilon,
                            "model": model_name,
                            "seed": seed,
                            "test_accuracy": accuracy,
                        }
                    )
                    print(
                        f"variant=rr_only, dataset={dataset_name}, epsilon={epsilon}, "
                        f"model={model_name}, seed={seed}, acc={accuracy:.4f}"
                    )

    result_path = output_dir / "rr_only_ablation_results.csv"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
