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
    preprocess_other_baseline,
    set_seed,
    train_and_test_node_classifier,
)

BASELINES = ["dprr", "lapgraph", "ldpgen"]
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DPRR, LAPGRAPH, and LDPGen comparisons.")
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=DEFAULT_DATASETS)
    parser.add_argument("--epsilons", nargs="+", type=float, default=DEFAULT_EPSILONS)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--baselines", nargs="+", choices=BASELINES, default=BASELINES)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    output_dir = ensure_output_dir("exp_other_baselines_comparison")
    results = []
    total = len(args.baselines) * len(args.datasets) * len(args.epsilons) * len(args.models)
    current = 0

    for baseline_name in args.baselines:
        for dataset_name in args.datasets:
            _, _, original_adj = load_graph_data(dataset_name, show_details=False)
            for epsilon in args.epsilons:
                set_seed(args.seed)
                processed_adj = preprocess_other_baseline(original_adj, epsilon, baseline_name)
                for model_type in args.models:
                    current += 1
                    print(
                        f"[{current}/{total}] baseline={baseline_name}, dataset={dataset_name}, epsilon={epsilon}, model={model_type}"
                    )
                    set_seed(args.seed)
                    accuracy = train_and_test_node_classifier(
                        dataset_name,
                        processed_adj,
                        model_type,
                        seed=args.seed,
                        use_weights=False,
                        device=device,
                    )
                    results.append(
                        {
                            "baseline": baseline_name,
                            "dataset": dataset_name,
                            "epsilon": epsilon,
                            "model": model_type,
                            "seed": args.seed,
                            "test_accuracy": accuracy,
                        }
                    )

    result_path = output_dir / "other_baselines_results.csv"
    pd.DataFrame(results).to_csv(result_path, index=False)
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
