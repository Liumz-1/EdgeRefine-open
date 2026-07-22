from __future__ import annotations

import argparse
import pandas as pd

from experiment_utils import (
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    DEFAULT_SINGLE_SEED,
    ensure_output_dir,
    get_device,
    load_graph_data,
    train_and_test_node_classifier,
)

SEED = DEFAULT_SINGLE_SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the noise-free Origin baseline grid.")
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    output_dir = ensure_output_dir("exp_no_noise_baseline")
    rows = []

    print("start...", flush=True)
    print(f"device={device}", flush=True)
    print(f"datasets={args.datasets}", flush=True)
    print(f"models={args.models}", flush=True)
    print(f"seed={args.seed}", flush=True)
    print("running original graph without privacy noise", flush=True)

    for dataset_name in args.datasets:
        _, _, original_adj = load_graph_data(dataset_name, show_details=False)
        for model_name in args.models:
            accuracy = train_and_test_node_classifier(
                dataset_name=dataset_name,
                processed_adj=original_adj,
                model_type=model_name,
                seed=args.seed,
                use_weights=False,
                device=device,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": args.seed,
                    "noise": "none",
                    "test_accuracy": accuracy,
                }
            )
            print(f"dataset={dataset_name}, model={model_name}, acc={accuracy:.4f}", flush=True)

    result = pd.DataFrame(rows)
    result_path = output_dir / "no_noise_baseline_results.csv"
    result.to_csv(result_path, index=False)

    print(f"Saved results to {result_path}", flush=True)


if __name__ == "__main__":
    main()
