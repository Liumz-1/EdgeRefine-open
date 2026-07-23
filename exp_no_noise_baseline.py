from __future__ import annotations

import argparse
import pandas as pd

from experiment_utils import (
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    ensure_output_dir,
    get_device,
    load_graph_data,
    resolve_experiment_scope,
    resolve_training_limits,
    train_and_test_node_classifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the noise-free Origin baseline grid.")
    parser.add_argument("--preset", choices=["single", "paper", "smoke"], default="single")
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets, _, models, seeds = resolve_experiment_scope(
        args.preset, args.datasets, [0.0], args.models, args.seeds
    )
    max_epochs, patience = resolve_training_limits(args.preset, args.max_epochs, args.patience)
    device = get_device()
    output_dir = ensure_output_dir("exp_no_noise_baseline")
    rows = []

    print("start...", flush=True)
    print(f"device={device}", flush=True)
    print(f"datasets={datasets}", flush=True)
    print(f"models={models}", flush=True)
    print(f"seeds={seeds}", flush=True)
    print("running original graph without privacy noise", flush=True)

    for dataset_name in datasets:
        _, _, original_adj = load_graph_data(dataset_name, show_details=False)
        for run_id, seed in enumerate(seeds, start=1):
            for model_name in models:
                accuracy = train_and_test_node_classifier(
                    dataset_name=dataset_name,
                    processed_adj=original_adj,
                    model_type=model_name,
                    seed=seed,
                    use_weights=False,
                    device=device,
                    max_epochs=max_epochs,
                    patience=patience,
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "run_id": run_id,
                        "seed": seed,
                        "noise": "none",
                        "test_accuracy": accuracy,
                    }
                )
                print(
                    f"dataset={dataset_name}, model={model_name}, run={run_id}, "
                    f"seed={seed}, acc={accuracy:.4f}",
                    flush=True,
                )

    result = pd.DataFrame(rows)
    result_path = output_dir / "no_noise_baseline_results.csv"
    result.to_csv(result_path, index=False)

    print(f"Saved results to {result_path}", flush=True)


if __name__ == "__main__":
    main()
