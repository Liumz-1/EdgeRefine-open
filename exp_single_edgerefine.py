from __future__ import annotations

import argparse
import pandas as pd

from experiment_utils import (
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    ensure_output_dir,
    get_device,
    load_graph_data,
    preprocess_edge_refine,
    resolve_experiment_scope,
    resolve_training_limits,
    set_seed,
    train_and_test_node_classifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a scoped EdgeRefine node-classification grid.")
    parser.add_argument("--preset", choices=["single", "paper", "smoke"], default="single")
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS)
    parser.add_argument("--epsilons", nargs="+", type=float)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets, epsilons, models, seeds = resolve_experiment_scope(
        args.preset, args.datasets, args.epsilons, args.models, args.seeds
    )
    max_epochs, patience = resolve_training_limits(args.preset, args.max_epochs, args.patience)
    device = get_device()
    output_dir = ensure_output_dir("exp_single_edgerefine")
    results = []
    total = len(datasets) * len(epsilons) * len(models) * len(seeds)
    current = 0

    print("start...", flush=True)
    print(f"device={device}", flush=True)
    print(f"datasets={datasets}", flush=True)
    print(f"epsilons={epsilons}", flush=True)
    print(f"models={models}", flush=True)
    print(f"seeds={seeds}", flush=True)

    for dataset_name in datasets:
        _, _, original_adj = load_graph_data(dataset_name, show_details=False)
        for epsilon in epsilons:
            for run_id, seed in enumerate(seeds, start=1):
                set_seed(seed)
                processed_adj = preprocess_edge_refine(original_adj, epsilon)
                for model_type in models:
                    current += 1
                    print(
                        f"[{current}/{total}] preparing dataset={dataset_name}, epsilon={epsilon}, "
                        f"model={model_type}, run={run_id}, seed={seed}",
                        flush=True,
                    )
                    accuracy = train_and_test_node_classifier(
                        dataset_name=dataset_name,
                        processed_adj=processed_adj,
                        model_type=model_type,
                        seed=seed,
                        use_weights=False,
                        device=device,
                        max_epochs=max_epochs,
                        patience=patience,
                    )
                    results.append(
                        {
                            "dataset": dataset_name,
                            "epsilon": epsilon,
                            "model": model_type,
                            "run_id": run_id,
                            "seed": seed,
                            "test_accuracy": accuracy,
                        }
                    )
                    print(
                        f"  done dataset={dataset_name}, epsilon={epsilon}, model={model_type}, "
                        f"run={run_id}, seed={seed}, acc={accuracy:.4f}",
                        flush=True,
                    )

    pd.DataFrame(results).to_csv(output_dir / "edgerefine_single_results.csv", index=False)
    print(f"Saved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
