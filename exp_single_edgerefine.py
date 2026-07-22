from __future__ import annotations

import pandas as pd

from experiment_utils import (
    DEFAULT_DATASETS,
    DEFAULT_EPSILONS,
    DEFAULT_MODELS,
    DEFAULT_SINGLE_SEED,
    ensure_output_dir,
    get_device,
    load_graph_data,
    preprocess_edge_refine,
    set_seed,
    train_and_test_node_classifier,
)

DATASETS = DEFAULT_DATASETS
EPSILON_VALUES = DEFAULT_EPSILONS
MODELS = DEFAULT_MODELS
SEED = DEFAULT_SINGLE_SEED


def main() -> None:
    device = get_device()
    output_dir = ensure_output_dir("exp_single_edgerefine")
    results = []
    total = len(DATASETS) * len(EPSILON_VALUES) * len(MODELS)
    current = 0

    print("start...", flush=True)
    print(f"device={device}", flush=True)
    print(f"datasets={DATASETS}", flush=True)
    print(f"epsilons={EPSILON_VALUES}", flush=True)
    print(f"models={MODELS}", flush=True)
    print(f"seed={SEED}", flush=True)

    for dataset_name in DATASETS:
        _, _, original_adj = load_graph_data(dataset_name, show_details=False)
        for epsilon in EPSILON_VALUES:
            set_seed(SEED)
            processed_adj = preprocess_edge_refine(original_adj, epsilon)
            for model_type in MODELS:
                current += 1
                print(
                    f"[{current}/{total}] preparing dataset={dataset_name}, epsilon={epsilon}, model={model_type}",
                    flush=True,
                )
                print(f"  training seed={SEED}", flush=True)
                accuracy = train_and_test_node_classifier(
                    dataset_name=dataset_name,
                    processed_adj=processed_adj,
                    model_type=model_type,
                    seed=SEED,
                    use_weights=False,
                    device=device,
                )
                results.append(
                    {
                        "dataset": dataset_name,
                        "epsilon": epsilon,
                        "model": model_type,
                        "seed": SEED,
                        "test_accuracy": accuracy,
                    }
                )
                print(
                    f"  done dataset={dataset_name}, epsilon={epsilon}, model={model_type}, seed={SEED}, acc={accuracy:.4f}",
                    flush=True,
                )

    pd.DataFrame(results).to_csv(output_dir / "edgerefine_single_results.csv", index=False)
    print(f"Saved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
