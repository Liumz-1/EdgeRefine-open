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
    preprocess_blink_hard,
    preprocess_blink_hybrid,
    preprocess_other_baseline,
    set_seed,
    train_and_test_node_classifier,
)

DATASETS = DEFAULT_DATASETS
EPSILON_VALUES = DEFAULT_EPSILONS
MODELS = DEFAULT_MODELS
SEED = DEFAULT_SINGLE_SEED
BASELINES = {
    "blink_hard": (preprocess_blink_hard, False),
    "blink_hybrid": (preprocess_blink_hybrid, True),
    "dprr": (lambda adj, eps: preprocess_other_baseline(adj, eps, "dprr"), False),
    "lapgraph": (lambda adj, eps: preprocess_other_baseline(adj, eps, "lapgraph"), False),
    "ldpgen": (lambda adj, eps: preprocess_other_baseline(adj, eps, "ldpgen"), False),
}


def main() -> None:
    device = get_device()
    output_dir = ensure_output_dir("exp_single_baselines")
    results = []
    total = len(BASELINES) * len(DATASETS) * len(EPSILON_VALUES) * len(MODELS)
    current = 0

    print("start...", flush=True)
    print(f"device={device}", flush=True)
    print(f"baselines={list(BASELINES.keys())}", flush=True)
    print(f"datasets={DATASETS}", flush=True)
    print(f"epsilons={EPSILON_VALUES}", flush=True)
    print(f"models={MODELS}", flush=True)
    print(f"seed={SEED}", flush=True)

    for baseline_name, (preprocess_fn, use_weights) in BASELINES.items():
        for dataset_name in DATASETS:
            _, _, original_adj = load_graph_data(dataset_name, show_details=False)
            for epsilon in EPSILON_VALUES:
                set_seed(SEED)
                processed_adj = preprocess_fn(original_adj, epsilon)
                for model_type in MODELS:
                    current += 1
                    print(
                        f"[{current}/{total}] preparing baseline={baseline_name}, dataset={dataset_name}, epsilon={epsilon}, model={model_type}",
                        flush=True,
                    )
                    print(f"  training seed={SEED}", flush=True)
                    accuracy = train_and_test_node_classifier(
                        dataset_name=dataset_name,
                        processed_adj=processed_adj,
                        model_type=model_type,
                        seed=SEED,
                        use_weights=use_weights,
                        device=device,
                    )
                    results.append(
                        {
                            "baseline": baseline_name,
                            "dataset": dataset_name,
                            "epsilon": epsilon,
                            "model": model_type,
                            "seed": SEED,
                            "use_weights": use_weights,
                            "test_accuracy": accuracy,
                        }
                    )
                    print(
                        f"  done baseline={baseline_name}, dataset={dataset_name}, epsilon={epsilon}, model={model_type}, seed={SEED}, acc={accuracy:.4f}",
                        flush=True,
                    )

    pd.DataFrame(results).to_csv(output_dir / "baseline_single_results.csv", index=False)
    print(f"Saved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
