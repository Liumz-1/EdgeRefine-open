from __future__ import annotations

import pandas as pd

from experiment_utils import (
    DEFAULT_DATASETS,
    DEFAULT_EPSILONS,
    DEFAULT_MODELS,
    ensure_output_dir,
    get_device,
    load_graph_data,
    preprocess_blink_hybrid,
    set_seed,
    train_and_test_node_classifier,
)

DATASETS = DEFAULT_DATASETS
EPSILON_VALUES = DEFAULT_EPSILONS
MODELS = DEFAULT_MODELS
SEED = 42


def main() -> None:
    device = get_device()
    output_dir = ensure_output_dir("exp_blink_hybrid_comparison")
    results = []
    total = len(DATASETS) * len(EPSILON_VALUES) * len(MODELS)
    current = 0

    for dataset_name in DATASETS:
        _, _, original_adj = load_graph_data(dataset_name, show_details=False)
        for epsilon in EPSILON_VALUES:
            set_seed(SEED)
            processed_adj = preprocess_blink_hybrid(original_adj, epsilon)
            for model_type in MODELS:
                current += 1
                print(f"[{current}/{total}] dataset={dataset_name}, epsilon={epsilon}, model={model_type}")
                set_seed(SEED)
                accuracy = train_and_test_node_classifier(
                    dataset_name,
                    processed_adj,
                    model_type,
                    seed=SEED,
                    use_weights=True,
                    device=device,
                )
                results.append(
                    {
                        "baseline": "blink_hybrid",
                        "dataset": dataset_name,
                        "epsilon": epsilon,
                        "model": model_type,
                        "test_accuracy": accuracy,
                    }
                )

    result_path = output_dir / "blink_hybrid_results.csv"
    pd.DataFrame(results).to_csv(result_path, index=False)
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
