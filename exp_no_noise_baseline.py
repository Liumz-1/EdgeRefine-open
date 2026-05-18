from __future__ import annotations

import pandas as pd

from experiment_utils import (
    DEFAULT_SINGLE_SEED,
    ensure_output_dir,
    get_device,
    load_graph_data,
    train_and_test_node_classifier,
)

DATASET_NAME = "amap"
MODEL_NAME = "gin"
SEED = DEFAULT_SINGLE_SEED


def main() -> None:
    device = get_device()
    output_dir = ensure_output_dir("exp_no_noise_baseline")
    _, _, original_adj = load_graph_data(DATASET_NAME, show_details=False)

    print("start...", flush=True)
    print(f"device={device}", flush=True)
    print(f"dataset={DATASET_NAME}", flush=True)
    print(f"model={MODEL_NAME}", flush=True)
    print(f"seed={SEED}", flush=True)
    print("running original graph without privacy noise", flush=True)

    accuracy = train_and_test_node_classifier(
        dataset_name=DATASET_NAME,
        processed_adj=original_adj,
        model_type=MODEL_NAME,
        seed=SEED,
        use_weights=False,
        device=device,
    )

    result = pd.DataFrame(
        [
            {
                "dataset": DATASET_NAME,
                "model": MODEL_NAME,
                "seed": SEED,
                "noise": "none",
                "test_accuracy": accuracy,
            }
        ]
    )
    result_path = output_dir / "no_noise_baseline_results.csv"
    result.to_csv(result_path, index=False)

    print(f"acc={accuracy:.4f}", flush=True)
    print(f"Saved results to {result_path}", flush=True)


if __name__ == "__main__":
    main()
