from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_utils import ensure_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Table 3/4 metrics from generated accuracy CSV files.")
    parser.add_argument("--edgerefine-csv", required=True)
    parser.add_argument("--baseline-csvs", nargs="+", required=True)
    parser.add_argument("--origin-csv", required=True)
    return parser.parse_args()


def standardize_private_results(path: str, default_method: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"dataset", "model", "epsilon", "test_accuracy"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if "method" not in frame.columns:
        if "baseline" in frame.columns:
            frame["method"] = frame["baseline"]
        else:
            frame["method"] = default_method
    frame["model"] = frame["model"].astype(str).str.lower()
    return frame[["method", "dataset", "model", "epsilon", "test_accuracy"]]


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir("exp_summary_metrics")
    private_frames = [standardize_private_results(args.edgerefine_csv, "edgerefine")]
    private_frames.extend(standardize_private_results(path, Path(path).stem) for path in args.baseline_csvs)
    private = pd.concat(private_frames, ignore_index=True)

    origin = pd.read_csv(args.origin_csv)
    required_origin = {"dataset", "model", "test_accuracy"}
    missing_origin = required_origin.difference(origin.columns)
    if missing_origin:
        raise ValueError(f"{args.origin_csv} is missing columns: {sorted(missing_origin)}")
    origin["model"] = origin["model"].astype(str).str.lower()
    origin_lookup = origin.groupby(["dataset", "model"])["test_accuracy"].mean()

    rows = []
    grouped = private.groupby(["method", "dataset", "model"], sort=True)
    for (method, dataset_name, model_name), group in grouped:
        key = (dataset_name, model_name)
        if key not in origin_lookup.index:
            raise ValueError(f"No Origin result for dataset={dataset_name}, model={model_name}")
        origin_accuracy = float(origin_lookup.loc[key])
        if origin_accuracy <= 0:
            raise ValueError(f"Origin accuracy must be positive for dataset={dataset_name}, model={model_name}")

        by_epsilon = group.groupby("epsilon", as_index=False)["test_accuracy"].mean().sort_values("epsilon")
        accuracies = by_epsilon["test_accuracy"].to_numpy(dtype=float)
        mean_accuracy = float(np.mean(accuracies))
        variance = float(np.var(accuracies))
        cv = float(np.std(accuracies) / mean_accuracy) if mean_accuracy != 0 else np.nan
        aur = float(np.mean(accuracies / origin_accuracy))

        if len(accuracies) > 1:
            downward_changes = np.maximum(0.0, accuracies[:-1] - accuracies[1:]) / origin_accuracy
            mcf = float(1.0 - np.mean(downward_changes))
        else:
            mcf = 1.0

        rows.append(
            {
                "method": method,
                "dataset": dataset_name,
                "model": model_name,
                "num_privacy_budgets": len(accuracies),
                "origin_accuracy": origin_accuracy,
                "mean_accuracy": mean_accuracy,
                "variance": variance,
                "cv": cv,
                "aur": aur,
                "mcf": mcf,
                "pubi": aur * mcf,
            }
        )

    result_path = output_dir / "paper_summary_metrics.csv"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
