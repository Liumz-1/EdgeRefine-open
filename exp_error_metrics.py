from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from experiment_utils import DEFAULT_DATASETS, DEFAULT_EPSILONS, ensure_output_dir, load_graph_data, set_seed
from structure import jaccard_probability, symmetric_randomized_response_perturbation

DATASETS = DEFAULT_DATASETS
EPSILON_VALUES = DEFAULT_EPSILONS
SEED = 42


class SuppressAllOutput:
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.null_fd = open(os.devnull, "w")
        sys.stdout = self.null_fd
        sys.stderr = self.null_fd
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.null_fd.close()


def calculate_brier_score(y_true, y_prob):
    return np.mean((y_true - y_prob) ** 2)


def calculate_mae(y_true, y_prob):
    return np.mean(np.abs(y_true - y_prob))


def calculate_ece(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    ece = 0.0
    total_samples = len(y_true)
    for idx in range(n_bins):
        mask = bin_indices == idx
        if np.sum(mask) > 0:
            avg_prob = np.mean(y_prob[mask])
            avg_true = np.mean(y_true[mask])
            ece += (np.sum(mask) / total_samples) * np.abs(avg_prob - avg_true)
    return ece


def calculate_pmad(y_prob):
    return np.mean(np.abs(y_prob - np.mean(y_prob)))


def calculate_metrics(original_adj: np.ndarray, epsilon: float):
    with SuppressAllOutput():
        noisy_adj = symmetric_randomized_response_perturbation(original_adj, epsilon)
        jaccard_prob_matrix = jaccard_probability(noisy_adj, method="histogram")
        y_true = original_adj.flatten()
        y_prob = jaccard_prob_matrix.flatten()
        ece = calculate_ece(y_true, y_prob)
        pmad = calculate_pmad(y_prob)
        return {
            "mae": calculate_mae(y_true, y_prob),
            "brier": calculate_brier_score(y_true, y_prob),
            "ece": ece,
            "pmad": pmad,
            "ece_pmad_ratio": ece / pmad if pmad > 0 else np.nan,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the probability-error metrics used in Figure 4.")
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=DEFAULT_DATASETS)
    parser.add_argument("--epsilons", nargs="+", type=float, default=DEFAULT_EPSILONS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir("exp_error_metrics")
    rows = []

    for dataset_name in args.datasets:
        _, _, original_adj = load_graph_data(dataset_name, show_details=False)
        for epsilon in args.epsilons:
            set_seed(args.seed)
            metrics = calculate_metrics(original_adj, epsilon)
            rows.append({"dataset": dataset_name, "epsilon": epsilon, "seed": args.seed, **metrics})
            print(
                f"dataset={dataset_name}, epsilon={epsilon}, mae={metrics['mae']:.6f}, "
                f"brier={metrics['brier']:.6f}, ece={metrics['ece']:.6f}, "
                f"pmad={metrics['pmad']:.6f}, ece/pmad={metrics['ece_pmad_ratio']:.6f}"
            )

    pd.DataFrame(rows).to_csv(output_dir / "error_metrics_results.csv", index=False)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
