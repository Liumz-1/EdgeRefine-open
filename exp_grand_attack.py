from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_utils import ensure_output_dir, load_graph_data, set_seed
from grand_attack_core import (
    DeterministicAttack,
    Graph,
    compute_reconstruction_metrics,
    privacy_gain_over_random,
)
from structure import graph_sampling_with_ratio, jaccard_probability, symmetric_randomized_response_perturbation

DATASETS = ["cora", "amap"]
EPSILON_VALUES = [0.5]
SEED = 42


def to_builtin(value):
    if isinstance(value, dict):
        return {key: to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


class BoundedDeterministicAttack(DeterministicAttack):
    def __init__(self, Ga, A, max_iterations=5):
        super().__init__(Ga, A)
        self.max_iterations = max_iterations
        self.module_modifications = {}

    def run(self, run_matching=True, run_completion=True, run_degree=True, run_triangle=True):
        if run_degree:
            old = self.modifications
            self.degree_attack()
            self.module_modifications["degree_attack"] = self.modifications - old

        for _ in range(self.max_iterations):
            old_modifications = self.modifications
            if run_matching:
                old = self.modifications
                self.matching_attacks()
                self.module_modifications["matching_attacks"] = self.module_modifications.get("matching_attacks", 0) + self.modifications - old
                old = self.modifications
                self.degree_matching_attack()
                self.module_modifications["degree_matching_attack"] = self.module_modifications.get("degree_matching_attack", 0) + self.modifications - old
            if run_completion:
                old = self.modifications
                self.completion_attacks()
                self.module_modifications["completion_attacks"] = self.module_modifications.get("completion_attacks", 0) + self.modifications - old
                old = self.modifications
                self.degree_completion_attack()
                self.module_modifications["degree_completion_attack"] = self.module_modifications.get("degree_completion_attack", 0) + self.modifications - old
            if run_triangle:
                old = self.modifications
                self.triangle_attack()
                self.module_modifications["triangle_attack"] = self.module_modifications.get("triangle_attack", 0) + self.modifications - old
            if self.modifications == old_modifications:
                break


def perturb_graph_edgerefine(adj_original: np.ndarray, epsilon: float) -> np.ndarray:
    edge_num_origin = int(np.sum(adj_original != 0) / 2)
    adj_rr = symmetric_randomized_response_perturbation(adj_original.copy(), epsilon)
    prob_matrix = jaccard_probability(adj_rr.copy(), method="histogram")
    return graph_sampling_with_ratio(adj_rr, prob_matrix, epsilon, edge_num_origin)


def run_grand_attack(original_adj: np.ndarray, perturbed_adj: np.ndarray) -> dict:
    n = original_adj.shape[0]
    original_graph = Graph.from_adj_matrix(original_adj, with_fixed_edges=True)
    perturbed_graph = Graph.from_adj_matrix(perturbed_adj, with_fixed_edges=True)
    second_order = np.dot(perturbed_adj, perturbed_adj)

    edges_i, edges_j = np.triu_indices(n, k=1)
    mean_degree = np.mean(np.sum(perturbed_adj, axis=1))
    threshold = max(2, int(mean_degree * 0.3))

    is_non_edge = perturbed_adj[edges_i, edges_j] == 0
    hidden_mask = is_non_edge & (second_order[edges_i, edges_j] >= threshold)
    hidden_i, hidden_j = edges_i[hidden_mask], edges_j[hidden_mask]

    is_edge = perturbed_adj[edges_i, edges_j] == 1
    fake_mask = is_edge & (second_order[edges_i, edges_j] == 0)
    fake_i, fake_j = edges_i[fake_mask], edges_j[fake_mask]

    for i, j in zip(hidden_i, hidden_j):
        perturbed_graph.non_adj_list[i].discard(j)
        perturbed_graph.non_adj_list[j].discard(i)
        perturbed_graph.unknown_list[i].add(j)
        perturbed_graph.unknown_list[j].add(i)

    for i, j in zip(fake_i, fake_j):
        perturbed_graph.adj_list[i].discard(j)
        perturbed_graph.adj_list[j].discard(i)
        perturbed_graph.unknown_list[i].add(j)
        perturbed_graph.unknown_list[j].add(i)

    if hasattr(perturbed_graph, "_adj_matrix_cache"):
        perturbed_graph._adj_matrix_cache = None

    attack = BoundedDeterministicAttack(perturbed_graph, second_order, max_iterations=5)
    attack.run()
    reconstructed = attack.get_Gstar()
    metrics = compute_reconstruction_metrics(reconstructed, original_graph)
    original_edges = len(original_graph.edges()) // 2
    metrics["n_nodes"] = n
    metrics["n_edges_original"] = original_edges
    metrics["n_edges_perturbed"] = len(perturbed_graph.edges()) // 2
    metrics["n_edges_reconstructed"] = len(reconstructed.edges()) // 2
    metrics["privacy_gain"] = privacy_gain_over_random(reconstructed, original_graph)
    metrics["edge_preservation_rate"] = metrics["TP"] / original_edges if original_edges > 0 else 0.0
    metrics["total_grand_modifications"] = attack.modifications
    metrics["grand_module_modifications"] = attack.module_modifications
    return metrics


def main() -> None:
    output_dir = ensure_output_dir("exp_grand_attack")
    log_path = output_dir / "grand_attack_log.txt"
    json_path = output_dir / "grand_attack_results.json"
    csv_path = output_dir / "grand_attack_results.csv"
    all_results = []

    with log_path.open("w", encoding="utf-8") as log_file:
        for dataset_name in DATASETS:
            _, _, original_adj = load_graph_data(dataset_name, show_details=False)
            for epsilon in EPSILON_VALUES:
                set_seed(SEED)
                log_file.write(f"dataset={dataset_name}, epsilon={epsilon}\n")
                perturbed_adj = perturb_graph_edgerefine(original_adj, epsilon)
                metrics = run_grand_attack(original_adj, perturbed_adj)
                result = to_builtin({"dataset": dataset_name, "epsilon": epsilon, **metrics})
                all_results.append(result)
                log_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(
                    f"dataset={dataset_name}, epsilon={epsilon}, edge_accuracy={metrics['edge_accuracy']:.4f}, rae={metrics['rae']:.4f}"
                )

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2, ensure_ascii=False)
    pd.DataFrame(all_results).drop(columns=["grand_module_modifications"]).to_csv(csv_path, index=False)
    print(f"Saved results to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
