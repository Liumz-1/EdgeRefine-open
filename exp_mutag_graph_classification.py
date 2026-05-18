from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from experiment_utils import ensure_output_dir, set_seed
from model import GATNet, GCNNet, GINNet
from structure import graph_sampling_with_ratio, jaccard_probability, symmetric_randomized_response_perturbation

try:
    from tabulate import tabulate

    USE_TABULATE = True
except ImportError:
    USE_TABULATE = False


DATASET_NAME = "MUTAG"
NUM_RUNS = 5
EPSILONS = ["baseline", 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
MODELS = ["GAT", "GCN", "GIN"]


class NodeToGraphWrapper(nn.Module):
    def __init__(self, base_node_model, num_classes):
        super().__init__()
        self.base_model = base_node_model
        self.final_fc = nn.LazyLinear(num_classes)

    def forward(self, x, edge_index, batch):
        pseudo_data = Data(x=x, edge_index=edge_index)
        node_out = self.base_model(pseudo_data)
        graph_emb = global_mean_pool(node_out, batch)
        return self.final_fc(graph_emb)


def resolve_tudataset_root() -> Path:
    current_root = Path(__file__).resolve().parent / "dataset" / "TUDataset"
    if current_root.exists():
        return current_root

    legacy_root = Path(r"E:\Edgerefine\EdgeRefine (3)\dataset\TUDataset")
    if legacy_root.exists():
        return legacy_root

    raise FileNotFoundError("TUDataset root not found in openscience(2) or EdgeRefine (3).")


def perturb_data_list(data_list, epsilon):
    if epsilon == "baseline":
        return [copy.deepcopy(data) for data in data_list]

    perturbed_list = []
    for data in data_list:
        num_nodes = data.num_nodes
        adj = np.zeros((num_nodes, num_nodes), dtype=int)
        edge_index = data.edge_index.cpu().numpy()
        rows, cols = edge_index[0], edge_index[1]
        adj[rows, cols] = 1
        adj = np.maximum(adj, adj.T)

        adj_noisy = symmetric_randomized_response_perturbation(adj, epsilon)
        adj_prob = jaccard_probability(adj_noisy, method="histogram")
        original_edges = int(np.sum(adj) / 2)
        adj_final = graph_sampling_with_ratio(adj_noisy, adj_prob, epsilon, original_edges)

        rows_new, cols_new = np.where(adj_final > 0)
        mask = rows_new != cols_new
        rows_new, cols_new = rows_new[mask], cols_new[mask]
        new_edge_index = torch.tensor([rows_new, cols_new], dtype=torch.long)

        perturbed_data = copy.deepcopy(data)
        perturbed_data.edge_index = new_edge_index
        perturbed_list.append(perturbed_data)

    return perturbed_list


def train(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)


def run_experiment(model, train_loader, val_loader, test_loader, epochs=150):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    best_val_acc = 0.0
    best_test_acc = 0.0
    patience = 20
    trigger_times = 0

    for _ in range(1, epochs + 1):
        train(model, train_loader, optimizer)
        val_acc = test(model, val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test(model, test_loader)
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                break
    return best_test_acc


def create_wrapped_model(model_name: str, num_features: int, num_classes: int):
    if model_name == "GCN":
        base_model = GCNNet(num_feature=num_features, num_label=num_classes, hidden_dim=64)
    elif model_name == "GAT":
        base_model = GATNet(num_feature=num_features, num_label=num_classes, hidden_dim=32, heads=4)
    elif model_name == "GIN":
        base_model = GINNet(num_feature=num_features, num_label=num_classes, hidden_dim=64)
    else:
        raise ValueError(f"unknown model: {model_name}")
    return NodeToGraphWrapper(base_model, num_classes)


def main():
    output_dir = ensure_output_dir("exp_mutag_graph_classification")
    dataset_root = resolve_tudataset_root()
    all_results = defaultdict(lambda: {model_name: [] for model_name in MODELS})
    raw_rows = []

    print(f"Running {DATASET_NAME} graph classification experiment")
    print(f"TUDataset root: {dataset_root}")
    print(f"Total runs: {NUM_RUNS}")

    for run_id in range(NUM_RUNS):
        current_seed = 42 + run_id
        set_seed(current_seed)
        print(f"\n>>> Run {run_id + 1}/{NUM_RUNS} (seed={current_seed}) <<<")

        dataset = TUDataset(root=str(dataset_root), name=DATASET_NAME)
        num_features = dataset.num_features
        num_classes = dataset.num_classes

        dataset = dataset.shuffle()
        total_graphs = len(dataset)
        train_dataset = dataset[: int(total_graphs * 0.8)]
        val_dataset = dataset[int(total_graphs * 0.8): int(total_graphs * 0.9)]
        test_dataset = dataset[int(total_graphs * 0.9):]

        for epsilon in EPSILONS:
            train_pert = perturb_data_list(train_dataset, epsilon)
            val_pert = perturb_data_list(val_dataset, epsilon)
            test_pert = perturb_data_list(test_dataset, epsilon)

            train_loader = DataLoader(train_pert, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_pert, batch_size=32)
            test_loader = DataLoader(test_pert, batch_size=32)

            for model_name in MODELS:
                model = create_wrapped_model(model_name, num_features, num_classes)
                test_acc = run_experiment(model, train_loader, val_loader, test_loader)
                all_results[epsilon][model_name].append(test_acc)
                raw_rows.append(
                    {
                        "run_id": run_id + 1,
                        "seed": current_seed,
                        "epsilon": epsilon,
                        "model": model_name,
                        "test_accuracy": test_acc,
                    }
                )
                print(f"epsilon={epsilon}, model={model_name}, acc={test_acc:.4f}")

    summary_rows = []
    for epsilon in EPSILONS:
        row = {"epsilon": epsilon}
        for model_name in MODELS:
            accs = all_results[epsilon][model_name]
            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs))
            row[f"{model_name}_mean"] = mean_acc
            row[f"{model_name}_std"] = std_acc
        summary_rows.append(row)

    raw_path = output_dir / "mutag_graph_classification_raw_results.csv"
    summary_path = output_dir / "mutag_graph_classification_summary.csv"
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("MUTAG graph classification results (mean ± std)")
    print("=" * 60)
    display_rows = []
    for row in summary_rows:
        display_rows.append(
            [
                row["epsilon"],
                f"{row['GAT_mean']:.3f} ± {row['GAT_std']:.3f}",
                f"{row['GCN_mean']:.3f} ± {row['GCN_std']:.3f}",
                f"{row['GIN_mean']:.3f} ± {row['GIN_std']:.3f}",
            ]
        )

    headers = ["Epsilon", "GAT", "GCN", "GIN"]
    if USE_TABULATE:
        print(tabulate(display_rows, headers=headers, tablefmt="grid"))
    else:
        print(f"{headers[0]:<10} | {headers[1]:<20} | {headers[2]:<20} | {headers[3]:<20}")
        print("-" * 80)
        for row in display_rows:
            print(f"{str(row[0]):<10} | {row[1]:<20} | {row[2]:<20} | {row[3]:<20}")

    print(f"\nSaved results to {raw_path} and {summary_path}")


if __name__ == "__main__":
    main()
