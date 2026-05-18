from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from model import GATNet, GCNNet, GINNet
from structure import (
    DPRR,
    LAPGRAPH,
    LDPGen,
    blink_hard,
    blink_hybrid,
    graph_sampling_with_ratio,
    jaccard_probability,
    symmetric_randomized_response_perturbation,
)

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"

DEFAULT_DATASETS = ["dblp", "acm", "cora", "amap"]
DEFAULT_EPSILONS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
DEFAULT_MODELS = ["gat", "gcn", "gin"]
DEFAULT_SINGLE_SEED = 42
DEFAULT_METHOD_COMBOS = [
    ("jaccard", "simple"),
    ("jaccard", "isotonic"),
    ("jaccard", "beta"),
    ("jaccard", "temperature"),
    ("jaccard", "histogram"),
    ("adamic_adar", "histogram"),
]


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ensure_output_dir(experiment_name: str) -> Path:
    output_dir = RESULT_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_repeat_seeds(dataset_name: str) -> list[int]:
    return [DEFAULT_SINGLE_SEED]


@lru_cache(maxsize=None)
def _load_graph_data_cached(dataset_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    load_path = BASE_DIR / "dataset" / dataset_name / dataset_name
    feat = np.load(f"{load_path}_feat.npy", allow_pickle=True)
    label = np.load(f"{load_path}_label.npy", allow_pickle=True)
    adj = np.load(f"{load_path}_adj.npy", allow_pickle=True)
    return feat, label, adj


def load_graph_data(dataset_name: str, show_details: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feat, label, adj = _load_graph_data_cached(dataset_name)
    feat = feat.copy()
    label = label.copy()
    adj = adj.copy()
    if show_details:
        print(f"dataset name: {dataset_name}")
        print(f"feature shape: {feat.shape}")
        print(f"label shape: {label.shape}")
        print(f"adj shape: {adj.shape}")
        print(f"undirected edge num: {int(np.nonzero(adj)[0].shape[0] / 2)}")
        print(f"category num: {int(max(label) - min(label) + 1)}")
    return feat, label, adj


def build_masks(total_samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_samples, generator=generator)
    train_size = int(0.6 * total_samples)
    val_size = int(0.2 * total_samples)

    train_mask = torch.zeros(total_samples, dtype=torch.bool)
    val_mask = torch.zeros(total_samples, dtype=torch.bool)
    test_mask = torch.zeros(total_samples, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    return train_mask, val_mask, test_mask


def get_model(model_type: str, num_feature: int, num_label: int, device: torch.device):
    if model_type == "gat":
        return GATNet(num_feature=num_feature, num_label=num_label).to(device)
    if model_type == "gcn":
        return GCNNet(num_feature=num_feature, num_label=num_label).to(device)
    if model_type == "gin":
        return GINNet(num_feature=num_feature, num_label=num_label).to(device)
    raise ValueError(f"unknown model: {model_type}")


def normalize_adj_shape(adj: np.ndarray, target_nodes: int) -> np.ndarray:
    if adj.shape[0] == target_nodes:
        return adj
    if adj.shape[0] > target_nodes:
        return adj[:target_nodes, :target_nodes]
    normalized = np.zeros((target_nodes, target_nodes), dtype=adj.dtype)
    normalized[:adj.shape[0], :adj.shape[1]] = adj
    return normalized


def preprocess_edge_refine(adj: np.ndarray, epsilon: float, sample_ratio: float = 0.01) -> np.ndarray:
    edge_num_origin = int(np.sum(adj != 0) / 2)
    noisy_adj = symmetric_randomized_response_perturbation(adj.copy(), epsilon)
    prob_matrix = jaccard_probability(noisy_adj.copy(), method="histogram")
    target_total_edges = int(edge_num_origin * sample_ratio)
    return graph_sampling_with_ratio(noisy_adj, prob_matrix, epsilon, target_total_edges)


def preprocess_blink_hard(adj: np.ndarray, epsilon: float, eps_blink_d: float = 0.1) -> np.ndarray:
    return blink_hard(adj.copy(), epsilon, eps_d=eps_blink_d)


def preprocess_blink_hybrid(adj: np.ndarray, epsilon: float, eps_blink_d: float = 0.1) -> np.ndarray:
    return blink_hybrid(adj.copy(), epsilon, eps_d=eps_blink_d)


def preprocess_other_baseline(adj: np.ndarray, epsilon: float, baseline_name: str) -> np.ndarray:
    baseline_name = baseline_name.lower()
    if baseline_name == "dprr":
        return DPRR(adj.copy(), epsilon)
    if baseline_name == "lapgraph":
        return LAPGRAPH(adj.copy(), epsilon)
    if baseline_name == "ldpgen":
        return LDPGen(adj.copy(), epsilon, k=2)
    raise ValueError(f"unknown baseline: {baseline_name}")


def adj_to_edge_index_with_weights(adj_matrix: np.ndarray | torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = torch.FloatTensor(adj_matrix)
    edge_mask = adj_matrix != 0
    if not edge_mask.any():
        node_count = adj_matrix.size(0)
        edge_index = torch.arange(node_count).repeat(2, 1)
        edge_weight = torch.ones(node_count, dtype=torch.float)
    else:
        edge_index = edge_mask.nonzero().t().contiguous()
        edge_weight = adj_matrix[edge_mask]
    return edge_index.to(device), edge_weight.to(device)


def train_and_test_node_classifier(
    dataset_name: str,
    processed_adj: np.ndarray,
    model_type: str,
    seed: int = 42,
    use_weights: bool = False,
    device: torch.device | None = None,
    max_epochs: int = 700,
    patience: int = 100,
) -> float:
    device = device or get_device()
    set_seed(seed)
    feat, label, _ = load_graph_data(dataset_name, show_details=False)
    processed_adj = normalize_adj_shape(processed_adj, feat.shape[0])

    features = torch.FloatTensor(feat).to(device)
    labels = torch.LongTensor(label).to(device)
    data = Data(x=features, y=labels)

    if use_weights:
        edge_index, edge_weight = adj_to_edge_index_with_weights(processed_adj, device)
        data.edge_index = edge_index
        data.edge_weight = edge_weight
    else:
        adj_matrix = torch.FloatTensor(processed_adj)
        data.edge_index = (adj_matrix > 0).nonzero().t().contiguous().to(device)

    train_mask, val_mask, test_mask = build_masks(len(features), seed)
    data.train_mask = train_mask.to(device)
    data.val_mask = val_mask.to(device)
    data.test_mask = test_mask.to(device)

    model = get_model(model_type, feat.shape[1], len(np.unique(label)), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.00001)

    best_val_acc = 0.0
    patience_counter = 0

    for _ in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        try:
            if use_weights and hasattr(data, "edge_weight") and hasattr(model, "forward_with_weights"):
                out = model.forward_with_weights(data.x, data.edge_index, data.edge_weight)
            else:
                out = model(data)
        except Exception:
            out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            try:
                if use_weights and hasattr(data, "edge_weight") and hasattr(model, "forward_with_weights"):
                    pred = model.forward_with_weights(data.x, data.edge_index, data.edge_weight).argmax(dim=1)
                else:
                    pred = model(data).argmax(dim=1)
            except Exception:
                pred = model(data).argmax(dim=1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    model.eval()
    with torch.no_grad():
        try:
            if use_weights and hasattr(data, "edge_weight") and hasattr(model, "forward_with_weights"):
                pred = model.forward_with_weights(data.x, data.edge_index, data.edge_weight).argmax(dim=1)
            else:
                pred = model(data).argmax(dim=1)
        except Exception:
            pred = model(data).argmax(dim=1)
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return test_acc


def get_neg_edges_from_adj(adj_matrix: np.ndarray, num_samples: int) -> torch.Tensor:
    n = adj_matrix.shape[0]
    adj_no_diag = adj_matrix.copy()
    np.fill_diagonal(adj_no_diag, 0)
    rows, cols = np.where(np.triu(adj_no_diag == 0, k=1))
    candidates = np.stack([rows, cols], axis=1)
    replace = len(candidates) < num_samples
    chosen_indices = np.random.choice(len(candidates), size=num_samples, replace=replace)
    return torch.tensor(candidates[chosen_indices].T, dtype=torch.long)


def get_node_embeddings(model, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    x = model.conv1(x, edge_index)
    x = F.relu(x)
    x = F.dropout(x, p=model.dropout, training=model.training)
    x = model.conv2(x, edge_index)
    return x
