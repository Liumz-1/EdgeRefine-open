from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data

from experiment_utils import (
    ensure_output_dir,
    get_device,
    get_neg_edges_from_adj,
    get_node_embeddings,
    load_graph_data,
    preprocess_edge_refine,
    remove_edges_from_adj,
    set_seed,
    split_link_prediction_edges,
)
from model import GATNet

DATASET_NAME = "dblp"
EPSILON_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
SAMPLING_RATIO = 0.05
SEED = 42


def main() -> None:
    output_dir = ensure_output_dir("exp_link_prediction")
    device = get_device()
    set_seed(SEED)

    feat, label, original_adj = load_graph_data(DATASET_NAME, show_details=False)
    train_original_adj, test_pos_edge_index = split_link_prediction_edges(original_adj, test_ratio=0.5, seed=SEED)
    test_neg_edge_index = get_neg_edges_from_adj(original_adj, test_pos_edge_index.size(1))
    features = torch.FloatTensor(feat).to(device)
    results = []

    for epsilon in EPSILON_VALUES:
        set_seed(SEED)
        processed_adj = preprocess_edge_refine(train_original_adj, epsilon, sample_ratio=SAMPLING_RATIO)
        processed_adj = remove_edges_from_adj(processed_adj, test_pos_edge_index)
        processed_edge_num = int(np.sum(processed_adj != 0) / 2)
        train_edge_index = (torch.FloatTensor(processed_adj) > 0).nonzero().t().contiguous().to(device)
        data = Data(x=features, edge_index=train_edge_index)
        model = GATNet(num_feature=feat.shape[1], num_label=len(np.unique(label))).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.00001)
        negative_sampling_adj = np.logical_or(processed_adj != 0, original_adj != 0)
        train_neg_edge = get_neg_edges_from_adj(negative_sampling_adj, train_edge_index.size(1)).to(device)

        model.train()
        for _ in range(1, 201):
            optimizer.zero_grad()
            embeddings = get_node_embeddings(model, data.x, data.edge_index)
            pos_out = torch.sum(embeddings[train_edge_index[0]] * embeddings[train_edge_index[1]], dim=1)
            neg_out = torch.sum(embeddings[train_neg_edge[0]] * embeddings[train_neg_edge[1]], dim=1)
            out = torch.cat([pos_out, neg_out])
            gt_labels = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]).to(device)
            loss = F.binary_cross_entropy_with_logits(out, gt_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            embeddings = get_node_embeddings(model, data.x, data.edge_index)
            test_pos_out = torch.sum(
                embeddings[test_pos_edge_index[0].to(device)] * embeddings[test_pos_edge_index[1].to(device)], dim=1
            )
            test_neg_out = torch.sum(
                embeddings[test_neg_edge_index[0].to(device)] * embeddings[test_neg_edge_index[1].to(device)], dim=1
            )
            test_pred = torch.cat([test_pos_out, test_neg_out]).cpu().numpy()
            test_true = np.concatenate([np.ones(len(test_pos_out)), np.zeros(len(test_neg_out))])
            auc = roc_auc_score(test_true, test_pred)
            ap = average_precision_score(test_true, test_pred)

        results.append(
            {
                "dataset": DATASET_NAME,
                "epsilon": epsilon,
                "sampling_ratio": SAMPLING_RATIO,
                "processed_edge_num": processed_edge_num,
                "auc": auc,
                "ap": ap,
            }
        )
        print(f"epsilon={epsilon}, auc={auc:.4f}, ap={ap:.4f}")

    pd.DataFrame(results).to_csv(output_dir / "link_prediction_results.csv", index=False)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
