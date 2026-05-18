from __future__ import annotations

import numpy as np


class Graph:
    def __init__(self, size: int, with_fixed_edges: bool = False):
        self.size = size
        self.nodes = set(range(size))
        self.adj_list = [set() for _ in range(size)]
        self.non_adj_list = [set() for _ in range(size)]
        self.unknown_list = [set() for _ in range(size)]

    def write_value(self, edge, value):
        n1, n2 = edge
        self._adj_matrix_cache = None
        if value == 1:
            self.adj_list[n1].add(n2)
            self.adj_list[n2].add(n1)
            self.unknown_list[n1].discard(n2)
            self.unknown_list[n2].discard(n1)
        elif value == 0:
            self.adj_list[n1].discard(n2)
            self.adj_list[n2].discard(n1)
            self.unknown_list[n1].discard(n2)
            self.unknown_list[n2].discard(n1)
        elif value == 2:
            self.adj_list[n1].discard(n2)
            self.adj_list[n2].discard(n1)
            self.unknown_list[n1].add(n2)
            self.unknown_list[n2].add(n1)

    def add_edge(self, edge):
        self.write_value(edge, 1)

    def remove_edge(self, edge):
        self.write_value(edge, 0)

    def neighbors(self, node):
        return self.adj_list[node]

    def does_not_know_edge(self, edge):
        return edge[1] in self.unknown_list[edge[0]]

    def degree(self, node):
        return len(self.neighbors(node))

    def common_neighbors(self, edge):
        return self.neighbors(edge[0]).intersection(self.neighbors(edge[1]))

    def edges(self):
        return [[i, j] for i in range(self.size) for j in range(self.size) if i in self.adj_list[j]]

    def edges_no_repeat(self):
        return [[i, j] for i in range(self.size) for j in range(i, self.size) if i in self.adj_list[j]]

    def adjacency_matrix(self, use_cache: bool = True):
        if use_cache and hasattr(self, "_adj_matrix_cache") and self._adj_matrix_cache is not None:
            return self._adj_matrix_cache.copy()
        adj_matrix = np.zeros((self.size, self.size), dtype=np.int8)
        for i in range(self.size):
            for j in self.adj_list[i]:
                adj_matrix[i, j] = 1
            for j in self.unknown_list[i]:
                adj_matrix[i, j] = 2
        if use_cache:
            self._adj_matrix_cache = adj_matrix
        return adj_matrix

    def stats(self):
        num_absent, num_present, num_unknown = 0, 0, 0
        for i in range(self.size):
            for j in range(i, self.size):
                inc = 1 if i == j else 2
                if i in self.adj_list[j]:
                    num_present += inc
                elif i in self.unknown_list[j]:
                    num_unknown += inc
                else:
                    num_absent += inc
        return num_absent, num_present, num_unknown

    def copy(self):
        copied = Graph(self.size, with_fixed_edges=True)
        copied.adj_list = [self.adj_list[i].copy() for i in range(self.size)]
        copied.non_adj_list = [set() for _ in range(self.size)]
        copied.unknown_list = [self.unknown_list[i].copy() for i in range(self.size)]
        copied._adj_matrix_cache = None
        return copied

    @staticmethod
    def from_adj_matrix(adj_matrix, with_fixed_edges: bool = False):
        graph = Graph(len(adj_matrix), with_fixed_edges=with_fixed_edges)
        for i in range(len(adj_matrix)):
            for j in range(i + 1, len(adj_matrix)):
                graph.write_value((i, j), adj_matrix[i][j])
        return graph


class DeterministicAttack:
    def __init__(self, Ga, A, graph1_prop=0.0, dataset_name=None, log=False, expe_number=None):
        self.Gstar = Ga.copy()
        self.A = A
        self.modifications = 0

    def matching_attacks(self):
        modifs = 0
        neighbors = [self.Gstar.neighbors(node) for node in self.Gstar.nodes]
        for i in range(len(self.Gstar.nodes)):
            neighbors_i = neighbors[i]
            for j in range(i + 1, len(self.Gstar.nodes)):
                neighbors_j = neighbors[j]
                common_neighbors = neighbors_i & neighbors_j
                if self.A[i, j] == len(common_neighbors):
                    for node in neighbors_i.copy():
                        if self.Gstar.does_not_know_edge((j, node)):
                            self.Gstar.remove_edge((j, node))
                            modifs += 2
                    for node in neighbors_j.copy():
                        if self.Gstar.does_not_know_edge((i, node)):
                            self.Gstar.remove_edge((i, node))
                            modifs += 2
        self.modifications += modifs

    def degree_matching_attack(self):
        modifs = 0
        nodes = [node for node in self.Gstar.nodes if self.A[node, node] == self.Gstar.degree(node)]
        for node in nodes:
            for other in self.Gstar.unknown_list[node].copy():
                self.Gstar.remove_edge((node, other))
                modifs += 2
        self.modifications += modifs

    def completion_attacks(self):
        modifs = 0
        for i in range(len(self.Gstar.nodes)):
            neighbors_i = self.Gstar.neighbors(i)
            for j in range(i + 1, len(self.Gstar.nodes)):
                neighbors_j = self.Gstar.neighbors(j)
                unknown_i = self.Gstar.unknown_list[i]
                unknown_j = self.Gstar.unknown_list[j]
                if len(neighbors_i) == self.A[i, j] - len(unknown_i):
                    for k in unknown_i.copy():
                        self.Gstar.add_edge((i, k))
                        self.Gstar.add_edge((j, k))
                        modifs += 4
                if len(neighbors_j) == self.A[i, j] - len(unknown_j):
                    for k in unknown_j.copy():
                        self.Gstar.add_edge((j, k))
                        self.Gstar.add_edge((i, k))
                        modifs += 4
        self.modifications += modifs

    def degree_completion_attack(self):
        modifs = 0
        for i in range(len(self.Gstar.nodes)):
            if len(self.Gstar.neighbors(i)) == self.A[i, i] - len(self.Gstar.unknown_list[i]):
                for j in self.Gstar.unknown_list[i].copy():
                    self.Gstar.add_edge((i, j))
                    modifs += 2
        self.modifications += modifs

    def degree_attack(self):
        modifs = 0
        degree_sequence = np.diag(self.A)
        for i in range(self.A.shape[0]):
            degree = np.sum(self.A[i])
            candidate = None
            possibilities = np.where((degree_sequence <= degree - self.A[i, i] + 1))[0]
            non_possibilities = np.where((degree_sequence > degree - self.A[i, i] + 1))[0]
            if int(self.A[i, i]) <= 2:
                from itertools import combinations

                for comb in combinations(possibilities, int(self.A[i, i])):
                    if sum(self.A[k, k] for k in comb) == degree:
                        candidate = comb if candidate is None else None
                        break
                if candidate is not None:
                    for j in candidate:
                        if self.Gstar.does_not_know_edge((i, j)):
                            self.Gstar.add_edge((i, j))
                            modifs += 2
            for j in non_possibilities:
                if self.Gstar.does_not_know_edge((i, j)):
                    self.Gstar.remove_edge((i, j))
                    modifs += 2
        self.modifications += modifs

    def triangle_attack(self):
        modifs = 0
        edges = self.Gstar.edges_no_repeat()
        for u, v in edges:
            if self.A[u, v] == 0 or self.A[u, u] < 2 or self.A[v, v] < 2:
                continue
            g2_u = np.setdiff1d(np.where(self.A[u] > 0)[0], u)
            g2_v = np.setdiff1d(np.where(self.A[v] > 0)[0], v)
            candidates = np.setdiff1d(np.intersect1d(g2_u, g2_v), self.Gstar.common_neighbors((u, v)))
            if len(candidates) == self.A[u, v] - len(self.Gstar.common_neighbors((u, v))):
                for w in candidates:
                    if self.Gstar.does_not_know_edge((u, w)):
                        self.Gstar.add_edge((u, w))
                        modifs += 2
                    if self.Gstar.does_not_know_edge((v, w)):
                        self.Gstar.add_edge((v, w))
                        modifs += 2
        self.modifications += modifs

    def run(self, run_matching=True, run_completion=True, run_degree=True, run_triangle=True, max_iterations=5):
        if run_degree:
            self.degree_attack()
        for _ in range(max_iterations):
            old = self.modifications
            if run_matching:
                self.matching_attacks()
                self.degree_matching_attack()
            if run_completion:
                self.completion_attacks()
                self.degree_completion_attack()
            if run_triangle:
                self.triangle_attack()
            if self.modifications == old:
                break

    def get_Gstar(self):
        return self.Gstar


def compute_reconstruction_metrics(prediction, groundtruth):
    pred_adj = prediction.adjacency_matrix()
    truth_adj = groundtruth.adjacency_matrix()
    gsquare = np.dot(truth_adj, truth_adj)
    gsquare_pred = np.dot(pred_adj, pred_adj)
    accuracy = np.sum(np.abs(gsquare - gsquare_pred)) / np.sum(gsquare)
    distance = np.linalg.norm(pred_adj - truth_adj, "fro")
    rae_stat = distance ** 2 / np.linalg.norm(truth_adj, "fro") ** 2
    pred_edges = {(i, j) for i in range(prediction.size) for j in prediction.adj_list[i] if i < j}
    truth_edges = {(i, j) for i in range(groundtruth.size) for j in groundtruth.adj_list[i] if i < j}
    edge_accuracy = len(pred_edges & truth_edges) / len(truth_edges) if truth_edges else 0
    tp = int(np.sum(np.logical_and(pred_adj == 1, truth_adj == 1)))
    fp = int(np.sum(np.logical_and(pred_adj == 1, truth_adj == 0)))
    tn = int(np.sum(np.logical_and(pred_adj == 0, truth_adj == 0)))
    fn = int(np.sum(np.logical_and(pred_adj == 0, truth_adj == 1)))
    return {
        "accuracy": accuracy,
        "frobenius_distance": distance,
        "rae": rae_stat,
        "edge_accuracy": edge_accuracy,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def privacy_gain_over_random(prediction, groundtruth):
    n = groundtruth.size
    num_edges = len(groundtruth.edges()) // 2
    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return 0.0
    pred_edges = {(i, j) for i in range(prediction.size) for j in prediction.adj_list[i] if i < j}
    truth_edges = {(i, j) for i in range(groundtruth.size) for j in groundtruth.adj_list[i] if i < j}
    return (len(pred_edges & truth_edges) / len(truth_edges) if truth_edges else 0) - (num_edges / total_pairs)
