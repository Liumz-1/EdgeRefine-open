import numpy as np
import math
from sklearn.preprocessing import normalize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import beta
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
import networkx as nx

def symmetric_randomized_response_perturbation(adj_matrix, epsilon):

    if not isinstance(adj_matrix, np.ndarray):
        raise TypeError("error: adj_matrix must be a numpy array")
    if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("error: adj_matrix must be a square matrix")
    if epsilon < 0:
        raise ValueError("error: epsilon must be non-negative")


    if not np.allclose(adj_matrix, adj_matrix.T):
        raise ValueError("error: adj_matrix must be symmetric")

    n = adj_matrix.shape[0]
    p_flip = 1.0 / (1.0 + math.exp(epsilon))
    random_values = np.random.uniform(0, 1, size=(n, n))
    perturbed_adj = np.zeros_like(adj_matrix)
    triu_indices = np.triu_indices(n, k=1)

    upper_triangle_original = adj_matrix[triu_indices]
    random_upper = random_values[triu_indices]

    upper_triangle_perturbed = np.where(random_upper < p_flip,
                                        1 - upper_triangle_original,
                                        upper_triangle_original)


    perturbed_adj[triu_indices] = upper_triangle_perturbed


    lower_indices = (triu_indices[1], triu_indices[0])
    perturbed_adj[lower_indices] = upper_triangle_perturbed


    np.fill_diagonal(perturbed_adj, 0)

    return perturbed_adj.astype(np.int64)

def compute_jaccard_matrix(adj, block_size=512):
    """Compute the legacy all-pairs Jaccard matrix in bounded-memory blocks.

    For a binary adjacency matrix, ``A @ A.T`` gives the number of common
    neighbors.  Processing rows in blocks avoids the Python-level pair loop
    and does not require a second full-size intersection matrix.
    """
    adj = np.asarray(adj)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("error: adj must be a square matrix")
    if block_size <= 0:
        raise ValueError("error: block_size must be positive")

    n = adj.shape[0]
    binary_adj = (adj == 1).astype(np.float32, copy=False)
    degrees = np.sum(binary_adj, axis=1, dtype=np.float64)
    jaccard = np.zeros((n, n), dtype=np.float64)

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        intersections = binary_adj[start:stop] @ binary_adj.T
        unions = degrees[start:stop, None] + degrees[None, :] - intersections
        block = np.zeros(intersections.shape, dtype=np.float64)
        np.divide(intersections, unions, out=block, where=unions != 0)
        # Preserve the scaling produced by the previous upper-triangle
        # symmetrization so existing experiment outputs remain comparable.
        block *= 0.5
        jaccard[start:stop] = block

    np.fill_diagonal(jaccard, 0.0)
    return jaccard


def adamic_adar_prob_matrix(adj_matrix, n_bins=10):
    """Estimate link probabilities with Adamic-Adar scores plus histogram calibration."""
    adj_matrix = np.asarray(adj_matrix, dtype=np.float64)
    n = adj_matrix.shape[0]
    degrees = np.sum(adj_matrix, axis=1) - np.diag(adj_matrix)
    adamic_adar = np.zeros((n, n))

    for i in range(n):
        neighbors_i = np.where(adj_matrix[i] == 1)[0]
        for j in range(i + 1, n):
            neighbors_j = np.where(adj_matrix[j] == 1)[0]
            common = np.intersect1d(neighbors_i, neighbors_j)
            total = 0.0
            for node in common:
                degree = degrees[node]
                if degree > 1:
                    total += 1.0 / np.log(degree)
            adamic_adar[i, j] = total
            adamic_adar[j, i] = total

    feature_values = adamic_adar[np.triu_indices(n, k=1)].flatten()
    labels = (adj_matrix[np.triu_indices(n, k=1)] > 0).flatten()
    if np.all(feature_values == 0):
        return np.zeros((n, n))

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    bins = discretizer.fit_transform(feature_values.reshape(-1, 1)).flatten().astype(int)
    bin_probs = []
    for bin_idx in range(n_bins):
        mask = bins == bin_idx
        bin_probs.append(np.mean(labels[mask]) if np.any(mask) else 0.5)
    prob_flat = np.asarray(bin_probs)[bins]
    return _rebuild_matrix(n, prob_flat)


def jaccard_probability(adj, method="sigmoid", max_j=None, random_state=None, **kwargs):


    n = adj.shape[0]
    np.fill_diagonal(adj, 0)
    jaccard = compute_jaccard_matrix(adj)


    if np.all(jaccard == 0):
        return np.zeros((n, n))


    if method == "simple":
        if max_j is None:
            upper_tri = jaccard[np.triu_indices(n, k=1)]
            max_j = np.max(upper_tri) if np.any(upper_tri > 0) else 1.0
        prob = jaccard / (max_j + 1e-10)
        prob = np.clip(prob, 0, 1)

    elif method == "sigmoid":
        X = jaccard[np.triu_indices(n, k=1)].flatten()
        y = (adj[np.triu_indices(n, k=1)] > 0).flatten()
        if np.all(X == 0) or np.all(X == 1) or len(np.unique(y)) < 2:
            prob = np.zeros((n, n))
        else:
            model = LogisticRegression(random_state=random_state, class_weight='balanced')
            model.fit(X.reshape(-1, 1), y)
            prob_flat = model.predict_proba(X.reshape(-1, 1))[:, 1]
            prob = _rebuild_matrix(n, prob_flat)

    elif method == "isotonic":
        X = jaccard[np.triu_indices(n, k=1)].flatten()
        y = (adj[np.triu_indices(n, k=1)] > 0).flatten()
        if np.all(X == 0) or np.all(X == 1) or len(np.unique(y)) < 2:
            prob = np.zeros((n, n))
        else:
            ir = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
            ir.fit(X, y)
            prob_flat = ir.transform(X)
            prob = _rebuild_matrix(n, prob_flat)

    elif method == "beta":
        X = jaccard[np.triu_indices(n, k=1)].flatten()
        y = (adj[np.triu_indices(n, k=1)] > 0).flatten().astype(bool)
        if np.all(X == 0) or np.all(X == 1) or len(np.unique(y)) < 2:
            prob = np.zeros((n, n))
        else:
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=random_state)
            X_train = np.clip(X_train, 1e-6, 1 - 1e-6)
            pos_params = beta.fit(X_train[y_train], floc=0, fscale=1)
            neg_params = beta.fit(X_train[~y_train], floc=0, fscale=1)
            prior_pos = np.mean(y_train)
            X_safe = np.clip(X, 1e-6, 1 - 1e-6)
            prob_pos = beta.pdf(X_safe, *pos_params) * prior_pos
            prob_neg = beta.pdf(X_safe, *neg_params) * (1 - prior_pos)
            prob_flat = prob_pos / (prob_pos + prob_neg + 1e-10)
            prob = _rebuild_matrix(n, prob_flat)

    elif method == "temperature":
        X = np.clip(jaccard[np.triu_indices(n, k=1)].flatten(), 1e-6, 1 - 1e-6)
        y = (adj[np.triu_indices(n, k=1)] > 0).flatten()
        if np.all(X == 0) or np.all(X == 1) or len(np.unique(y)) < 2:
            prob = np.zeros((n, n))
        else:
            def loss(temp):
                log_odds = np.log(X / (1 - X)) / temp
                prob_pred = 1.0 / (1.0 + np.exp(-log_odds))
                prob_pred = np.clip(prob_pred, 1e-10, 1 - 1e-10)
                return -np.mean(y * np.log(prob_pred) + (1 - y) * np.log(1 - prob_pred))

            result = minimize_scalar(loss, bounds=(0.1, 10), method="bounded", options={"xatol": 1e-3})
            temp = result.x
            log_odds = np.log(X / (1 - X)) / temp
            prob_flat = 1.0 / (1.0 + np.exp(-log_odds))
            prob = _rebuild_matrix(n, prob_flat)

    elif method == "histogram":

        n_bins = kwargs.get('n_bins', 10)
        X = jaccard[np.triu_indices(n, k=1)].flatten()
        y = (adj[np.triu_indices(n, k=1)] > 0).flatten()
        if np.all(X == 0) or np.all(X == 1):
            prob = np.zeros((n, n))
        else:

            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
            bins = discretizer.fit_transform(X.reshape(-1, 1)).flatten().astype(int)

            bin_probs = []
            for bin_idx in range(n_bins):
                mask = (bins == bin_idx)
                if np.any(mask):
                    prob_bin = np.mean(y[mask])
                else:
                    prob_bin = 0.5
                bin_probs.append(prob_bin)
            prob_flat = np.asarray(bin_probs)[bins]
            prob = _rebuild_matrix(n, prob_flat)

    else:
        raise ValueError(
            "error: method must be one of 'simple', 'sigmoid', 'isotonic', 'beta', 'temperature', or 'histogram'"
        )

    return prob

def _rebuild_matrix(n, prob_flat):
    prob = np.zeros((n, n))
    upper = np.triu_indices(n, k=1)
    prob[upper] = prob_flat
    prob[(upper[1], upper[0])] = prob_flat
    return prob

def _blink_posterior(noisy_adj, noisy_deg, eps_a):

    n = noisy_adj.shape[0]


    noisy_deg_clipped = np.clip(noisy_deg, 1, n - 2)


    def estimate_prior(deg_seq):

        beta = np.zeros((n, 1))


        def phi(x):
            exp_x = np.exp(x)
            exp_minus_x = np.exp(-x)


            denominator = 1.0 / (np.dot(exp_x, np.ones((1, n))) + np.dot(np.ones((n, 1)), exp_minus_x.T))
            r = denominator - np.diag(np.diag(denominator))


            return np.log(deg_seq.reshape(-1, 1)) - np.log(np.sum(r, axis=1).reshape(-1, 1)) - np.diag(r).reshape(-1, 1)


        for _ in range(200):
            beta = phi(beta)


        s = np.dot(beta, np.ones((1, n))) + np.dot(np.ones((n, 1)), beta.T)
        prior = np.exp(s) / (1 + np.exp(s))
        np.fill_diagonal(prior, 0)
        return prior

    prior = estimate_prior(noisy_deg_clipped)
    p_flip = 1.0 / (1.0 + math.exp(eps_a))
    x = noisy_adj + noisy_adj.T

    pr_y_edge = (
            0.5 * (x - 1) * (x - 2) * p_flip ** 2 +
            0.5 * x * (x - 1) * (1 - p_flip) ** 2 -
            1 * x * (x - 2) * p_flip * (1 - p_flip)
    )

    pr_y_no_edge = (
            0.5 * (x - 1) * (x - 2) * (1 - p_flip) ** 2 +
            0.5 * x * (x - 1) * p_flip ** 2 -
            1 * x * (x - 2) * p_flip * (1 - p_flip)
    )

    return (pr_y_edge * prior) / (pr_y_edge * prior + pr_y_no_edge * (1 - prior) + 1e-10)


def blink_hard_processing(noisy_adj, noisy_deg, eps_a, eps_d):
    pij = _blink_posterior(noisy_adj, noisy_deg, eps_a)
    result_adj = (pij > 0.5).astype(np.float64)
    return result_adj


def blink_hybrid_processing(noisy_adj, noisy_deg, eps_a, eps_d):
    pij = _blink_posterior(noisy_adj, noisy_deg, eps_a)
    return np.where(pij > 0.5, pij, 0.0)


def blink_hard(adj, epsilon, eps_d=0.1):
    """
    Run the complete BLINK-hard pipeline from the original adjacency matrix.
    Degree-sequence noise uses eps_d, and adjacency RR uses epsilon - eps_d.
    """
    if epsilon <= eps_d:
        raise ValueError("error: epsilon must be greater than eps_d for BLINK")
    eps_a = epsilon - eps_d
    degree = generate_degree_sequence(matrix_to_graph(adj), eps_d)
    noisy_adj = symmetric_randomized_response_perturbation(np.asarray(adj).copy(), eps_a)
    return blink_hard_processing(noisy_adj, degree, eps_a, eps_d)


def blink_hybrid(adj, epsilon, eps_d=0.1):
    """
    Run the complete BLINK-hybrid pipeline from the original adjacency matrix.
    Degree-sequence noise uses eps_d, and adjacency RR uses epsilon - eps_d.
    """
    if epsilon <= eps_d:
        raise ValueError("error: epsilon must be greater than eps_d for BLINK")
    eps_a = epsilon - eps_d
    degree = generate_degree_sequence(matrix_to_graph(adj), eps_d)
    noisy_adj = symmetric_randomized_response_perturbation(np.asarray(adj).copy(), eps_a)
    return blink_hybrid_processing(noisy_adj, degree, eps_a, eps_d)

def graph_sampling_with_ratio(A, L, epsilon, target_total_edges):

    N = A.shape[0]

    FA = 1 - A

    A_perturbed = np.zeros((N, N), dtype=int)


    exp_eps = np.exp(epsilon)
    ratio_real = exp_eps / (1 + exp_eps)
    ratio_fake = 1 - ratio_real

    print(f"Based on ε={epsilon}: Target ratio - {ratio_real:.3f} real edges, {ratio_fake:.3f} fake edges")


    K_real = int(round(target_total_edges * ratio_real))
    K_fake = int(target_total_edges - K_real)


    K_real = max(0, min(K_real, np.sum(A) // 2))
    K_fake = max(0, min(K_fake, N * (N - 1) // 2 - np.sum(A) // 2))

    print(f"Target edges: {K_real} real + {K_fake} fake = {K_real + K_fake} total")


    real_edge_candidates = []
    fake_edge_candidates = []
    for i in range(N):
        for j in range(i + 1, N):
            weight = L[i, j]
            if A[i, j] == 1:
                real_edge_candidates.append((i, j, weight))
            if FA[i, j] == 1:
                fake_edge_candidates.append((i, j, weight))


    real_edge_candidates.sort(key=lambda x: x[2], reverse=True)
    fake_edge_candidates.sort(key=lambda x: x[2], reverse=True)


    sampled_real_edges = real_edge_candidates[:K_real] if real_edge_candidates else []
    for (i, j, _) in sampled_real_edges:
        A_perturbed[i, j] = 1
        A_perturbed[j, i] = 1


    sampled_fake_edges = fake_edge_candidates[:K_fake] if fake_edge_candidates else []
    for (i, j, _) in sampled_fake_edges:
        A_perturbed[i, j] = 1
        A_perturbed[j, i] = 1  #


    actual_total = np.sum(A_perturbed) // 2
    actual_ratio_real = len(sampled_real_edges) / actual_total if actual_total > 0 else 0
    print(
        f"Actual result: {len(sampled_real_edges)} real + {len(sampled_fake_edges)} fake = {actual_total} total edges")
    print(f"Actual ratio: {actual_ratio_real:.3f} real edges")
    return A_perturbed

def generate_degree_sequence(G, eps):

    assert G.number_of_nodes() == len(G.nodes()), "Number of nodes in graph does not match length of node list"
    original_degrees = np.array([d for n, d in G.degree()])


    sensitivity = 1.0
    noise = np.random.laplace(0, sensitivity / (2 * eps), size=G.number_of_nodes())
    noisy_degrees = np.clip(original_degrees + noise, 0, None).astype(np.int64)


    noisy_degrees = np.where(noisy_degrees < 1, 1, noisy_degrees)


    if np.sum(noisy_degrees) % 2 != 0:
        idx = np.random.randint(0, len(noisy_degrees))
        noisy_degrees[idx] += np.random.choice([-1, 1])
        noisy_degrees = np.clip(noisy_degrees, 1, None)


    isolated_nodes = np.where(noisy_degrees == 1)[0]
    for node in isolated_nodes:
        neighbors = list(G.neighbors(node))
        if neighbors:
            target = np.random.choice(neighbors)
            noisy_degrees[node] += 1
            noisy_degrees[target] += 1
        else:

            pass


    assert len(noisy_degrees) == G.number_of_nodes(), "Number of noisy degrees does not match number of nodes"
    assert np.isclose(np.sum(noisy_degrees), int(np.sum(noisy_degrees))), "errors in noisy degrees"
    assert (noisy_degrees >= 1).all(), "errors in noisy degrees"
    return noisy_degrees

def matrix_to_graph(matrix):
    n = matrix.shape[0]
    G = nx.Graph()


    G.add_nodes_from(range(n))


    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] == 1:
                G.add_edge(i, j)


    assert G.number_of_nodes() == n, f"node do not fit: {G.number_of_nodes()} vs {n}"
    return G


def DPRR(A, epsilon):

    n = A.shape[0]

    alpha = 0.9
    epsilon1 = max(np.sqrt(8 / (n - 1)), (1 - alpha) * epsilon)
    epsilon1 = min(epsilon1, epsilon)
    epsilon2 = epsilon - epsilon1


    B = np.zeros_like(A, dtype=int)


    for i in range(n):

        d_i = np.sum(A[i, :])


        d_i_star = d_i + np.random.laplace(0, 1 / epsilon1)


        p = np.exp(epsilon2) / (np.exp(epsilon2) + 1)


        denominator = d_i_star * (2 * p - 1) + (n - 1) * (1 - p)
        if denominator == 0:
            q_i = 0
        else:
            q_i = d_i_star / denominator
        q_i = np.clip(q_i, 0, 1)


        for j in range(n):
            if i == j:
                continue
            a_ij = A[i, j]

            if np.random.rand() < p:
                b_ij = a_ij
            else:
                b_ij = 1 - a_ij
            if b_ij == 1:
                if np.random.rand() < q_i:
                    B[i, j] = 1
                else:
                    B[i, j] = 0
            else:
                B[i, j] = 0

    return B


def LAPGRAPH(A, epsilon):

    n = A.shape[0]
    # ε1 = 0.01ε, ε2 = ε - ε1
    epsilon1 = 0.01 * epsilon
    epsilon2 = epsilon - epsilon1


    upper_tri_indices = np.triu_indices(n, k=1)
    T_original = np.sum(A[upper_tri_indices])


    T_perturbed = T_original + np.random.laplace(0, 1 / epsilon1)
    T_perturbed = int(np.round(T_perturbed))
    T_perturbed = max(0, T_perturbed)


    upper_tri_values = A[upper_tri_indices]


    noise = np.random.laplace(0, 1 / epsilon2, size=upper_tri_values.shape)
    noisy_upper = upper_tri_values + noise


    new_upper = np.zeros_like(noisy_upper)
    if T_perturbed > 0:

        sorted_indices = np.argsort(noisy_upper)[::-1]
        top_T_indices = sorted_indices[:T_perturbed]
        new_upper[top_T_indices] = 1


    B = np.zeros_like(A)
    B[upper_tri_indices] = new_upper

    lower_tri_indices = np.tril_indices(n, k=-1)
    B[lower_tri_indices] = B.T[lower_tri_indices]

    return B


def LDPGen(A, epsilon, k=2):

    n = A.shape[0]

    group_assignment = np.random.randint(0, k, n)
    groups = []
    for i in range(k):
        groups.append(np.where(group_assignment == i)[0])


    deg_vec = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            deg_vec[i, j] = np.sum(A[i, groups[j]])


    noisy_vec = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            noise = np.random.laplace(0, 1 / epsilon)
            noisy_vec[i, j] = deg_vec[i, j] + noise
            noisy_vec[i, j] = max(0, round(noisy_vec[i, j]))


    B = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(k):
            group_j = groups[j]

            available_nodes = [v for v in group_j if v != i]
            num_to_select = min(int(noisy_vec[i, j]), len(available_nodes))
            if num_to_select > 0:
                selected = np.random.choice(available_nodes, size=num_to_select, replace=False)
                for v in selected:
                    B[i, v] = 1
                    B[v, i] = 1
    return B

def keep_global_top_partition(matrix, k):

    partition_indices = np.argpartition(matrix, -k, axis=None)[-k:]


    rows, cols = np.unravel_index(partition_indices, matrix.shape)

    result = np.zeros_like(matrix)
    result[rows, cols] = matrix[rows, cols]

    return result
