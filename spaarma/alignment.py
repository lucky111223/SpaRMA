import hashlib
import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.neighbors import NearestNeighbors


class PositiveAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W_Q = nn.Linear(dim, dim, bias=False)
        self.W_K = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.W_Q.weight.copy_(torch.eye(dim))
            self.W_K.weight.copy_(torch.eye(dim))

    def forward(self, anchors, positives, valid):
        q = F.normalize(self.W_Q(anchors), dim=-1, eps=1e-3)
        k = F.normalize(self.W_K(positives), dim=-1, eps=1e-3)
        scores = (q[:, None, :] * k).sum(-1).masked_fill(~valid, -torch.inf)
        return torch.softmax(scores, dim=1)


def _mutual_pairs(a, b, k):
    ka, kb = min(k, len(b)), min(k, len(a))
    ab = NearestNeighbors(n_neighbors=ka).fit(b).kneighbors(a, return_distance=False)
    ba = NearestNeighbors(n_neighbors=kb).fit(a).kneighbors(b, return_distance=False)
    reverse = [set(row.tolist()) for row in ba]
    return [(i, j) for i, row in enumerate(ab) for j in row if i in reverse[j]]


def build_candidates(embedding, batches, positive_k, mnn_k, batch_pairs=None,
                     spot_names=None, backend="hnsw", return_pairs=False):
    batches = np.asarray(batches)
    if positive_k < 1 or mnn_k < 1:
        raise ValueError("positive_k and mnn_k must be positive")
    levels = list(dict.fromkeys(batches.tolist()))
    pairs = batch_pairs or [(levels[i], levels[j]) for i in range(len(levels)) for j in range(i + 1, len(levels))]
    names = np.asarray(spot_names if spot_names is not None else [str(i) for i in range(len(batches))])
    if len(set(names.tolist())) != len(names):
        raise ValueError("Spot names must be unique")
    lookup = {name: i for i, name in enumerate(names)}
    anchors, rows, pair_ids = [], [], []
    for left, right in pairs:
        li, ri = np.flatnonzero(batches == left), np.flatnonzero(batches == right)
        if not len(li) or not len(ri):
            raise ValueError("A prescribed correspondence batch is empty")
        def directed(source, target):
            k = min(mnn_k, len(target))
            if backend == "hnsw":
                import hnswlib
                index = hnswlib.Index(space="l2", dim=embedding.shape[1])
                index.init_index(max_elements=len(target), ef_construction=100, M=16)
                index.set_ef(10)
                index.add_items(embedding[target])
                neighbors, _ = index.knn_query(embedding[source], k=k)
            elif backend == "exact":
                neighbors = NearestNeighbors(n_neighbors=k, p=2).fit(embedding[target]).kneighbors(embedding[source], return_distance=False)
            else:
                raise ValueError("backend must be hnsw or exact")
            return {(names[source[i]], names[target[j]]) for i, row in enumerate(neighbors) for j in row}
        # Preserve the executed STAligner set -> NetworkX sparse-adjacency path.
        # Do not replace this with distance sorting or merge different batch pairs.
        forward, reverse = directed(ri, li), directed(li, ri)
        mutual = forward & {(b, a) for a, b in reverse}
        graph = nx.Graph()
        graph.add_edges_from(mutual)
        if not graph:
            continue
        nodes = np.asarray(list(graph.nodes))
        adjacency = nx.adjacency_matrix(graph).tocsr()
        for i, anchor in enumerate(nodes):
            neighbors = nodes[adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]]]
            anchors.append(lookup[anchor])
            rows.append(np.asarray([lookup[n] for n in neighbors[:positive_k]], dtype=int))
            pair_ids.append(f"{left}_{right}")
    if not anchors:
        raise RuntimeError("No cross-slice MNN correspondences were found")
    width = max(len(row) for row in rows)
    positive = np.zeros((len(rows), width), dtype=np.int64)
    valid = np.zeros((len(rows), width), dtype=bool)
    for i, row in enumerate(rows):
        positive[i, :len(row)] = row; valid[i, :len(row)] = True
    result = (np.asarray(anchors, dtype=np.int64), positive, valid)
    return (*result, pair_ids) if return_pairs else result


def keyed_negatives(anchors, batches, spot_names, pair_ids, seed, epoch):
    """Executed per-anchor draw; independent of traversal of other anchors."""
    batches, names = np.asarray(batches), np.asarray(spot_names)
    result = []
    for anchor, pair in zip(anchors, pair_ids):
        pool = np.flatnonzero(batches == batches[anchor])
        key = f"{seed}|{epoch}|{pair}|{names[anchor]}"
        draw = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little") % len(pool)
        result.append(pool[draw])
    return np.asarray(result, dtype=np.int64)


def sample_negatives(anchors, batches, rng):
    batches = np.asarray(batches)
    return np.asarray([rng.choice(np.flatnonzero(batches == batches[a])) for a in anchors], dtype=np.int64)


def weighted_triplet(anchor, positive, negative, valid, weights, margin):
    dap = torch.linalg.vector_norm(anchor[:, None, :] - positive, dim=2)
    dan = torch.linalg.vector_norm(anchor - negative, dim=1)
    losses = torch.relu(dap - dan[:, None] + margin).masked_fill(~valid, 0)
    return (weights * losses).sum(1).mean(), dan[:, None] - dap


def calibrate_margin(gaps: torch.Tensor, valid: torch.Tensor, target_rate: float, reference=1.0):
    values = torch.sort(gaps[valid].detach().cpu()).values.numpy()
    candidates = np.unique(np.concatenate([values, np.nextafter(values, np.inf), [reference]]))
    rates = np.searchsorted(values, candidates, side="left") / len(values)
    best = int(np.lexsort((np.abs(candidates - reference), np.abs(rates - target_rate)))[0])
    return float(candidates[best])
