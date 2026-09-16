from collections import defaultdict
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
        q = F.normalize(self.W_Q(anchors), dim=-1)
        k = F.normalize(self.W_K(positives), dim=-1)
        scores = (q[:, None, :] * k).sum(-1).masked_fill(~valid, -torch.inf)
        return torch.softmax(scores, dim=1)


def _mutual_pairs(a, b, k):
    ka, kb = min(k, len(b)), min(k, len(a))
    ab = NearestNeighbors(n_neighbors=ka).fit(b).kneighbors(a, return_distance=False)
    ba = NearestNeighbors(n_neighbors=kb).fit(a).kneighbors(b, return_distance=False)
    reverse = [set(row.tolist()) for row in ba]
    return [(i, j) for i, row in enumerate(ab) for j in row if i in reverse[j]]


def build_candidates(embedding, batches, positive_k, mnn_k, batch_pairs=None):
    batches = np.asarray(batches)
    levels = list(dict.fromkeys(batches.tolist()))
    pairs = batch_pairs or [(levels[i], levels[j]) for i in range(len(levels)) for j in range(i + 1, len(levels))]
    groups = defaultdict(list)
    for left, right in pairs:
        li, ri = np.flatnonzero(batches == left), np.flatnonzero(batches == right)
        for a, p in _mutual_pairs(embedding[li], embedding[ri], mnn_k):
            groups[int(li[a])].append(int(ri[p])); groups[int(ri[p])].append(int(li[a]))
    anchors, rows = [], []
    for anchor, candidates in groups.items():
        unique = np.array(list(dict.fromkeys(candidates)), dtype=int)
        dist = np.linalg.norm(embedding[unique] - embedding[anchor], axis=1)
        anchors.append(anchor); rows.append(unique[np.argsort(dist)[:positive_k]])
    if not anchors:
        raise RuntimeError("No cross-slice MNN correspondences were found")
    width = max(len(row) for row in rows)
    positive = np.zeros((len(rows), width), dtype=np.int64)
    valid = np.zeros((len(rows), width), dtype=bool)
    for i, row in enumerate(rows):
        positive[i, :len(row)] = row; valid[i, :len(row)] = True
    return np.asarray(anchors, dtype=np.int64), positive, valid


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
