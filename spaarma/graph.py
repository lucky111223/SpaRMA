import numpy as np
import torch
from scipy.spatial import cKDTree


def spatial_edges(coordinates, batches, radius):
    coordinates, batches = np.asarray(coordinates), np.asarray(batches)
    edges = []
    for batch in dict.fromkeys(batches.tolist()):
        idx = np.flatnonzero(batches == batch)
        for i, j in cKDTree(coordinates[idx]).query_pairs(radius, output_type="ndarray"):
            edges.extend([(idx[i], idx[j]), (idx[j], idx[i])])
        edges.extend((int(i), int(i)) for i in idx)
    return torch.as_tensor(np.asarray(edges).T, dtype=torch.long)
