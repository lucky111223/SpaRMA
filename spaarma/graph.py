import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def spatial_edges(coordinates, batches, radius):
    coordinates, batches = np.asarray(coordinates), np.asarray(batches)
    edges = []
    for batch in dict.fromkeys(batches.tolist()):
        idx = np.flatnonzero(batches == batch)
        distances, neighbors = NearestNeighbors(n_neighbors=min(51, len(idx)), algorithm="ball_tree").fit(coordinates[idx]).kneighbors(coordinates[idx])
        for i in range(len(idx)):
            for distance, j in zip(distances[i, 1:], neighbors[i, 1:]):
                if distance < radius:
                    edges.append((int(idx[i]), int(idx[j])))
        edges.extend((int(i), int(i)) for i in idx)
    return torch.as_tensor(np.asarray(edges).T, dtype=torch.long)
