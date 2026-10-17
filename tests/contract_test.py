"""Small-input checks; not a rerun of manuscript experiments."""
import hashlib
import numpy as np
import torch
from unittest.mock import patch
from spaarma.alignment import build_candidates, keyed_negatives, PositiveAttention, calibrate_margin
from spaarma.graph import spatial_edges
from spaarma.training import fit, SpaRMAConfig

batches = np.repeat(["a", "b", "c"], 3)
names = np.array([f"spot{i}" for i in range(9)])
x = np.arange(18, dtype=np.float32).reshape(9, 2)
a, p, valid, pairs = build_candidates(x, batches, 2, 3, spot_names=names, backend="exact", return_pairs=True)
assert len(a) == 18  # every spot contributes in two different batch pairs
assert np.all(valid.sum(1) == 2)
assert len(set(zip(pairs, a))) == 18
n = keyed_negatives(a, batches, names, pairs, 42, 500)
for i, anchor in enumerate(a):
    pool = np.flatnonzero(batches == batches[anchor])
    key = f"42|500|{pairs[i]}|{names[anchor]}"
    draw = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little") % len(pool)
    assert n[i] == pool[draw]
q = torch.tensor([[1e-5, 0.0]], requires_grad=True)
v = torch.tensor([[[1., 0.], [0., 1.]]])
w = PositiveAttention(2)(q, v, torch.ones((1,2), dtype=torch.bool))
assert torch.allclose(w, torch.softmax(torch.tensor([[0.01, 0.]]), 1))
w[0,0].backward()
assert torch.isfinite(q.grad).all()
gaps = torch.tensor([0., 1., 2.])
assert np.isfinite(calibrate_margin(gaps, torch.ones(3,dtype=torch.bool), 1/3))
coords = np.column_stack([np.arange(65), np.zeros(65)])
edges = spatial_edges(coords, np.repeat("a",65), 1000).numpy()
assert len(edges[0]) == 65 * 51  # 50 neighbors plus self, not all radius neighbors
boundary = spatial_edges(np.array([[0,0],[1,0]]), ["a","a"], 1).numpy()
assert boundary.shape[1] == 2  # strict radius
loops = torch.arange(9)
clipped = []
original_clip = torch.nn.utils.clip_grad_norm_
def track_clip(parameters, limit):
    parameters = list(parameters)
    clipped.append((len(parameters), limit))
    return original_clip(parameters, limit)
with patch('torch.nn.utils.clip_grad_norm_', track_clip):
    emb, settings, _ = fit(x, torch.stack([loops,loops]), batches,
        SpaRMAConfig(hidden_dim=8,latent_dim=3,pretrain_epochs=2,joint_epochs=2,
                     mnn_k=3,positive_k=2,mnn_backend="exact"),
        device="cpu", spot_names=names)
assert clipped == [(4,5.),(1,5.),(4,5.),(1,5.),(6,5.),(6,5.)]
assert emb.shape == (9,3) and np.isfinite(emb).all()
print("contract_tests=PASS")
