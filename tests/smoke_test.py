import numpy as np
import torch
from spaarma.alignment import PositiveAttention, calibrate_margin, weighted_triplet
from spaarma.model import SpatialGraphAutoencoder
from spaarma.training import SpaRMAConfig, _sample_masked_spots, fit

model = SpatialGraphAutoencoder(4, 8, 3, 0.0)
x = torch.randn(5, 4)
edge = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 1, 2, 3, 4, 0]])
z, out = model(x, edge, zero_latent_rows=torch.tensor([1, 3]))
assert z.shape == (5, 3) and out.shape == x.shape
a, p, n = torch.randn(4, 3), torch.randn(4, 2, 3), torch.randn(4, 3)
valid = torch.tensor([[1, 1], [1, 0], [1, 1], [1, 1]], dtype=torch.bool)
weights = PositiveAttention(3)(a, p, valid)
loss, gaps = weighted_triplet(a, p, n, valid, weights, 1.0)
assert torch.isfinite(loss)
assert torch.allclose(weights.sum(1), torch.ones(4))
assert np.isfinite(calibrate_margin(gaps, valid, 0.5))
base = np.array([[0, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0]], dtype=np.float32)
features = np.vstack([base, base + 0.01])
loops = torch.arange(6)
embedding, settings, _ = fit(
    features,
    torch.stack([loops, loops]),
    np.array(["slice1"] * 3 + ["slice2"] * 3),
    SpaRMAConfig(hidden_dim=8, latent_dim=3, pretrain_epochs=1, joint_epochs=1,
                 refresh_interval=1, positive_k=2, mnn_k=2, seed=1),
    device="cpu",
)
assert embedding.shape == (6, 3) and np.isfinite(embedding).all()
assert np.isfinite(settings["final_margin"])
selected = _sample_masked_spots(
    np.array(["s1"] * 10 + ["s2"] * 20), 0.2, 7, 3, "cpu"
).numpy()
assert np.sum(selected < 10) == 2 and np.sum(selected >= 10) == 4
print("core_tests=PASS")
