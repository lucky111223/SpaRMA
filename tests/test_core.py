import numpy as np
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from spaarma.alignment import PositiveAttention, calibrate_margin, weighted_triplet
from spaarma.model import SpatialGraphAutoencoder
from spaarma.training import _sample_masked_spots


def test_shapes_and_masked_decode():
    model = SpatialGraphAutoencoder(4, 8, 3, 0.0)
    x = torch.randn(5, 4)
    edge = torch.tensor([[0,1,2,3,4,0,1,2,3,4],[0,1,2,3,4,1,2,3,4,0]])
    z, out = model(x, edge, zero_latent_rows=torch.tensor([1,3]))
    assert z.shape == (5,3) and out.shape == x.shape


def test_attention_and_loss_are_finite():
    a, p, n = torch.randn(4,3), torch.randn(4,2,3), torch.randn(4,3)
    valid = torch.tensor([[1,1],[1,0],[1,1],[1,1]], dtype=torch.bool)
    weights = PositiveAttention(3)(a,p,valid)
    loss, gaps = weighted_triplet(a,p,n,valid,weights,1.0)
    assert torch.isfinite(loss) and torch.allclose(weights.sum(1), torch.ones(4))
    assert np.isfinite(calibrate_margin(gaps, valid, 0.5))


def test_masking_is_stratified_by_slice():
    batches = np.array(["s1"] * 10 + ["s2"] * 20)
    selected = _sample_masked_spots(batches, 0.2, 7, 3, "cpu").numpy()
    assert np.sum(selected < 10) == 2
    assert np.sum(selected >= 10) == 4
