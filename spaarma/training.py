from dataclasses import asdict, dataclass
import random
import numpy as np
import torch
from torch.nn import functional as F
from .alignment import PositiveAttention, build_candidates, calibrate_margin, sample_negatives, weighted_triplet
from .model import SpatialGraphAutoencoder


@dataclass
class SpaRMAConfig:
    hidden_dim: int = 512
    latent_dim: int = 30
    pretrain_epochs: int = 500
    joint_epochs: int = 500
    mask_rate: float = 0.10
    dropout: float = 0.05
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    refresh_interval: int = 100
    positive_k: int = 3
    mnn_k: int = 50
    base_margin: float = 1.0
    seed: int = 0


def _seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def fit(x, edge_index, batches, config: SpaRMAConfig, device=None):
    _seed(config.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    edge_index = edge_index.to(device)
    model = SpatialGraphAutoencoder(x.shape[1], config.hidden_dim, config.latent_dim, config.dropout).to(device)
    mask_token = torch.nn.Parameter(torch.zeros(1, x.shape[1], device=device))
    optimizer = torch.optim.Adam(list(model.parameters()) + [mask_token], lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    count = max(1, round(config.mask_rate * len(x)))
    model.train()
    for _ in range(config.pretrain_epochs):
        selected = torch.randperm(len(x), generator=generator, device=device)[:count]
        masked = x.clone(); masked[selected] = mask_token
        _, reconstructed = model(masked, edge_index, zero_latent_rows=selected)
        loss = F.mse_loss(reconstructed[selected], x[selected])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    attention = PositiveAttention(config.latent_dim).to(device)
    optimizer.add_param_group({"params": attention.parameters(), "lr": config.learning_rate, "weight_decay": config.weight_decay})
    rng = np.random.default_rng(config.seed)
    current_margin = config.base_margin
    for epoch in range(config.joint_epochs):
        z, reconstructed = model(x, edge_index)
        if epoch % config.refresh_interval == 0:
            z_np = z.detach().cpu().numpy()
            anchors, positives, valid_np = build_candidates(z_np, batches, config.positive_k, config.mnn_k)
            negatives = sample_negatives(anchors, batches, rng)
            anchor_i = torch.as_tensor(anchors, device=device)
            positive_i = torch.as_tensor(positives, device=device)
            negative_i = torch.as_tensor(negatives, device=device)
            valid = torch.as_tensor(valid_np, device=device)
            with torch.no_grad():
                a, p, n = z[anchor_i], z[positive_i], z[negative_i]
                reference_gap = torch.linalg.vector_norm(a - n, dim=1) - torch.linalg.vector_norm(a - p[:, 0], dim=1)
                target = float((reference_gap < config.base_margin).float().mean())
                candidate_gap = torch.linalg.vector_norm(a - n, dim=1)[:, None] - torch.linalg.vector_norm(a[:, None] - p, dim=2)
                row_gap = candidate_gap.masked_fill(~valid, torch.inf).min(dim=1).values
                current_margin = calibrate_margin(row_gap, torch.ones_like(row_gap, dtype=torch.bool), target, config.base_margin)
        a, p, n = z[anchor_i], z[positive_i], z[negative_i]
        weights = attention(a, p, valid)
        alignment, _ = weighted_triplet(a, p, n, valid, weights, current_margin)
        loss = F.mse_loss(reconstructed, x) + alignment
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
    model.eval()
    with torch.no_grad():
        embedding, _ = model.encode(x, edge_index)
    return embedding.cpu().numpy(), {**asdict(config), "final_margin": current_margin}, model
