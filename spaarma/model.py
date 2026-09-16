import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import softmax


class SpatialAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.att_src = nn.Parameter(torch.empty(out_dim))
        self.att_dst = nn.Parameter(torch.empty(out_dim))
        self.dropout = float(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight, gain=1.414)
        nn.init.xavier_normal_(self.att_src.view(1, -1), gain=1.414)
        nn.init.xavier_normal_(self.att_dst.view(1, -1), gain=1.414)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight

    def attend(self, transformed: torch.Tensor, edge_index: torch.Tensor, node_scores=None):
        src, dst = edge_index
        if node_scores is None:
            score_src = (transformed * self.att_src).sum(1)
            score_dst = (transformed * self.att_dst).sum(1)
            node_scores = (score_src, score_dst)
        logits = torch.sigmoid(node_scores[0][src] + node_scores[1][dst])
        alpha = softmax(logits, dst, num_nodes=transformed.shape[0])
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.zeros_like(transformed)
        out.index_add_(0, dst, transformed[src] * alpha[:, None])
        return out, node_scores


class SpatialGraphAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, latent_dim: int = 30, dropout: float = 0.05):
        super().__init__()
        self.encoder_attention = SpatialAttentionLayer(input_dim, hidden_dim, dropout)
        self.encoder_projection = nn.Parameter(torch.empty(hidden_dim, latent_dim))
        nn.init.xavier_normal_(self.encoder_projection, gain=1.414)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor):
        h, scores = self.encoder_attention.attend(self.encoder_attention.transform(x), edge_index)
        return F.elu(h) @ self.encoder_projection, scores

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor, encoder_scores):
        h = z @ self.encoder_projection.T
        h, _ = self.encoder_attention.attend(h, edge_index, node_scores=encoder_scores)
        return F.elu(h) @ self.encoder_attention.weight.T

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, zero_latent_rows=None):
        z, scores = self.encode(x, edge_index)
        decode_z = z
        if zero_latent_rows is not None:
            decode_z = z.clone()
            decode_z[zero_latent_rows] = 0
        return z, self.decode(decode_z, edge_index, scores)
