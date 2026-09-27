"""Phase 6 model registry: phase-5 gated.py extended with GCN / SAGE / GATv2.

This is the verbatim phase-5 ``gated.py`` (decompiled via PyLingual) with the
external-baseline registry patches applied (GCNConv import, the GCNConv no-aggr
branch in ``_DirectedLayer``, and the ``gcn_dir`` / ``sage_dir`` / ``gatv2_dir``
registrations in ``_make_encoder`` and ``build_model``).  All model definitions,
descriptors, gates and decoders are identical to the paper's Phase-5 model.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GATv2Conv, SAGEConv, GCNConv


def compute_descriptors(
    features: torch.Tensor,
    fit_mp_edges: torch.Tensor,
    core_mask: torch.Tensor,
    dates: np.ndarray,
    cutoff: str,
    dt_days: int = 730,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Per-node structural + temporal descriptors from the fit MP graph.

    ``dates`` is the node-date array (``graph["dates"]``, ISO strings) and
    ``cutoff`` the fit-window end date (ISO string); the recent window is the
    last ``dt_days`` days of the fit window.  Returns a standardised ``[N, 13]``
    tensor and the z-score statistics (fit-core only).
    """
    from scipy import sparse
    n = features.shape[0]
    src = fit_mp_edges[0].numpy()
    tgt = fit_mp_edges[1].numpy()
    ones = np.ones(len(src), dtype=np.float32)
    a = sparse.csr_matrix((ones, (src, tgt)), shape=(n, n))
    at = a.T.tocsr()
    d_in = np.asarray(a.sum(0)).ravel()
    d_out = np.asarray(a.sum(1)).ravel()
    safe_in = np.maximum(d_in, 1.0)
    safe_out = np.maximum(d_out, 1.0)
    mean_in_nn_deg = np.asarray(at @ d_out).ravel() / safe_in
    mean_out_nn_deg = np.asarray(a @ d_in).ravel() / safe_out
    x = features.numpy()
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-08)
    sum_in = np.asarray(at @ xn)
    sum_out = np.asarray(a @ xn)
    cos_in = np.where(safe_in > 0, (sum_in * xn).sum(axis=1) / safe_in, 0.0)
    cos_out = np.where(safe_out > 0, (sum_out * xn).sum(axis=1) / safe_out, 0.0)
    mutual = np.asarray(a.multiply(at).sum(1)).ravel()
    recip_out = mutual / safe_out
    has_edge = (d_in + d_out > 0).astype(np.float32)
    day = np.datetime64
    node_days = np.array([(day(d) - day('1970-01-01')) // np.timedelta64(1, 'D') for d in dates], dtype=np.float64)
    cutoff_days = float((day(cutoff) - day('1970-01-01')) // np.timedelta64(1, 'D'))
    edge_day = node_days[src]
    recent = edge_day >= cutoff_days - dt_days
    recent_src = src[recent]
    recent_tgt = tgt[recent]
    recent_in = np.bincount(recent_tgt, minlength=n).astype(np.float64)
    recent_out = np.bincount(recent_src, minlength=n).astype(np.float64)
    recent_in_share = np.where(d_in > 0, recent_in / np.maximum(d_in, 1.0), 0.0)
    age_pairs = np.maximum(node_days[src] - node_days[tgt], 0.0)
    age_log = np.log1p(age_pairs)
    age_sum = np.bincount(src, weights=age_log, minlength=n)
    cit_age = np.where(d_out > 0, age_sum / safe_out, 0.0)
    node_recency = np.log1p(np.maximum(cutoff_days - node_days, 0.0))
    raw = np.stack([np.log1p(d_in), np.log1p(d_out), np.log1p(mean_in_nn_deg), np.log1p(mean_out_nn_deg), cos_in, cos_out, recip_out, has_edge, np.log1p(recent_in), np.log1p(recent_out), recent_in_share, cit_age, node_recency], axis=1).astype(np.float32)
    core = core_mask.numpy().astype(bool)
    mean = raw[core].mean(axis=0, keepdims=True)
    std = raw[core].std(axis=0, keepdims=True)
    std = np.where(std < 1e-06, 1.0, std)
    normed = (raw - mean) / std
    meta = {'dim': 13, 'source': 'fit_mp_only_plus_node_dates', 'core_count': int(core.sum()), 'zero_row_count': int((has_edge == 0).sum()), 'dt_days': int(dt_days), 'cutoff': cutoff, 'dropped_dims': ['reciprocity recency (mutual edges are ~0.06% of fit edges on HepTh/HepPh; the dim is constant zero and carries no signal)'], 'train_stats': {'mean': mean.tolist(), 'std': std.tolist()}}
    return (torch.tensor(normed, dtype=torch.float32), meta)


def _orthogonal_layer(in_dim: int, out_dim: int) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim, bias=False)
    nn.init.orthogonal_(layer.weight)
    return layer


def _zero_init(layer: nn.Module) -> None:
    if hasattr(layer, 'weight') and layer.weight is not None:
        nn.init.zeros_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int | None = None) -> torch.Tensor:
    """Mean scatter with deterministic fallback (no torch_scatter dependency)."""
    if dim_size is None:
        dim_size = int(index.max()) + 1
    out = torch.zeros(dim_size, src.shape[1], dtype=src.dtype, device=src.device)
    counts = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    counts.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    counts = counts.clamp_min(1.0)
    return out / counts.unsqueeze((-1))


class _Gate(nn.Module):
    """g = 1 + tanh(MLP(descriptor)); zero-init => g == 1 at initialisation."""

    def __init__(self, desc_dim: int):
        super().__init__()
        self.mlp = nn.Linear(desc_dim, 1)
        _zero_init(self.mlp)

    def forward(self, desc: torch.Tensor) -> torch.Tensor:
        return 1.0 + torch.tanh(self.mlp(desc))


class _DirectedLayer(nn.Module):
    """Two-view directed layer, optionally gate-modulated per direction.

    Phase 5 adds an optional *time branch*: a zero-initialised linear map from
    per-edge time features is mean-scattered onto each aggregation target.
    With ``time_proj`` at zero the layer is numerically identical to the
    Phase 2/4 layer, so the epoch-0 anchor (content floor) is preserved; the
    time branch is the only difference once trained.
    """

    def __init__(
        self,
        conv_cls,
        in_dim: int,
        out_dim: int,
        heads: int,
        dropout: float,
        desc_dim: int,
        gated: bool,
        use_time: bool = False,
        time_dim: int = 1,
    ):
        super().__init__()
        common = {'bias': True}
        if conv_cls in (GATConv, GATv2Conv):
            common.update(heads=heads, concat=False, dropout=dropout, add_self_loops=True)
        elif conv_cls is SAGEConv:
            common.update(aggr='mean')
        # GCNConv takes no ``aggr`` argument (mean over normalized adjacency).
        self.conv_in = conv_cls(in_dim, out_dim, **common)
        self.conv_out = conv_cls(in_dim, out_dim, **common)
        self.combine = nn.Linear(2 * out_dim, out_dim)
        self.gated = gated
        if gated:
            self.gate_in = _Gate(desc_dim)
            self.gate_out = _Gate(desc_dim)
        self.dropout = dropout
        self.use_time = use_time
        if use_time:
            self.time_proj = nn.Linear(time_dim, out_dim, bias=False)
            nn.init.zeros_(self.time_proj.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, desc: torch.Tensor, edge_time: torch.Tensor | None = None) -> torch.Tensor:
        reversed_edges = edge_index.flip(0)
        h_in = self.conv_in(x, edge_index)
        h_out = self.conv_out(x, reversed_edges)
        if self.use_time and edge_time is not None:
            n = x.shape[0]
            t_in = _scatter_mean(self.time_proj(edge_time), edge_index[1], dim=0, dim_size=n)
            t_out = _scatter_mean(self.time_proj(edge_time), edge_index[0], dim=0, dim_size=n)
            h_in = h_in + t_in
            h_out = h_out + t_out
        if self.gated:
            h_in = h_in * self.gate_in(desc)
            h_out = h_out * self.gate_out(desc)
        h = self.combine(torch.cat([h_in, h_out], dim=(-1)))
        return F.dropout(h, p=self.dropout, training=self.training)


class DirectedGateEncoder(nn.Module):
    """Two-layer direction-separated GNN with optional RA-GAT gates.

    Content-residual low-rank correction identical in structure to Phase 2:
    ``z = x + U * branch(Dx)``; ``U`` zero-initialised, gates zero-initialised,
    so the untrained model equals the ungated Phase 2 model and the content
    floor.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        dropout: float,
        desc_dim: int,
        heads: int = 4,
        variant: str = 'gat',
        gated: bool = True,
        use_time: bool = False,
        **_unused: Any,
    ):
        super().__init__()
        if variant == 'gat':
            conv_cls = GATConv
        elif variant == 'gatv2':
            conv_cls = GATv2Conv
        elif variant == 'sage':
            conv_cls = SAGEConv
        elif variant == 'gcn':
            conv_cls = GCNConv
        else:
            raise ValueError(f'unknown variant {variant!r}')
        self.in_dim = in_dim
        self.use_time = use_time
        self.down = _orthogonal_layer(in_dim, hidden)
        self.layer1 = _DirectedLayer(conv_cls, hidden, hidden, heads, dropout, desc_dim, gated, use_time=self.use_time)
        self.norm = nn.LayerNorm(hidden)
        self.layer2 = _DirectedLayer(conv_cls, hidden, hidden, heads, dropout, desc_dim, gated, use_time=self.use_time)
        self.up = nn.Linear(hidden, in_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, desc: torch.Tensor, edge_time: torch.Tensor | None = None) -> torch.Tensor:
        h = self.down(x)
        h = F.elu(self.layer1(h, edge_index, desc, edge_time))
        h = self.norm(h)
        correction = self.up(self.layer2(h, edge_index, desc, edge_time))
        return x + correction


class SymRanker(nn.Module):
    """Symmetric cosine scorer with learned log-temperature (Phase 2 parity)."""

    def __init__(self, encoder: nn.Module, init_scale: float = 10.0):
        super().__init__()
        self.encoder = encoder
        self.log_scale = nn.Parameter(torch.tensor(float(init_scale)).log())

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, desc: torch.Tensor, edge_time: torch.Tensor | None = None) -> torch.Tensor:
        return F.normalize(self.encoder(x, edge_index, desc, edge_time), dim=(-1), eps=1e-08)

    def scale(self) -> torch.Tensor:
        return self.log_scale.exp()

    def score(self, z: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        left = z.index_select(0, pairs[0])
        right = z.index_select(0, pairs[1])
        return self.scale() * (left * right).sum(dim=(-1))


class AsymRanker(nn.Module):
    """Asymmetric scorer: cosine anchor plus learned asymmetric MLP residual."""

    def __init__(self, encoder: nn.Module, hidden_dec: int = 64, init_scale: float = 10.0):
        super().__init__()
        self.encoder = encoder
        self.log_scale = nn.Parameter(torch.tensor(float(init_scale)).log())
        self.decoder = nn.Sequential(nn.Linear(4 * encoder.in_dim, hidden_dec), nn.ELU(), nn.Linear(hidden_dec, 1))
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, desc: torch.Tensor, edge_time: torch.Tensor | None = None) -> torch.Tensor:
        return F.normalize(self.encoder(x, edge_index, desc, edge_time), dim=(-1), eps=1e-08)

    def scale(self) -> torch.Tensor:
        return self.log_scale.exp()

    def score(self, z: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        u = z.index_select(0, pairs[0])
        v = z.index_select(0, pairs[1])
        anchor = self.scale() * (u * v).sum(dim=(-1))
        feat = torch.cat([u, v, u * v, u - v], dim=(-1))
        return anchor + self.decoder(feat).squeeze((-1))


def _make_encoder(model_name: str, in_dim: int, cfg: dict[str, Any]) -> nn.Module:
    hidden = int(cfg['hidden'])
    dropout = float(cfg['dropout'])
    heads = int(cfg.get('heads', 4))
    if model_name in ['gat_dir', 'gat_asym']:
        gated, variant, use_time = (False, 'gat', False)
    elif model_name in ['gcn_dir']:
        gated, variant, use_time = (False, 'gcn', False)
    elif model_name in ['sage_dir']:
        gated, variant, use_time = (False, 'sage', False)
    elif model_name in ['gatv2_dir']:
        gated, variant, use_time = (False, 'gatv2', False)
    elif model_name in ['ragat_sym', 'ragat_asym', 'ragatv2_sym']:
        gated, variant, use_time = (True, 'gat', False)
    elif model_name == 'gat_time':
        gated, variant, use_time = (False, 'gat', True)
    elif model_name == 'ragat_time':
        gated, variant, use_time = (True, 'gat', True)
    else:
        raise ValueError(f'unknown model {model_name!r}')
    return DirectedGateEncoder(in_dim, hidden, dropout, desc_dim=cfg['desc_dim'], heads=heads, variant=variant, gated=gated, use_time=use_time)


def build_model(model_name: str, in_dim: int, cfg: dict[str, Any], desc_dim: int = 13) -> nn.Module:
    model_name = canonical(model_name)
    cfg = dict(cfg)
    cfg['desc_dim'] = desc_dim
    encoder = _make_encoder(model_name, in_dim, cfg)
    if model_name in ['gat_dir', 'ragat_sym', 'gat_time', 'ragat_time', 'ragatv2_sym', 'gcn_dir', 'sage_dir', 'gatv2_dir']:
        return SymRanker(encoder)
    if model_name in ['gat_asym', 'ragat_asym']:
        return AsymRanker(encoder)
    raise ValueError(f'unknown model {model_name!r}')


# Code name -> paper name.  The recovered registry uses ``ragat_*`` (from the
# decompiled module), the manuscript calls the same cells RC-GAT
# (``rcgat_sym`` / ``rcgat_time``).  Both spellings resolve to one model.
MODEL_ALIASES = {
    'ragat_sym': 'rcgat_sym',
    'ragat_time': 'rcgat_time',
    'ragat_asym': 'rcgat_asym',
    'ragatv2_sym': 'rcgatv2_sym',
}


def canonical(name: str) -> str:
    """Translate a paper name (rcgat_*) into the registry name (ragat_*)."""
    reverse = {v: k for k, v in MODEL_ALIASES.items()}
    return reverse.get(name, name)


def model_names() -> list[str]:
    return ['gat_dir', 'gcn_dir', 'sage_dir', 'gatv2_dir', 'ragat_sym', 'gat_time', 'ragat_time']
