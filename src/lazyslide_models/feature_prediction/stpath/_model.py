"""STPath network, vendored from ``stpath/model/`` and ``stpath/model/nn_utils/``.

Upstream: https://github.com/Graph-and-Geometric-Learning/STPath

Vendored because the upstream distribution declares ``packages=['stpath']`` and
so does not install its own subpackages, is absent from PyPI, and carries no
release tags. Only the inference path is kept — training losses, the masked
pre-training ``forward``, and the unused ``MLP`` backbone are dropped.

Module and attribute names are load-bearing: they are the keys of the released
``stfm.pth`` state dict. Do not rename ``input_encoder``, ``model``,
``gene_exp_head``, ``blks``, ``attn``, ``layernorm_qkv``, ``W_output``,
``edge_bias`` or ``mlp``.
"""

from __future__ import annotations

import torch
from torch import nn

#: Reserved gene-token ids: 0 is padding, 1 is the mask used at inference.
PAD_TOKEN_ID = 0
MASK_TOKEN_ID = 1

#: Shapes of the released checkpoint.
DEFAULT_CONFIG = {
    "feature_dim": 1536,  # GigaPath tile embedding width
    "d_model": 512,
    "n_layers": 4,
    "n_heads": 4,
    "mlp_ratio": 2.0,
    "dropout": 0.1,
    "attn_dropout": 0.1,
}


class FrameAveraging(nn.Module):
    """O(d)-equivariant frame averaging over 2-D coordinate offsets."""

    def __init__(self, dim: int = 2):
        super().__init__()
        self.dim = dim
        self.n_frames = 2**dim
        self.ops = self._create_ops(dim)

    @staticmethod
    def _create_ops(dim: int) -> torch.Tensor:
        # einops lives in the optional ``model`` dependency group, so it is
        # imported where it is used — a module-level import would break
        # ``import lazyslide_models`` for anyone on the base install.
        from einops import rearrange

        directions = torch.tensor([-1, 1])
        accum = []
        for ind in range(dim):
            dim_slice: list = [None] * dim
            dim_slice[ind] = slice(None)
            # tuple() rather than the upstream list: list indexing is deprecated
            # and becomes tensor indexing in PyTorch 2.9.
            accum.append(directions[tuple(dim_slice)])
        operations = torch.stack(torch.broadcast_tensors(*accum), dim=-1)
        return rearrange(operations, "... d -> (...) d")

    def create_frame(self, X: torch.Tensor, mask: torch.Tensor | None = None):
        if mask is None:
            mask = torch.ones(*X.shape[:-1], device=X.device).bool()
        mask = mask.unsqueeze(-1)
        center = (X * mask).sum(dim=1) / mask.sum(dim=1)
        X = X - center.unsqueeze(1) * mask
        X_ = X.masked_fill(~mask, 0.0)

        C = torch.bmm(X_.transpose(1, 2), X_).detach()
        # MPS has no eigh kernel (pytorch#141287). C is only [B, 2, 2], so
        # rounding through the CPU costs a few KB and keeps results identical.
        if C.device.type == "mps":
            eigenvectors = torch.linalg.eigh(C.cpu(), UPLO="U")[1].to(C.device)
        else:
            _, eigenvectors = torch.linalg.eigh(C, UPLO="U")
        F_ops = self.ops.unsqueeze(1).unsqueeze(0).to(
            X.device
        ) * eigenvectors.unsqueeze(1)
        h = torch.einsum("boij,bpj->bopi", F_ops.transpose(2, 3), X)
        return h.view(X.size(0) * self.n_frames, X.size(1), self.dim)


class Attention(FrameAveraging):
    """Self-attention over all spots with a frame-averaged spatial bias.

    Memory is quadratic in the number of spots: the spatial bias materialises
    ``[N * n_frames, N, dim]``. See :class:`STPath` for the practical ceiling.
    """

    def __init__(self, d_model, n_heads=1, proj_drop=0.0, attn_drop=0.0):
        super().__init__(dim=2)
        self.d_head, self.n_heads = d_model // n_heads, n_heads
        self.scale = self.d_head**-0.5

        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3)
        )
        self.W_output = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Dropout(proj_drop)
        )
        self.attn_dropout = nn.Dropout(attn_drop)
        self.edge_bias = nn.Sequential(
            nn.Linear(self.dim + 1, self.n_heads, bias=False)
        )

    def forward(self, x, coords, pad_mask=None):
        from einops import rearrange

        B, N, C = x.shape
        q, k, v = self.layernorm_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            rearrange(t, "b n (h d) -> b h n d", h=self.n_heads) for t in (q, k, v)
        )

        attn = (q * self.scale) @ k.transpose(-2, -1)

        # Pairwise spatial representation, frame-averaged for rotation invariance.
        radial_coords = coords.unsqueeze(dim=2) - coords.unsqueeze(dim=1)
        radial_coord_norm = radial_coords.norm(dim=-1).reshape(B * N, N, 1)
        radial_coords = rearrange(radial_coords, "b n m d -> (b n) m d")
        neighbor_masks = (
            ~rearrange(pad_mask, "b n m -> (b n) m") if pad_mask is not None else None
        )
        frame_feats = self.create_frame(radial_coords, neighbor_masks)
        frame_feats = frame_feats.view(B * N, self.n_frames, N, -1)
        radial_coord_norm = radial_coord_norm.unsqueeze(dim=1).expand(
            B * N, self.n_frames, N, -1
        )

        spatial_bias = self.edge_bias(
            torch.cat([frame_feats, radial_coord_norm], dim=-1)
        ).mean(dim=1)
        attn = attn + rearrange(spatial_bias, "(b n) m h -> b h n m", b=B, n=N)

        if pad_mask is not None:
            attn.masked_fill_(pad_mask.unsqueeze(1), -1e9)
        attn = self.attn_dropout(attn.softmax(dim=-1))

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.W_output(x)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.norm = nn.LayerNorm(hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(self.drop1(self.act(self.fc1(x))))
        return self.drop2(self.fc2(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, attn_drop, proj_drop, mlp_ratio):
        super().__init__()
        self.attn = Attention(
            d_model=d_model, n_heads=n_heads, proj_drop=proj_drop, attn_drop=attn_drop
        )
        self.mlp = MLP(
            in_features=d_model,
            hidden_features=int(d_model * mlp_ratio),
            out_features=d_model,
            drop=proj_drop,
        )

    def forward(self, token_embs, coords, padding_mask=None):
        token_embs = token_embs + self.attn(token_embs, coords, padding_mask)
        return token_embs + self.mlp(token_embs)


class SpatialTransformer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.blks = nn.ModuleList(
            [
                TransformerBlock(
                    cfg["d_model"],
                    n_heads=cfg["n_heads"],
                    attn_drop=cfg["attn_dropout"],
                    proj_drop=cfg["dropout"],
                    mlp_ratio=cfg["mlp_ratio"],
                )
                for _ in range(cfg["n_layers"])
            ]
        )

    def forward(self, features, coords, batch_idx):
        batch_mask = ~(batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1))
        features = features.unsqueeze(0)
        coords = coords.unsqueeze(0)
        batch_mask = batch_mask.unsqueeze(0)
        for blk in self.blks:
            features = blk(features, coords, padding_mask=batch_mask)
        return features.squeeze(0)


class EncodeInputs(nn.Module):
    def __init__(self, cfg: dict, n_genes: int, n_tech: int, n_organs: int):
        super().__init__()
        self.image_embed = nn.Linear(cfg["feature_dim"], cfg["d_model"])
        self.gene_embed = nn.Linear(n_genes, cfg["d_model"], bias=False)
        self.tech_embed = nn.Embedding(n_tech, cfg["d_model"])
        self.organ_embed = nn.Embedding(n_organs, cfg["d_model"])

    def forward(self, img_tokens, ge_tokens, tech_tokens, organ_tokens):
        if ge_tokens is None:
            # Every spot is masked, i.e. ge_tokens is one-hot on the mask token.
            # ``Linear(n_genes, d_model, bias=False)`` applied to that one-hot is
            # exactly the corresponding weight column, so take it directly rather
            # than materialising an [n_spots, 38984] matrix of zeros.
            gene_embed = self.gene_embed.weight[:, MASK_TOKEN_ID]
        else:
            gene_embed = self.gene_embed(ge_tokens)
        return (
            self.image_embed(img_tokens)
            + gene_embed
            + self.tech_embed(tech_tokens)
            + self.organ_embed(organ_tokens)
        )


class STFM(nn.Module):
    """STPath's spatial foundation model, inference path only."""

    def __init__(
        self,
        n_genes: int,
        n_tech: int,
        n_organs: int,
        cfg: dict | None = None,
    ):
        super().__init__()
        cfg = {**DEFAULT_CONFIG, **(cfg or {})}
        self.input_encoder = EncodeInputs(cfg, n_genes, n_tech, n_organs)
        self.model = SpatialTransformer(cfg)
        self.gene_exp_head = nn.Sequential(
            nn.LayerNorm(cfg["d_model"]),
            nn.Linear(cfg["d_model"], n_genes),
        )

    def prediction_head(
        self,
        img_tokens,
        coords,
        batch_idx,
        tech_tokens,
        organ_tokens,
        ge_tokens=None,
    ):
        """Predict log1p expression for every spot. ``ge_tokens=None`` masks all spots."""
        x = self.input_encoder(
            img_tokens=img_tokens,
            ge_tokens=ge_tokens,
            tech_tokens=tech_tokens,
            organ_tokens=organ_tokens,
        )
        return self.gene_exp_head(self.model(x, coords, batch_idx))

    forward = prediction_head


def rescale_coords(coords: torch.Tensor, new_max: float = 100) -> torch.Tensor:
    """Shift to the origin and rescale to ``[0, new_max]``, as STPath trains on."""
    coords = coords - coords.min(dim=0).values
    coord_range = coords.max(dim=0).values
    # A degenerate axis (every spot in one row or column) has zero range;
    # upstream divides by it and yields NaN. Leave those coordinates at 0.
    coord_range = torch.where(
        coord_range > 0, coord_range, torch.ones_like(coord_range)
    )
    return coords / coord_range * new_max
