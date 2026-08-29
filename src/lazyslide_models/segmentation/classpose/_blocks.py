"""Network blocks vendored from Classpose (``src/classpose/{unet,vit_sam}.py``).

Upstream: https://github.com/sohmandal/classpose

Vendored rather than imported because the ``classpose`` distribution pins
``timm>=0.4.0,<0.5`` against our ``timm>=1.0.3``, requires Python >=3.13 against
our >=3.11, and is not published on PyPI. Only the inference path is kept: the
training-time helpers (``freeze_*``, ``save_model``, ``load_classification_head``)
and the random layer-drop branch of ``forward`` are dropped.

``ClassTransformer`` subclasses Cellpose's own ``Transformer``, so the backbone
comes from the installed ``cellpose`` package and only the semantic head is
reproduced here.
"""

from __future__ import annotations

from itertools import pairwise
from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn


class UNetBlock(nn.Module):
    """Two 3x3 convolutions with ReLU, the basic unit of the semantic head."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, skip_last_activation: bool = False):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        if not skip_last_activation:
            x = self.relu(x)
        return x


class UNetBlockDown(nn.Module):
    """A :class:`UNetBlock` followed by a strided down-convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = UNetBlock(in_channels, out_channels)
        self.downconv = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, skip_last_activation: bool = False):
        x = self.block(x, skip_last_activation=skip_last_activation)
        return x, self.downconv(x)


class UNetBlockUp(nn.Module):
    """A :class:`UNetBlock` followed by a transposed convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = UNetBlock(in_channels, out_channels)
        self.upconv = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=2, stride=2
        )

    def forward(self, x, skip_last_activation: bool = False):
        x = self.block(x, skip_last_activation=skip_last_activation)
        return self.upconv(x)


class UNet(nn.Module):
    """Optional UNet variant of the Classpose semantic head.

    Checkpoints that use it are detected from the presence of
    ``out_class.encoder_blocks.*`` keys in the state dict; the others use a
    single ``Conv2d``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_channels: list[int] | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels = n_channels or [64, 128, 256, 512]

        in_seq = [self.in_channels, *self.n_channels]
        out_seq = [*self.n_channels[::-1], self.out_channels]
        self.encoder_blocks = nn.ModuleList(
            [UNetBlockDown(i, o) for i, o in pairwise(in_seq)]
        )
        self.decoder_blocks = nn.ModuleList(
            [UNetBlockUp(i * 2, o) for i, o in pairwise(out_seq)]
        )
        self.bottleneck_down = UNetBlockDown(in_seq[-1], in_seq[-1])
        self.bottleneck_up = UNetBlockUp(in_seq[-1], in_seq[-1])

    def forward(self, x):
        encoded = []
        for block in self.encoder_blocks:
            _, x = block(x)
            encoded.append(x)
        encoded = encoded[::-1]
        _, x = self.bottleneck_down(x)
        x = self.bottleneck_up(x)
        n_dec = len(self.decoder_blocks)
        for i, block in enumerate(self.decoder_blocks):
            x = block(
                torch.cat((x, encoded[i]), dim=1),
                skip_last_activation=i == (n_dec - 1),
            )
        return x


def _flash_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for SAM's ``Attention.forward`` using SDPA.

    Numerically equivalent to the upstream implementation, but it never
    materialises the ``[B, heads, L, L]`` attention matrix, which is what makes
    ViT-L inference on 256x256 tiles fit in a sensible amount of memory.
    """
    from segment_anything.modeling.image_encoder import get_rel_pos

    B, H, W, _ = x.shape
    L = H * W

    qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    attn_mask = None
    if getattr(self, "use_rel_pos", False):
        head_dim = q.shape[-1]
        q_hw = q.reshape(B, self.num_heads, H, W, head_dim)
        Rh = get_rel_pos(H, H, self.rel_pos_h)
        Rw = get_rel_pos(W, W, self.rel_pos_w)
        rel_h = torch.einsum("b n h w c, h k c -> b n h w k", q_hw, Rh)
        rel_w = torch.einsum("b n h w c, w k c -> b n h w k", q_hw, Rw)
        attn_mask = (rel_h[..., :, None] + rel_w[..., None, :]).reshape(
            B, self.num_heads, L, L
        )

    x = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False, scale=self.scale
    )
    return self.proj(x.transpose(1, 2).reshape(B, H, W, -1))


def patch_attention_forwards(model: nn.Module, attn_class_name: str = "Attention"):
    """Swap every SAM ``Attention.forward`` for the SDPA version."""
    for module in model.modules():
        if module.__class__.__name__ == attn_class_name:
            module.forward = MethodType(_flash_forward, module)


def build_class_transformer(
    n_cell_classes: int,
    feature_transformation_structure: list[int] | None = None,
    ps: int = 8,
    bsize: int = 256,
    dtype: torch.dtype = torch.float32,
):
    """Build the Classpose network: Cellpose-SAM plus a semantic head.

    Returns a ``cellpose.vit_sam.Transformer`` extended with ``out_class`` and
    ``W3``. ``forward`` yields ``[B, n_cell_classes + 3, H, W]`` — the semantic
    logits first, then Cellpose's ``(dY, dX, cellprob)``.
    """
    from cellpose.vit_sam import Transformer

    class ClassTransformer(Transformer):
        def __init__(self):
            super().__init__(ps=ps, nout=3, bsize=bsize, rdrop=0.0, dtype=dtype)
            self.n_cell_classes = n_cell_classes
            if feature_transformation_structure is not None:
                self.out_class = UNet(
                    in_channels=256,
                    out_channels=n_cell_classes * ps**2,
                    n_channels=list(feature_transformation_structure),
                )
            else:
                self.out_class = nn.Conv2d(256, n_cell_classes * ps**2, kernel_size=1)
            self.W3 = nn.Parameter(
                torch.eye(n_cell_classes * ps**2).reshape(
                    n_cell_classes * ps**2, n_cell_classes, ps, ps
                ),
                requires_grad=False,
            )
            patch_attention_forwards(self)

        def forward(self, x):
            x = self.encoder.patch_embed(x)
            if self.encoder.pos_embed is not None:
                x = x + self.encoder.pos_embed
            for blk in self.encoder.blocks:
                x = blk(x)
            x = self.encoder.neck(x.permute(0, 3, 1, 2))

            flows = F.conv_transpose2d(self.out(x), self.W2, stride=self.ps, padding=0)
            logits = F.conv_transpose2d(
                self.out_class(x), self.W3, stride=self.ps, padding=0
            )
            return torch.cat((logits, flows), 1)

    return ClassTransformer()
