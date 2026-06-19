"""
Mock input factories for each model task.

Pure functions — no model names, no if/else.
Input size is derived from ``model.input_constraint`` (set by ``@register``).
Slide encoder embedding dimension is read from ``model.encode_dim``.

The ``INPUT_FACTORY`` dispatch table maps ``ModelTask`` → factory function.
Extend only this table when a new task is added.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

from lazyslide_models.base import ModelTask

_RNG = np.random.default_rng(42)


# ── Input bundles ─────────────────────────────────────────────────────────────


class VisionInputs(NamedTuple):
    image: np.ndarray  # (H, W, 3) uint8


class MultimodalInputs(NamedTuple):
    image: np.ndarray  # (H, W, 3) uint8
    texts: list[str]


class SegmentationInputs(NamedTuple):
    image: np.ndarray  # (H, W, 3) uint8


class SlideEncoderInputs(NamedTuple):
    embeddings: torch.Tensor  # (N, D)
    coords: torch.Tensor  # (N, 2) float pixel coords


class TilePredictionInputs(NamedTuple):
    image: np.ndarray  # (H, W, 3) uint8


class FeaturePredictionInputs(NamedTuple):
    features: np.ndarray  # (N, D) float32


class StyleTransferInputs(NamedTuple):
    image: np.ndarray  # (H, W, 3) uint8


class ImageGenerationInputs(NamedTuple):
    pass  # generate() takes no arguments


# ── Shared helper ─────────────────────────────────────────────────────────────


def _random_image(model=None, size: int | None = None) -> np.ndarray:
    """Return a uint8 HWC numpy image.

    Parameters
    ----------
    model : optional
        When *size* is ``None``, the image uses the default size from
        ``model.input_constraint`` (or 224).
    size : int, optional
        Explicit spatial size — overrides the model's preferred size.
        Used by the multi-size transform tests.
    """
    if size is None:
        constraint = getattr(model, "input_constraint", None)
        if constraint is not None:
            size = constraint.default_size
        else:
            size = 224
    return _RNG.integers(0, 256, (size, size, 3), dtype=np.uint8)


# ── Factories ─────────────────────────────────────────────────────────────────


def make_vision(model=None, size: int | None = None) -> VisionInputs:
    return VisionInputs(_random_image(model, size=size))


def make_multimodal(model=None, size: int | None = None) -> MultimodalInputs:
    return MultimodalInputs(
        image=_random_image(model, size=size),
        texts=[
            "A histopathology tissue image.",
            "Tumor cells with high mitotic index.",
        ],
    )


def make_segmentation(model=None, size: int | None = None) -> SegmentationInputs:
    return SegmentationInputs(_random_image(model, size=size))


def _slide_input_dim(model) -> int:
    """Resolve the expected input embedding dim for a slide encoder.

    Slide encoders consume tile embeddings from an upstream vision encoder.
    If the model declares ``vision_encoder``, look up that encoder's
    ``encode_dim`` in the registry.  Otherwise fall back to the model's own
    ``encode_dim``, then 768.
    """
    from lazyslide_models import MODEL_REGISTRY

    vision_encoder = getattr(model, "vision_encoder", None)
    if vision_encoder and vision_encoder in MODEL_REGISTRY:
        dim = getattr(MODEL_REGISTRY[vision_encoder], "encode_dim", None)
        if dim is not None:
            return dim
    return getattr(model, "encode_dim", None) or 768


def make_slide_encoder(model=None) -> SlideEncoderInputs:
    """64-patch grid; embedding dim resolved from upstream vision encoder."""
    D = _slide_input_dim(model)
    N = 64
    embeddings = torch.randn(N, D)
    xs = torch.arange(8) * 256
    ys = torch.arange(8) * 256
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([gx.flatten(), gy.flatten()], dim=1).long()
    return SlideEncoderInputs(embeddings, coords)


def make_tile_prediction(model=None, size: int | None = None) -> TilePredictionInputs:
    return TilePredictionInputs(_random_image(model, size=size))


def make_feature_prediction(
    model=None, size: int | None = None
) -> FeaturePredictionInputs:
    from lazyslide_models import MODEL_REGISTRY

    encoder_name = getattr(model, "features_model_name", None)
    encoder = MODEL_REGISTRY.get(encoder_name) if encoder_name else None
    feature_dim = getattr(encoder, "encode_dim", None) or 768
    features = _RNG.standard_normal((2, feature_dim), dtype=np.float32)
    return FeaturePredictionInputs(features)


def make_style_transfer(model=None, size: int | None = None) -> StyleTransferInputs:
    return StyleTransferInputs(_random_image(model, size=size))


def make_image_generation(model=None) -> ImageGenerationInputs:
    return ImageGenerationInputs()


# ── Dispatch table ────────────────────────────────────────────────────────────

INPUT_FACTORY: dict = {
    ModelTask.vision: make_vision,
    ModelTask.multimodal: make_multimodal,
    ModelTask.segmentation: make_segmentation,
    ModelTask.slide_encoder: make_slide_encoder,
    ModelTask.tile_prediction: make_tile_prediction,
    ModelTask.cv_feature: make_tile_prediction,
    ModelTask.feature_prediction: make_feature_prediction,
    ModelTask.style_transfer: make_style_transfer,
    ModelTask.image_generation: make_image_generation,
}
