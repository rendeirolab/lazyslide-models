"""
Output contract validators for each model task.

Pure functions — no model names, no if/else.
Each function raises ``AssertionError`` with a descriptive message when the
contract is violated.

The ``VALIDATOR`` dispatch table maps ``ModelTask`` → validator function.
Extend only this table when a new task is added.

Note: ``check_multimodal`` takes two arguments (img_emb, txt_emb).
All other validators take a single ``output`` argument.
"""

from __future__ import annotations

import numpy as np
import torch

from lazyslide_models.base import ModelTask, SegmentationOutput

# ── Shared helpers ────────────────────────────────────────────────────────────


def _tensor(out, tag: str) -> torch.Tensor:
    assert isinstance(out, torch.Tensor), (
        f"{tag}: expected torch.Tensor, got {type(out).__name__}"
    )
    return out


def _2d_float(t: torch.Tensor, tag: str) -> None:
    assert t.ndim == 2, f"{tag}: expected 2-D tensor, got shape {tuple(t.shape)}"
    assert t.is_floating_point(), f"{tag}: expected float dtype, got {t.dtype}"


def _dict(out, tag: str) -> dict:
    assert isinstance(out, dict), f"{tag}: expected dict, got {type(out).__name__}"
    return out


# ── Per-task validators ───────────────────────────────────────────────────────


def check_vision(output) -> None:
    """encode_image → Tensor (B, D), float."""
    _2d_float(_tensor(output, "encode_image"), "encode_image")


def check_multimodal(img_emb, txt_emb) -> None:
    """encode_image + encode_text → both Tensor (B, D) float with same D."""
    _2d_float(_tensor(img_emb, "image_embedding"), "image_embedding")
    _2d_float(_tensor(txt_emb, "text_embedding"), "text_embedding")
    assert img_emb.shape[1] == txt_emb.shape[1], (
        f"Embedding dim mismatch: image={img_emb.shape[1]}, text={txt_emb.shape[1]}"
    )


def check_segmentation(output) -> None:
    """segment → SegmentationOutput NamedTuple."""
    assert isinstance(output, SegmentationOutput), (
        f"segment(): expected SegmentationOutput, got {type(output).__name__}"
    )

    prob = output.probability_map
    inst = output.instance_map
    token = output.patch_token_map
    classes = output.classes

    assert prob is not None or inst is not None, (
        "segment(): at least probability_map or instance_map must be set"
    )

    if prob is not None:
        assert isinstance(prob, (torch.Tensor, np.ndarray)), (
            f"segment().probability_map: expected Tensor or ndarray, got {type(prob).__name__}"
        )
        p = torch.as_tensor(prob)
        assert p.ndim == 4, (
            f"segment().probability_map: expected 4-D [B, C, H, W], got shape {tuple(p.shape)}"
        )
        assert p.is_floating_point(), (
            f"segment().probability_map: expected float dtype, got {p.dtype}"
        )

    if inst is not None:
        assert isinstance(inst, (torch.Tensor, np.ndarray)), (
            f"segment().instance_map: expected Tensor or ndarray, got {type(inst).__name__}"
        )
        i = torch.as_tensor(inst)
        assert i.ndim == 3, (
            f"segment().instance_map: expected 3-D [B, H, W], got shape {tuple(i.shape)}"
        )

    if token is not None:
        assert isinstance(token, (torch.Tensor, np.ndarray)), (
            f"segment().patch_token_map: expected Tensor or ndarray, got {type(token).__name__}"
        )
        t = torch.as_tensor(token)
        assert t.ndim == 4, (
            f"segment().patch_token_map: expected 4-D [B, D, H, W], got shape {tuple(t.shape)}"
        )
        assert t.is_floating_point(), (
            f"segment().patch_token_map: expected float dtype, got {t.dtype}"
        )

    if classes is not None:
        assert isinstance(classes, tuple), (
            f"segment().classes: expected tuple, got {type(classes).__name__}"
        )
        assert all(isinstance(c, str) for c in classes), (
            "segment().classes: all entries must be str"
        )
        if prob is not None:
            n_channels = torch.as_tensor(prob).shape[1]
            assert len(classes) == n_channels, (
                f"segment().classes length ({len(classes)}) != "
                f"probability_map channels ({n_channels})"
            )


def check_slide_encoder(output) -> None:
    """encode_slide → dict with at least 'embedding' key containing a float Tensor."""
    d = _dict(output, "encode_slide()")
    assert "embedding" in d, (
        f"encode_slide() must return a dict with 'embedding' key, got keys: {set(d.keys())}"
    )
    t = _tensor(d["embedding"], "encode_slide()['embedding']")
    assert t.is_floating_point(), (
        f"encode_slide()['embedding']: expected float dtype, got {t.dtype}"
    )
    assert t.ndim in (1, 2), (
        f"encode_slide()['embedding']: expected 1-D or 2-D tensor, got shape {tuple(t.shape)}"
    )
    # Validate any extra values are tensors
    for key, val in d.items():
        if key != "embedding":
            assert isinstance(val, torch.Tensor), (
                f"encode_slide()['{key}']: expected Tensor, got {type(val).__name__}"
            )


def check_tile_prediction(output) -> None:
    """predict → dict of numpy arrays."""
    d = _dict(output, "predict()")
    for key, val in d.items():
        assert isinstance(val, np.ndarray), (
            f"predict()['{key}']: expected numpy.ndarray, got {type(val).__name__}"
        )


def check_style_transfer(output) -> None:
    """predict → float Tensor, 2-D (B,C) or 4-D (B,C,H,W)."""
    t = _tensor(output, "StyleTransferModel.predict")
    assert t.is_floating_point(), f"style predict: expected float dtype, got {t.dtype}"
    assert t.ndim in (2, 4), (
        f"style predict: expected 2-D (tile->values) or 4-D (tile->image) tensor, got shape {tuple(t.shape)}"
    )


def check_image_generation(output) -> None:
    """generate() → non-None."""
    assert output is not None, "ImageGenerationModel.generate() returned None"


# ── Dispatch table ────────────────────────────────────────────────────────────

VALIDATOR: dict = {
    ModelTask.vision: check_vision,
    ModelTask.multimodal: check_multimodal,  # (img_emb, txt_emb)
    ModelTask.segmentation: check_segmentation,
    ModelTask.slide_encoder: check_slide_encoder,
    ModelTask.tile_prediction: check_tile_prediction,
    ModelTask.cv_feature: check_tile_prediction,
    ModelTask.feature_prediction: check_tile_prediction,
    ModelTask.style_transfer: check_style_transfer,
    ModelTask.image_generation: check_image_generation,
}
