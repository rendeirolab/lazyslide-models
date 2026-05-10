"""
Unified model test suite — protocol / method-based dispatch.

Tests are parametrised by **which methods a model class exposes** (via
``hasattr`` on the class), not by ``ModelTask``.  Adding a new model to the
registry automatically includes it in every test whose required method it
implements — no changes to this file required.

Device is configured via ``--device`` CLI flag (default: cpu).
Gated models carry the ``gated`` mark so they can be filtered with
``-m 'not gated'``.

Usage examples
--------------
pytest tests/test_models.py                              # CPU, all non-gated
pytest tests/test_models.py -m 'not gated'               # explicit filter
pytest tests/test_models.py --device=cuda                # GPU
pytest tests/test_models.py --device=mps                 # Apple Silicon
pytest tests/test_models.py -k uni                       # single model
pytest tests/test_models.py --skip-models=histoplus,sam  # manual exclusions
pytest tests/test_models.py -k encode_image              # one capability
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from conftest import all_models, models_with_method
from contracts import VALIDATOR
from inputs import INPUT_FACTORY

from lazyslide_models import MODEL_REGISTRY
from lazyslide_models.base import ModelTask

# ── Load references.bib keys once at import time ─────────────────────────────

BIB_FILE = Path(__file__).resolve().parent.parent / "references.bib"
BIB_KEYS: frozenset[str] = frozenset()
if BIB_FILE.exists():
    BIB_KEYS = frozenset(
        re.findall(r"@\w+\{([^,]+),", BIB_FILE.read_text(encoding="utf-8"))
    )

# ── Shared image-prep helper ──────────────────────────────────────────────────


def _prepare_image(model, image, device: str = "cpu"):
    """Apply model transform if present; otherwise return the raw image.

    Models with ``get_transform() is None`` handle preprocessing internally
    (e.g. via their own HuggingFace processor).  Passing a pre-converted
    tensor to those models would bypass their processor and cause errors.
    """
    transform = model.get_transform()
    if transform is None:
        # Return raw numpy image — the model's encode_image handles preprocessing
        return image
    t = transform(image)
    # Some transforms already return a batched tensor; add batch dim if not
    if isinstance(t, torch.Tensor) and t.ndim == 3:
        t = t.unsqueeze(0)
    if isinstance(t, torch.Tensor):
        t = t.to(device)
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Model attributes — every registered model must expose core metadata
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("model_name", all_models())
def test_model_attributes(model_name: str, load_model) -> None:
    """Every model must have name, task, license, commercial set"""
    cls = MODEL_REGISTRY[model_name]

    assert cls.name is not None, "model.name is None"

    task = getattr(cls, "task", None)
    assert task is not None, "cls.task is None"
    assert isinstance(task, (ModelTask, list)), (
        f"task should be ModelTask or list, got {type(task)}"
    )
    if cls.task != ModelTask.cv_feature:
        assert getattr(cls, "license", None) is not None, "cls.license is None"
        assert getattr(cls, "commercial", None) is not None, "cls.commercial is None"

    bib_key = getattr(cls, "bib_key", None)
    if bib_key is not None and BIB_KEYS:
        assert bib_key in BIB_KEYS, f"bib_key '{bib_key}' not found in references.bib"


# ═══════════════════════════════════════════════════════════════════════════════
# encode_image — models with encode_image method
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("model_name", models_with_method("encode_image"))
def test_encode_image(model_name: str, load_model, device: str) -> None:
    """encode_image() must return a 2-D float Tensor (B, D)."""
    model = load_model(model_name)
    inp = INPUT_FACTORY[ModelTask.vision](model)
    img = _prepare_image(model, inp.image, device)
    out = model.encode_image(img)
    VALIDATOR[ModelTask.vision](out)


# ═══════════════════════════════════════════════════════════════════════════════
# encode_image_dense — ViT models that expose patch-level features
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("model_name", models_with_method("encode_image_dense"))
def test_encode_image_dense(model_name: str, load_model, device: str) -> None:
    """encode_image_dense() must return a DenseTokens(cls_token, patch_tokens)."""
    from lazyslide_models.base import DenseTokens

    model = load_model(model_name)
    if model.get_transform() is None:
        pytest.skip(
            "model uses internal processor; encode_image_dense not testable via raw image"
        )
    inp = INPUT_FACTORY[ModelTask.vision](model)
    img = _prepare_image(model, inp.image, device)
    out = model.encode_image_dense(img)

    assert isinstance(out, DenseTokens), (
        f"encode_image_dense must return DenseTokens, got {type(out).__name__}"
    )

    assert isinstance(out.cls_token, torch.Tensor), "cls_token must be a Tensor"
    assert out.cls_token.ndim == 2, (
        f"cls_token should be (B, D), got shape {tuple(out.cls_token.shape)}"
    )
    assert out.cls_token.is_floating_point(), (
        f"cls_token expected float dtype, got {out.cls_token.dtype}"
    )

    assert isinstance(out.patch_tokens, torch.Tensor), "patch_tokens must be a Tensor"
    assert out.patch_tokens.ndim == 3, (
        f"patch_tokens should be (B, N, D), got shape {tuple(out.patch_tokens.shape)}"
    )
    assert out.patch_tokens.is_floating_point(), (
        f"patch_tokens expected float dtype, got {out.patch_tokens.dtype}"
    )

    assert out.cls_token.shape[-1] == out.patch_tokens.shape[-1], (
        f"embedding dim mismatch: cls_token {out.cls_token.shape[-1]} vs patch_tokens {out.patch_tokens.shape[-1]}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# encode_text — models with encode_text method
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("model_name", models_with_method("encode_text"))
def test_encode_text(model_name: str, load_model, device: str) -> None:
    """encode_image + encode_text must return matching (B, D) float Tensors."""
    model = load_model(model_name)
    inp = INPUT_FACTORY[ModelTask.multimodal](model)
    img = _prepare_image(model, inp.image, device)
    img_emb = model.encode_image(img)
    txt_emb = model.encode_text(inp.texts)
    VALIDATOR[ModelTask.multimodal](img_emb, txt_emb)


# ═══════════════════════════════════════════════════════════════════════════════
# segment — models with segment method
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("model_name", models_with_method("segment"))
def test_segment(model_name: str, load_model, device: str) -> None:
    """segment() must return a dict with keys from the canonical set."""
    model = load_model(model_name)
    inp = INPUT_FACTORY[ModelTask.segmentation](model)
    transform = model.get_transform()
    if transform is not None:
        img = transform(inp.image)
        if img.ndim == 3:
            img = img.unsqueeze(0)
    else:
        img = torch.from_numpy(inp.image).unsqueeze(
            0
        )  # keep uint8 for models that want it
    if isinstance(img, torch.Tensor):
        img = img.to(device)
    out = model.segment(img)
    VALIDATOR[ModelTask.segmentation](out)


# ═══════════════════════════════════════════════════════════════════════════════
# encode_slide — models with encode_slide method
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("model_name", models_with_method("encode_slide"))
def test_encode_slide(model_name: str, load_model, device: str) -> None:
    """encode_slide() must return a dict with 'embedding' key (float Tensor, 1-D or 2-D)."""
    model = load_model(model_name)
    inp = INPUT_FACTORY[ModelTask.slide_encoder](model)
    # Add batch dim: (1, N, D) embeddings and (1, N, 2) coords
    out = model.encode_slide(
        inp.embeddings.unsqueeze(0).to(device),
        coords=inp.coords.unsqueeze(0).to(device),
        base_tile_size=256,
    )
    VALIDATOR[ModelTask.slide_encoder](out)


# ═══════════════════════════════════════════════════════════════════════════════
# predict — tile prediction / cv_feature (NOT style transfer)
# ═══════════════════════════════════════════════════════════════════════════════


def _predict_models() -> list[pytest.param]:
    """Models with predict() but WITHOUT get_channel_names() (style transfer)."""
    all_predict = {p.values[0] for p in models_with_method("predict")}
    style = {p.values[0] for p in models_with_method("predict", "get_channel_names")}
    names = sorted(all_predict - style)
    params = []
    for name in names:
        cls = MODEL_REGISTRY[name]
        marks = [pytest.mark.gated] if getattr(cls, "is_gated", False) else []
        params.append(pytest.param(name, marks=marks, id=name))
    return params


@pytest.mark.parametrize("model_name", _predict_models())
def test_predict(model_name: str, load_model, device: str) -> None:
    """predict() must return a dict of numpy arrays."""
    model = load_model(model_name)
    # Resolve task for input lookup
    raw_task = MODEL_REGISTRY[model_name].task
    task = raw_task[0] if isinstance(raw_task, list) else raw_task
    inp = INPUT_FACTORY[task](model)
    transform = model.get_transform()
    if transform is not None:
        img = transform(inp.image)
        if isinstance(img, torch.Tensor) and img.ndim == 3:
            img = img.unsqueeze(0)
    else:
        img = inp.image  # cv_feature models accept raw numpy
    if isinstance(img, torch.Tensor):
        img = img.to(device)
    out = model.predict(img)
    VALIDATOR[task](out)


# ═══════════════════════════════════════════════════════════════════════════════
# style transfer — predict + get_channel_names
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "model_name", models_with_method("predict", "get_channel_names")
)
def test_style_transfer(model_name: str, load_model, device: str) -> None:
    """predict() must return a float Tensor, 3-D or 4-D."""
    model = load_model(model_name)
    inp = INPUT_FACTORY[ModelTask.style_transfer](model)
    img = _prepare_image(model, inp.image, device)
    with torch.inference_mode():
        out = model.predict(img)
    VALIDATOR[ModelTask.style_transfer](out)


# ═══════════════════════════════════════════════════════════════════════════════
# image generation — generate method
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("model_name", models_with_method("generate"))
def test_image_generation(model_name: str, load_model, device: str) -> None:
    """generate() must return non-None."""
    model = load_model(model_name)
    with torch.inference_mode():
        out = model.generate()
    VALIDATOR[ModelTask.image_generation](out)
