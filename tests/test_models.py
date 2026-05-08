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

import pytest
import torch
from conftest import all_models, models_with_method
from contracts import VALIDATOR
from inputs import INPUT_FACTORY

from lazyslide_models import MODEL_REGISTRY
from lazyslide_models.base import ModelTask

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
    """encode_image_dense() must return a 3-D float Tensor (B, N_patches, D)."""
    model = load_model(model_name)
    if model.get_transform() is None:
        pytest.skip(
            "model uses internal processor; encode_image_dense not testable via raw image"
        )
    inp = INPUT_FACTORY[ModelTask.vision](model)
    img = _prepare_image(model, inp.image, device)
    out = model.encode_image_dense(img)
    assert isinstance(out, torch.Tensor), "encode_image_dense must return Tensor"
    assert out.ndim == 3, f"expected (B, N, D) tensor, got shape {tuple(out.shape)}"
    assert out.is_floating_point(), f"expected float dtype, got {out.dtype}"


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
    """encode_slide() must return a float Tensor, 1-D or 2-D."""
    model = load_model(model_name)
    inp = INPUT_FACTORY[ModelTask.slide_encoder](model)
    # Add batch dim: (1, N, D) embeddings and (1, N, 2) coords
    out = model.encode_slide(
        inp.embeddings.unsqueeze(0).to(device),
        coords=inp.coords.unsqueeze(0).to(device),
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
