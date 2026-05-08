from __future__ import annotations

import gc
import re
from collections import Counter

import pytest
import torch

from lazyslide_models import MODEL_REGISTRY

# ── CLI options ───────────────────────────────────────────────────────────────


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for model tests (default: cpu)",
    )
    parser.addoption(
        "--skip-models",
        default="",
        help="Comma-separated model names to skip (e.g. 'gigapath,sam')",
    )


# ── Test reordering: group by model name ─────────────────────────────────────

_MODEL_NAME_RE = re.compile(r"\[([^\]]+)\]")

# Tracks how many remaining tests need each model (for eviction).
_model_remaining: Counter[str] = Counter()


def _extract_model_name(nodeid: str) -> str | None:
    """Extract model name from parametrized test node id like ``test_foo[plip]``."""
    m = _MODEL_NAME_RE.search(nodeid)
    return m.group(1) if m else None


def _model_sort_key(item: pytest.Item) -> tuple[int, str, str]:
    """Sort key: (weight_class, model_name, test_name).

    Weight classes:
      0 — lightweight cv_feature models (no torch, instant init)
      1 — non-gated heavy models (actually load weights)
      2 — gated models (skip instantly without HF credentials)
    """
    name = _extract_model_name(item.nodeid)
    if name is None:
        return (0, "", item.name)

    cls = MODEL_REGISTRY.get(name)
    if cls is None:
        return (0, name, item.name)

    if getattr(cls, "is_gated", False):
        weight = 2
    elif getattr(cls, "task", None) and (
        not hasattr(cls, "model")
        and not hasattr(cls, "encode_image")
        and hasattr(cls, "predict")
    ):
        # cv_feature / lightweight predict-only models
        weight = 0
    else:
        weight = 1

    return (weight, name, item.name)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Reorder tests so all tests for model X run consecutively, then eviction works."""
    items.sort(key=_model_sort_key)

    # Pre-count tests per model for eviction tracking
    _model_remaining.clear()
    for item in items:
        name = _extract_model_name(item.nodeid)
        if name:
            _model_remaining[name] += 1


# ── Session fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def device(request: pytest.FixtureRequest) -> str:
    d = request.config.getoption("--device")
    if d == "cuda" and not torch.cuda.is_available():
        pytest.skip("--device=cuda requested but CUDA is not available")
    if d == "mps" and not torch.backends.mps.is_available():
        pytest.skip("--device=mps requested but MPS is not available")
    return d


@pytest.fixture(scope="session")
def skip_models(request: pytest.FixtureRequest) -> frozenset[str]:
    raw = request.config.getoption("--skip-models")
    return frozenset(n.strip() for n in raw.split(",") if n.strip())


@pytest.fixture(scope="session")
def load_model(device: str, skip_models: frozenset[str]):
    """
    Session-scoped factory fixture.

    Call ``load_model(model_name)`` inside a test to get an initialised,
    device-placed model.  All skip logic (gated, missing deps, not-implemented,
    manual --skip-models) is handled here so test functions stay clean.

    Each model is loaded once and cached for the whole session.
    """
    from huggingface_hub.errors import GatedRepoError

    cache: dict[str, object] = {}

    def _load(name: str):
        if name in cache:
            return cache[name]

        if name in skip_models:
            pytest.skip(f"'{name}' in --skip-models list")

        try:
            model = MODEL_REGISTRY[name]()
        except GatedRepoError:
            pytest.skip(f"'{name}' is gated (no HF credentials present)")
        except Exception as exc:
            pytest.skip(f"'{name}' failed to load: {type(exc).__name__}: {exc}")

        model.to(device)
        cache[name] = model
        return model

    def _release(name: str) -> None:
        """Remove model from cache and free memory."""
        model = cache.pop(name, None)
        if model is not None:
            # Move to CPU first if on accelerator (frees device memory)
            try:
                model.to("cpu")
            except Exception:
                pass
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.mps.is_available():
                torch.mps.empty_cache()

    _load.release = _release  # type: ignore[attr-defined]
    _load.cache = cache  # type: ignore[attr-defined]
    return _load


# ── Model eviction after last use ────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _evict_after_last_use(request: pytest.FixtureRequest, load_model) -> None:
    """After each test, evict the model if no more tests need it."""
    yield
    name = _extract_model_name(request.node.nodeid)
    if name is None:
        return
    _model_remaining[name] -= 1
    if _model_remaining[name] <= 0:
        load_model.release(name)


# ── Inference mode for all tests ─────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _inference_mode():
    """Disable autograd for all tests — no test needs gradients."""
    with torch.inference_mode():
        yield


# ── Parametrization helper ────────────────────────────────────────────────────


def _gated_marks(cls) -> list:
    """Return ``[pytest.mark.gated]`` when the model class is gated, else ``[]``."""
    return [pytest.mark.gated] if getattr(cls, "is_gated", False) else []


def models_with_method(*methods: str) -> list[pytest.param]:
    """Return ``pytest.param`` for every model whose class defines **all** listed methods.

    Uses ``hasattr`` on the *class* (not an instance) so no model weights are
    downloaded at collection time.  Correctly captures models that implement
    methods without inheriting the "canonical" base class.

    When a single class is registered under multiple keys (e.g. ``titan`` /
    ``conch_v1.5``), only the **first** key is kept to avoid loading and
    testing the exact same model twice.
    """
    seen_cls: set[int] = set()
    params = []
    for name, cls in MODEL_REGISTRY.items():
        if id(cls) in seen_cls:
            continue
        if all(hasattr(cls, m) for m in methods):
            seen_cls.add(id(cls))
            params.append(pytest.param(name, marks=_gated_marks(cls), id=name))
    return params


def all_models() -> list[pytest.param]:
    """Return ``pytest.param`` for every registered model.

    Deduplicates classes registered under multiple keys — only the first
    key is parametrized so the same model is not tested twice.
    """
    seen_cls: set[int] = set()
    params = []
    for name, cls in MODEL_REGISTRY.items():
        if id(cls) in seen_cls:
            continue
        seen_cls.add(id(cls))
        params.append(pytest.param(name, marks=_gated_marks(cls), id=name))
    return params
