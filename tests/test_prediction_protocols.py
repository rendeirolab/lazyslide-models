"""Tests for the three prediction model classes.

None of these load weights — they read class attributes only, so the whole file
runs for every model on a selective CI run. The nine ``cv_feature`` models are
pure OpenCV/NumPy, so their declared ``columns`` are checked against a real
``predict`` call rather than trusted.

Note there is deliberately no check that a model's tensor actually has as many
channels as it declares; see the plan's "Explicitly not doing".
"""

from __future__ import annotations

import numpy as np
import pytest

from lazyslide_models import MODEL_REGISTRY
from lazyslide_models.base import (
    DensePredictionModel,
    MarkerMapModel,
    ModelTask,
    TilePredictionModel,
    VirtualStainModel,
)

#: The tasks whose models go through ``predict(image)``.
PREDICTION_TASKS = (
    ModelTask.tile_prediction,
    ModelTask.style_transfer,
    ModelTask.cv_feature,
)

PREDICTION_MODELS = sorted(
    key
    for key, cls in MODEL_REGISTRY.items()
    if getattr(cls, "task", None) in PREDICTION_TASKS
)

CV_FEATURE_MODELS = sorted(
    key
    for key, cls in MODEL_REGISTRY.items()
    if getattr(cls, "task", None) is ModelTask.cv_feature
)

CONCRETE = (TilePredictionModel, MarkerMapModel, VirtualStainModel)


def _names(cls) -> tuple | None:
    """Whichever naming attribute this class is supposed to carry."""
    if issubclass(cls, MarkerMapModel):
        return getattr(cls, "channel_names", None)
    if issubclass(cls, VirtualStainModel):
        return getattr(cls, "stains", None)
    return getattr(cls, "columns", None)


# ── Every prediction model picks exactly one class ────────────────────────────


@pytest.mark.parametrize("model_name", PREDICTION_MODELS)
def test_model_subclasses_exactly_one_prediction_class(model_name: str) -> None:
    """The class is the discriminator, so overlapping bases would be ambiguous."""
    cls = MODEL_REGISTRY[model_name]
    matched = [base for base in CONCRETE if issubclass(cls, base)]
    assert len(matched) == 1, (
        f"{model_name}: {cls.__name__} should subclass exactly one of "
        f"{[b.__name__ for b in CONCRETE]}, matched {[b.__name__ for b in matched]}"
    )


@pytest.mark.parametrize("model_name", PREDICTION_MODELS)
def test_declared_names_are_a_tuple_of_str(model_name: str) -> None:
    """A list would be shared mutable state across every instance."""
    names = _names(MODEL_REGISTRY[model_name])
    if names is None:
        return  # only known after loading weights, e.g. deepspotm
    assert isinstance(names, tuple), (
        f"{model_name}: expected a tuple, got {type(names).__name__}"
    )
    assert all(isinstance(n, str) for n in names), f"{model_name}: names must be str"


# ── The dense classes must name their output ──────────────────────────────────


@pytest.mark.parametrize("model_name", PREDICTION_MODELS)
def test_dense_models_name_their_output(model_name: str) -> None:
    """A dense output is stitched into an image the runner has to label."""
    cls = MODEL_REGISTRY[model_name]
    if not issubclass(cls, DensePredictionModel):
        return
    assert _names(cls), (
        f"{model_name}: a dense model must declare non-empty "
        f"{'stains' if issubclass(cls, VirtualStainModel) else 'channel_names'}"
    )


# ── The replaced mechanisms are gone ──────────────────────────────────────────


@pytest.mark.parametrize("model_name", PREDICTION_MODELS)
@pytest.mark.parametrize("attr", ["get_channel_names", "output_spec", "output_shape"])
def test_superseded_attributes_are_removed(model_name: str, attr: str) -> None:
    """Keeping an old mechanism alongside the new one invites drift."""
    cls = MODEL_REGISTRY[model_name]
    assert not hasattr(cls, attr), (
        f"{model_name}: {attr} is superseded by the prediction class attributes"
    )


# ── cv_feature: declared columns checked against a real call ──────────────────


@pytest.mark.parametrize("model_name", CV_FEATURE_MODELS)
def test_cv_feature_columns_match_what_predict_returns(model_name: str) -> None:
    """These need no weights, so the declaration can be verified rather than trusted."""
    model = MODEL_REGISTRY[model_name]()
    image = (np.random.default_rng(0).random((2, 64, 64, 3)) * 255).astype("uint8")
    assert tuple(model.predict(image)) == model.columns


def test_cv_compose_reports_the_columns_of_what_it_composes() -> None:
    """``CVCompose`` merges its children's dicts, so the class-name default
    (``("cvcompose",)``) would be actively wrong. It computes its own."""
    from lazyslide_models.tile_prediction.cv_features import (
        Brightness,
        CVCompose,
        SplitRGB,
    )

    model = CVCompose(Brightness(), SplitRGB())
    image = (np.random.default_rng(0).random((2, 64, 64, 3)) * 255).astype("uint8")
    assert model.columns == ("brightness", "red", "green", "blue")
    assert tuple(model.predict(image)) == model.columns


# ── VirtualStainModel has no registered occupant yet ──────────────────────────
# DTR and USIGAN are the intended first users. Until one lands, the classifying
# logic above would go unexercised for this branch, so cover it synthetically.


def test_virtual_stain_is_classified_as_dense_not_tile() -> None:
    class FakeStain(VirtualStainModel):
        stains = ("PAS",)

        def predict(self, image):
            raise NotImplementedError

    assert issubclass(FakeStain, DensePredictionModel)
    assert not issubclass(FakeStain, TilePredictionModel)
    assert _names(FakeStain) == ("PAS",)


def test_virtual_stain_supports_several_stains() -> None:
    class MultiStain(VirtualStainModel):
        stains = ("PAS", "Masson trichrome")

        def predict(self, image):
            raise NotImplementedError

    # C would be 3 * 2 = 6, RGB-major. Not asserted against a tensor here by
    # design — see the module docstring.
    assert _names(MultiStain) == ("PAS", "Masson trichrome")
