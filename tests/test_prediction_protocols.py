"""Tests for the three prediction model classes.

None of these load weights. They read class attributes only, so the whole file
runs for every model on a selective CI run. The tile-level ``cv_feature`` models
are pure OpenCV/NumPy, so their declared ``columns`` are checked against a real
``predict`` call rather than trusted.

There is deliberately no check that a model's tensor actually has as many
channels as it declares. That was a design decision: output shapes are not
validated anywhere, at runtime or in tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from skimage.color import combine_stains, rgb_from_hdx, rgb_from_hed, separate_stains

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

#: The tile-level ones: a dense cv_feature returns an image, not columns.
CV_FEATURE_MODELS = sorted(
    key
    for key, cls in MODEL_REGISTRY.items()
    if getattr(cls, "task", None) is ModelTask.cv_feature
    and issubclass(cls, TilePredictionModel)
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


# ── output_range ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_name", PREDICTION_MODELS)
def test_output_range_is_a_low_high_pair(model_name: str) -> None:
    """When declared, the range has to be usable for choosing a storage dtype."""
    cls = MODEL_REGISTRY[model_name]
    if not issubclass(cls, DensePredictionModel):
        return
    rng = cls.output_range
    if rng is None:
        return  # unbounded, which is allowed
    assert isinstance(rng, tuple) and len(rng) == 2, (
        f"{model_name}: output_range must be a (low, high) tuple, got {rng!r}"
    )
    low, high = rng
    assert isinstance(low, float) and isinstance(high, float), (
        f"{model_name}: output_range entries must be floats, got {rng!r}"
    )
    assert low < high, f"{model_name}: output_range low must be below high, got {rng!r}"


def test_output_range_defaults_to_unbounded() -> None:
    """A model that says nothing about its range is treated as unbounded."""

    class Unbounded(MarkerMapModel):
        channel_names = ("CD3",)

        def predict(self, image):
            raise NotImplementedError

    assert Unbounded.output_range is None


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


@pytest.mark.parametrize("model_name", CV_FEATURE_MODELS)
def test_cv_feature_rejects_float_tiles(model_name: str) -> None:
    """cv2 and the uint8 casts would read a float tile as garbage, silently."""
    model = MODEL_REGISTRY[model_name]()
    with pytest.raises(TypeError, match="uint8"):
        model.predict(np.zeros((1, 8, 8, 3), np.float32))


def test_cv_features_reject_an_ambiguous_layout() -> None:
    """A ``(B, 3, H, 3)`` batch is as much BCHW as BHWC, so neither is guessed."""
    from lazyslide_models.tile_prediction import Brightness

    with pytest.raises(ValueError, match="BCHW"):
        Brightness().predict(np.zeros((2, 3, 8, 3), np.uint8))
    # Unambiguous BCHW still works
    assert Brightness().predict(np.zeros((2, 3, 8, 8), np.uint8))[
        "brightness"
    ].shape == (2,)


@pytest.mark.parametrize(
    ("stain", "rgb_from_stain", "target"),
    [
        ("hed", rgb_from_hed, "dab"),
        ("hdx", rgb_from_hdx, "dab"),
        # Names and their order come from the dict, so they need not be H, E, DAB
        (
            {
                "brown": rgb_from_hed[2],
                "blue": rgb_from_hed[0],
                "pink": rgb_from_hed[1],
            },
            rgb_from_hed[[2, 0, 1]],
            "brown",
        ),
    ],
    # Not bare stain names: conftest reads a bracketed id as a model name
    ids=["preset-hed", "preset-hdx", "custom-dict"],
)
def test_color_deconvolution_recovers_a_known_stain(
    stain, rgb_from_stain, target
) -> None:
    """A tile of one stain at a known amount gives that amount back, and the
    tile model, the dense model and scikit-image agree on it."""
    from lazyslide_models.tile_prediction import (
        ColorDeconvolution,
        ColorDeconvolutionMap,
    )

    model, dense = ColorDeconvolution(stain), ColorDeconvolutionMap(stain)
    assert dense.channel_names == model.columns

    amounts = np.array([0.03, 0.08])
    stains = np.zeros((2, 32, 32, 3))
    stains[..., model.columns.index(target)] = amounts[:, None, None]
    tiles = np.round(combine_stains(stains, rgb_from_stain) * 255).astype(np.uint8)

    out = model.predict(tiles)
    for name, got in out.items():
        # 8-bit pixels are all the round trip loses
        want = amounts if name == target else 0
        np.testing.assert_allclose(got, want, atol=2e-3, err_msg=name)

    ref = separate_stains(tiles, np.linalg.inv(rgb_from_stain)).mean(axis=(1, 2))
    np.testing.assert_allclose(np.stack(list(out.values()), axis=1), ref, atol=1e-9)

    per_pixel = dense.predict(torch.stack([dense.get_transform()(t) for t in tiles]))
    np.testing.assert_allclose(per_pixel.mean(dim=(2, 3)).numpy(), ref, atol=1e-6)


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

    # C would be 3 * 2 = 6, RGB-major. Not asserted against a tensor here, by
    # design. See the module docstring.
    assert _names(MultiStain) == ("PAS", "Masson trichrome")
