"""NuLite's tile check, tested without the real weights."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

from lazyslide_models.segmentation import NuLite


def test_check_input_tile_warns_only_off_the_supported_mpp() -> None:
    """The mpp check used `or`, so it warned at every mpp, even 0.5 and 0.25."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        NuLite.check_input_tile(SimpleNamespace(mpp=0.5, width=224, height=224))
        NuLite.check_input_tile(SimpleNamespace(mpp=0.25, width=224, height=224))
    with pytest.warns(UserWarning, match="mpp"):
        NuLite.check_input_tile(SimpleNamespace(mpp=1.0, width=224, height=224))
