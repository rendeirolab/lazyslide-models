"""CytoSyn device placement, tested without the gated weights.

Owkin's remote pipeline and scheduler sample in float64, which MPS does not
support, so CytoSyn must stay on CPU when asked for MPS.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from lazyslide_models.image_generation.cytosyn import CytoSyn


class _Pipeline:
    """Stands in for the diffusers pipeline and records where it was moved."""

    device = None

    def to(self, device):
        self.device = torch.device(device)
        return self


def _cytosyn() -> CytoSyn:
    model = CytoSyn.__new__(CytoSyn)  # skip __init__, which downloads the weights
    model.model = _Pipeline()
    return model


@pytest.mark.parametrize("target", ["mps", "mps:0", torch.device("mps")])
def test_mps_runs_on_cpu_with_a_warning(target) -> None:
    model = _cytosyn()
    with pytest.warns(UserWarning, match="float64"):
        assert model.to(target) is model
    assert model.model.device == torch.device("cpu")


@pytest.mark.parametrize("target", ["cpu", "cuda", "cuda:1"])
def test_other_devices_are_kept_without_a_warning(target) -> None:
    model = _cytosyn()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.to(target)
    assert model.model.device == torch.device(target)
