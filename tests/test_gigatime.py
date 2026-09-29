"""GigaTIME loading, tested without the real weights.

A randomly initialised network is saved to a temporary file and loaded back, so
these run on every CI run without touching the gated Hugging Face repo.
"""

from __future__ import annotations

import huggingface_hub
import pytest
import torch

from lazyslide_models.style_transfer.gigatime import (
    GIGATIME_CHANNELS,
    GigaTIME,
    GigaTIMEModel,
)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "model.pth"
    torch.save(GigaTIMEModel(num_classes=len(GIGATIME_CHANNELS)).state_dict(), path)
    return str(path)


def test_model_path_loads_a_local_checkpoint_without_downloading(
    checkpoint, monkeypatch
) -> None:
    """Passing a checkpoint used to be silently ignored in favour of the Hub."""

    def no_download(*args, **kwargs):
        raise AssertionError("model_path was ignored: hf_hub_download was called")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_download)
    model = GigaTIME(model_path=checkpoint)
    out = model.predict(torch.zeros(1, 3, 64, 64))
    assert out.shape == (1, len(GIGATIME_CHANNELS), 64, 64)


def test_token_is_forwarded_to_the_download(checkpoint, monkeypatch) -> None:
    """A token passed to the constructor used to be dropped."""
    seen = {}

    def fake_download(*args, **kwargs):
        seen.update(kwargs)
        return checkpoint

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    GigaTIME(token="hf_test")
    assert seen.get("token") == "hf_test"
