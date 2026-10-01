"""Weight-free checks for the helpers in ``lazyslide_models._utils``."""

from __future__ import annotations

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError

from lazyslide_models._utils import hf_access


def test_hf_access_reraises_gated_repo_error_with_help() -> None:
    """On huggingface_hub 1.x the re-raise used to fail with a TypeError."""
    request = httpx.Request("GET", "https://huggingface.co/x/y")
    response = httpx.Response(403, request=request)
    with pytest.raises(GatedRepoError, match="request access"), hf_access("x/y"):
        raise GatedRepoError("403 Client Error.", response=response)
