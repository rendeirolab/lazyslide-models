"""CI shard script must cover every registry key (issue #20)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from lazyslide_models import MODEL_REGISTRY

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci_model_shards.py"
_spec = importlib.util.spec_from_file_location("ci_model_shards", _SCRIPT)
assert _spec is not None and _spec.loader is not None
_shards = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shards)


def test_src_scan_matches_the_live_registry():
    """GitHub lists shards without importing the package; that list cannot drift."""
    assert _shards.registry_keys_from_src() == sorted(MODEL_REGISTRY)


def test_shards_partition_the_registry():
    keys = _shards.registry_keys_from_src()
    shards = _shards.build_shards(keys, size=10)
    seen: list[str] = []
    for i, shard in enumerate(shards):
        chunk = shard["models"].split(",")
        assert chunk == keys[i * 10 : i * 10 + 10]
        seen.extend(chunk)
    assert seen == keys
    assert all(len(s["models"].split(",")) == 10 for s in shards[:-1])
