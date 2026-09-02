"""CI shard groups for pytest-split."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci_model_shards.py"
_spec = importlib.util.spec_from_file_location("ci_model_shards", _SCRIPT)
assert _spec is not None and _spec.loader is not None
_shards = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shards)


def test_groups_are_1_based():
    groups = _shards.build_groups(_shards.DEFAULT_SPLITS)
    assert [g["group"] for g in groups] == [
        str(i) for i in range(1, _shards.DEFAULT_SPLITS + 1)
    ]


def test_plan_embeds_splits_and_shards():
    plan = _shards.build_plan(_shards.DEFAULT_SPLITS)
    assert plan["splits"] == _shards.DEFAULT_SPLITS
    assert plan["shards"] == _shards.build_groups(_shards.DEFAULT_SPLITS)
