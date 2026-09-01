#!/usr/bin/env python3
"""Split MODEL_REGISTRY keys into Hugging Face Jobs shards (issue #20).

GitHub Actions calls this on ubuntu-latest *without* installing the package,
so keys are read from ``@register(...)`` in ``src/``. A unit test checks
that those keys match ``MODEL_REGISTRY``.

Shards are equal-sized chunks of the sorted key list (default 10). No
model-specific exceptions: if a chunk OOMs ``cpu-upgrade``, lower ``--size``.

Stdout: JSON array of ``{"id": "...", "models": "a,b,c"}``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "lazyslide_models"

DEFAULT_SIZE = 10

_REGISTER = re.compile(r"@register\s*\(", re.MULTILINE)
_KEY = re.compile(
    r'(?<![A-Za-z_])key\s*=\s*(?:"([^"]+)"|\'([^\']+)\'|\[([^\]]+)\])'
)
_LIST_ITEM = re.compile(r'"([^"]+)"|\'([^\']+)\'')


def _call_args(text: str, open_paren: int) -> str:
    """Return the source inside the ``(...)`` that starts at *open_paren*."""
    depth = 0
    for i, ch in enumerate(text[open_paren:], start=open_paren):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren + 1 : i]
    raise ValueError("unclosed @register( in source")


def registry_keys_from_src(src_root: Path = SRC_ROOT) -> list[str]:
    """Keys declared in ``@register`` under *src_root*, sorted unique."""
    found: set[str] = set()
    for path in src_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in _REGISTER.finditer(text):
            args = _call_args(text, match.end() - 1)
            km = _KEY.search(args)
            if km is None:
                continue
            single_dq, single_sq, listed = km.groups()
            if listed is not None:
                for item in _LIST_ITEM.finditer(listed):
                    found.add(next(g for g in item.groups() if g is not None))
            else:
                found.add(single_dq or single_sq)
    return sorted(found)


def build_shards(keys: list[str], *, size: int = DEFAULT_SIZE) -> list[dict[str, str]]:
    """Partition *keys* into consecutive chunks of *size*."""
    if size < 1:
        raise ValueError("size must be >= 1")
    shards: list[dict[str, str]] = []
    for i in range(0, len(keys), size):
        chunk = keys[i : i + size]
        shards.append({"id": f"s{i // size:02d}", "models": ",".join(chunk)})
    return shards


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help="models per shard (default: 10)",
    )
    args = parser.parse_args(argv)
    keys = registry_keys_from_src()
    if not keys:
        print("no @register keys found", file=sys.stderr)
        return 1
    shards = build_shards(keys, size=args.size)
    json.dump(shards, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
