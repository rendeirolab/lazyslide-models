#!/usr/bin/env python3
"""Emit pytest-split group ids for Hugging Face Jobs (issue #20).

This script does not chunk models. It only prints 1-based group ids for
``pytest --splits N --group K``. The full suite is split by *test*, so
slow cases (e.g. cellpose image sizes) can land on different Jobs.

Without a committed ``.test_durations`` file, pytest-split even-splits by
test count. Do not pass ``--splitting-algorithm least_duration`` until
that file exists.

Stdout: JSON object ``{"splits": N, "shards": [{"id": "1", "group": "1"}, ...]}``.
"""

from __future__ import annotations

import argparse
import json
import sys

DEFAULT_SPLITS = 18


def build_groups(splits: int = DEFAULT_SPLITS) -> list[dict[str, str]]:
    if splits < 1:
        raise ValueError("splits must be >= 1")
    return [{"id": str(i), "group": str(i)} for i in range(1, splits + 1)]


def build_plan(splits: int = DEFAULT_SPLITS) -> dict[str, object]:
    return {"splits": splits, "shards": build_groups(splits)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        type=int,
        default=DEFAULT_SPLITS,
        help=f"pytest-split --splits (default: {DEFAULT_SPLITS})",
    )
    args = parser.parse_args(argv)
    json.dump(build_plan(args.splits), sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
