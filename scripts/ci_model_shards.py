#!/usr/bin/env python3
"""Emit pytest-split group ids for Hugging Face Jobs (issue #20).

The full suite is split by *test*, not by model name, so slow cases
(e.g. cellpose image sizes) can land on different Jobs. pytest-split's
``least_duration`` packs by recorded times when ``.test_durations`` exists;
without that file every test is treated as average duration (round-robin).

Stdout: JSON array of ``{"id": "1", "group": "1"}`` (1-based).
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        type=int,
        default=DEFAULT_SPLITS,
        help="pytest-split --splits (default: 18)",
    )
    args = parser.parse_args(argv)
    json.dump(build_groups(args.splits), sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
