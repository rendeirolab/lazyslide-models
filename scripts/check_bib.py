#!/usr/bin/env python3
"""Structural check for BibTeX files.

Catches the two ways this repo's ``references.bib`` has actually broken:
an entry left unterminated (missing ``}``) and a duplicated citation key.
Deliberately does not validate fields — that is style policing, and
``bib_key`` existence is already asserted in ``tests/test_models.py``.

Stdlib only, so CI can run it with a bare ``python3`` and no install step.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ENTRY_START = re.compile(r"@(\w+)\s*\{\s*([^,\s}]*)")


def check(text: str) -> tuple[int, list[str]]:
    """Return (entry count, errors) for one BibTeX document."""
    errors: list[str] = []
    keys: dict[str, int] = {}
    depth = 0
    entry_line = 0
    entry_key = ""
    line = 1
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\n":
            line += 1
        elif depth == 0:
            if char == "@" and (match := ENTRY_START.match(text, i)):
                entry_line, entry_key = line, match.group(2)
                if entry_key in keys:
                    errors.append(
                        f"line {line}: duplicate key '{entry_key}' "
                        f"(first seen at line {keys[entry_key]})"
                    )
                else:
                    keys[entry_key] = line
                i = text.index("{", i)
                depth = 1
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == "@" and text[i - 1] == "\n":
            # A new entry began while the previous one was still open.
            errors.append(
                f"line {entry_line}: entry '{entry_key}' is not closed "
                f"(next entry starts at line {line})"
            )
            depth = 0
            continue  # re-read this '@' at depth 0
        i += 1
    if depth:
        errors.append(f"line {entry_line}: entry '{entry_key}' is not closed (EOF)")
    return len(keys), errors


def main(argv: list[str]) -> int:
    paths = [Path(p) for p in argv[1:]] or [Path("references.bib")]
    failed = False
    for path in paths:
        count, errors = check(path.read_text(encoding="utf-8"))
        if errors:
            failed = True
            print(f"{path}: {len(errors)} error(s)", file=sys.stderr)
            for error in errors:
                print(f"  {error}", file=sys.stderr)
        else:
            print(f"{path}: {count} entries, OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
