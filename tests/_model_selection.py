"""Map changed source files to the model registry keys they affect.

CI uses this so a pull request only tests the models it touches, instead of
downloading and exercising all ~80 registry entries every time.

The mapping is derived from the registry itself — ``inspect.getfile`` on each
registered class — so it cannot drift out of sync with the package layout.

Every rule fails **safe**: anything that cannot be mapped confidently selects
the whole registry. Under-selecting would silently stop testing something,
which is far worse than the cost of an occasional full run.
"""

from __future__ import annotations

import inspect
import subprocess
from functools import cache
from pathlib import Path

import lazyslide_models
from lazyslide_models import MODEL_REGISTRY

#: Root of the model package, as an absolute path.
PACKAGE_ROOT = Path(inspect.getfile(lazyslide_models)).resolve().parent

#: Top-level modules every model depends on. A change here selects everything.
CORE_MODULES = frozenset(
    {
        "base.py",
        "_model_registry.py",
        "_repr.py",
        "_utils.py",
        "_version.py",
        "__init__.py",
    }
)


def _is_irrelevant(path: Path) -> bool:
    """Whether a file outside the package provably cannot affect a weight test.

    Without this, selective testing would never fire: **every** model PR edits
    ``references.bib`` to add its citation, which would otherwise fall through
    to the "outside the package" rule and select the whole registry.

    Each entry needs a reason:

    - ``references.bib`` — read only by ``test_model_attributes``, which is
      never deselected, so its ``bib_key`` check still runs for every model.
    - ``scripts/`` — one-off weight-export scripts, never imported by the suite.
    - ``*.md`` — documentation.

    Everything else outside the package (``tests/``, ``pyproject.toml``,
    ``uv.lock``, workflows) still selects everything.
    """
    parts = path.parts
    return (
        path.name == "references.bib"
        or path.suffix == ".md"
        # No `scripts` directory exists inside the package, so this cannot
        # accidentally swallow a model file.
        or "scripts" in parts
    )


@cache
def _index() -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    """Build ``{file: keys}`` and ``{directory: keys}``, both package-relative."""
    by_file: dict[str, set[str]] = {}
    by_dir: dict[str, set[str]] = {}
    for key, cls in MODEL_REGISTRY.items():
        rel = Path(inspect.getfile(cls)).resolve().relative_to(PACKAGE_ROOT)
        by_file.setdefault(rel.as_posix(), set()).add(key)
        by_dir.setdefault(rel.parent.as_posix(), set()).add(key)
    return (
        {k: frozenset(v) for k, v in by_file.items()},
        {k: frozenset(v) for k, v in by_dir.items()},
    )


def _relative_to_package(raw) -> Path | None:
    """Return *raw* relative to the model package, or ``None`` if it is outside.

    Matches on the package name rather than resolving against the working
    directory, so it behaves the same whether git hands us
    ``src/lazyslide_models/vision/uni.py`` or an absolute path, and whatever
    directory pytest was invoked from.
    """
    parts = Path(raw).parts
    if PACKAGE_ROOT.name not in parts:
        return None
    # Last occurrence, in case the checkout itself lives under a directory of
    # the same name.
    idx = len(parts) - 1 - parts[::-1].index(PACKAGE_ROOT.name)
    tail = parts[idx + 1 :]
    return Path(*tail) if tail else None


def changed_models(paths) -> frozenset[str] | None:
    """Return the registry keys affected by *paths*.

    ``None`` means "no confident mapping — run every model". An empty set means
    the changes provably touch no model.

    Parameters
    ----------
    paths : iterable of str or Path
        Changed files, relative to the repository root or absolute.
    """
    by_file, by_dir = _index()
    selected: set[str] = set()

    for raw in paths:
        if _is_irrelevant(Path(raw)):
            continue

        rel = _relative_to_package(raw)
        if rel is None:
            # tests/, pyproject.toml, workflows — could affect any model.
            return None

        if len(rel.parts) == 1 and rel.name in CORE_MODULES:
            return None

        keys = by_file.get(rel.as_posix())
        if keys is not None:
            selected |= keys
            continue

        # A subpackage re-export shim, e.g. ``vision/__init__.py``: its own
        # directory already holds model files. Adding any model edits one of
        # these, so expanding it to the whole subpackage would select ~36
        # models and defeat the point. Safe to ignore — importing the package
        # builds the registry, so a broken shim fails collection outright.
        if rel.name == "__init__.py" and rel.parent.as_posix() in by_dir:
            continue

        # Shared helpers (``cellvit_family/postprocess.py``, ``classpose/
        # _blocks.py``): walk up to the nearest directory that defines models.
        parent = rel.parent
        while parent.as_posix() not in (".", "") and parent.as_posix() not in by_dir:
            parent = parent.parent
        keys = by_dir.get(parent.as_posix())
        if keys is None:
            return None
        selected |= keys

    return frozenset(selected)


def changed_models_since(ref: str) -> frozenset[str] | None:
    """Return the keys affected by the diff between *ref* and the working tree.

    Anything that makes the diff untrustworthy — git failing, or an empty diff —
    selects everything rather than nothing. CI only runs this job when paths
    under the package changed, so an empty diff there means the base ref is
    wrong or the clone is too shallow, not that there is nothing to test. The
    same guard stops a local run with uncommitted work from silently testing
    nothing.
    """
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", f"{ref}...HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None

    files = [line for line in out.splitlines() if line.strip()]
    if not files:
        return None
    return changed_models(files)
