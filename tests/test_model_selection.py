"""Tests for the changed-file → model-key mapping used by CI.

Expectations are derived from ``MODEL_REGISTRY`` rather than hard-coded model
names, so adding or removing a model does not break these tests.

The failure mode that matters here is *under*-selection: a mapping bug would
quietly stop testing a model and nobody would notice. Every rule is therefore
checked in the direction of "does it still select enough".
"""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import pytest
from _model_selection import (
    CORE_MODULES,
    PACKAGE_ROOT,
    _index,
    changed_models,
)

from lazyslide_models import MODEL_REGISTRY

PKG = f"src/{PACKAGE_ROOT.name}"


def _file_for(key: str) -> str:
    rel = Path(inspect.getfile(MODEL_REGISTRY[key])).resolve().relative_to(PACKAGE_ROOT)
    return f"{PKG}/{rel.as_posix()}"


def _a_subpackage() -> str:
    """A top-level subpackage that holds model files, e.g. ``vision``."""
    _, by_dir = _index()
    return next(d for d in by_dir if "/" not in d and d != ".")


# ── fail-safe rules ───────────────────────────────────────────────────────────


def test_no_changes_selects_nothing():
    assert changed_models([]) == frozenset()


@pytest.mark.parametrize("core", sorted(CORE_MODULES))
def test_core_module_selects_everything(core):
    """A change to shared machinery cannot be narrowed down."""
    assert changed_models([f"{PKG}/{core}"]) is None


@pytest.mark.parametrize(
    "path",
    [
        "tests/inputs.py",
        "pyproject.toml",
        "uv.lock",
    ],
    ids=["tests", "packaging", "lockfile"],
)
def test_paths_outside_the_package_select_everything(path):
    assert changed_models([path]) is None


def test_unmappable_path_inside_the_package_selects_everything():
    assert changed_models([f"{PKG}/some_new_subsystem/thing.py"]) is None


def test_unregistered_file_falls_back_to_its_subpackage():
    """A model file the registry does not know about must not select nothing.

    This is what a half-finished PR looks like — the module exists but is not
    imported in the subpackage ``__init__``, so it never registered.
    """
    _, by_dir = _index()
    subpackage = _a_subpackage()
    assert (
        changed_models([f"{PKG}/{subpackage}/brand_new_model.py"])
        == (by_dir[subpackage])
    )


# ── the narrowing rules ───────────────────────────────────────────────────────


@pytest.mark.parametrize("key", sorted(MODEL_REGISTRY))
def test_every_model_file_selects_its_own_key(key):
    """The file defining a model must always select that model."""
    selected = changed_models([_file_for(key)])
    assert selected is not None, f"{key}: unexpectedly fell back to a full run"
    assert key in selected


def test_multi_key_file_selects_all_its_keys():
    """One file can register several keys; changing it must select them all."""
    by_file, _ = _index()
    path, keys = max(by_file.items(), key=lambda kv: len(kv[1]))
    assert len(keys) > 1, "expected at least one file with multiple keys"
    assert changed_models([f"{PKG}/{path}"]) == keys


def test_subpackage_shim_selects_nothing():
    """``vision/__init__.py`` is a re-export shim.

    Adding any model edits one of these, so expanding it to the subpackage
    would select every model in it and defeat selective testing. A broken shim
    fails collection outright, so ignoring it is safe.
    """
    assert changed_models([f"{PKG}/{_a_subpackage()}/__init__.py"]) == frozenset()


def test_shared_helper_selects_the_whole_model_directory():
    """A helper next to model files affects every model in that directory."""
    _, by_dir = _index()
    # A nested directory, e.g. segmentation/cellvit_family — its helpers are
    # shared by exactly the models defined there.
    nested = next(d for d in by_dir if "/" in d and len(by_dir[d]) > 1)
    assert changed_models([f"{PKG}/{nested}/_shared_helper.py"]) == by_dir[nested]


@pytest.mark.parametrize(
    "path",
    [
        "references.bib",
        "README.md",
        "scripts/export_models/export_grandqc.py",
        ".github/workflows/test_models.yml",
        "tests/test_ci_model_shards.py",
    ],
    ids=["citations", "docs", "export-script", "workflow", "ci-shard-helper"],
)
def test_irrelevant_paths_select_nothing(path):
    """These cannot affect a weight test, so they must not force a full run."""
    assert changed_models([path]) == frozenset()


def test_adding_one_model_selects_only_that_model():
    """The whole point: a model PR touches its file, the shim and references.bib.

    Modelled on the real mSTAR commit (55b25a0). Without the ``references.bib``
    exemption this would select the entire registry, since every model PR adds
    a citation.
    """
    by_file, _ = _index()
    subpackage = _a_subpackage()
    # A file registering exactly one key, so the expectation is unambiguous.
    path, keys = next(
        (p, k)
        for p, k in by_file.items()
        if len(k) == 1 and p.startswith(f"{subpackage}/")
    )
    selected = changed_models(
        ["references.bib", f"{PKG}/{subpackage}/__init__.py", f"{PKG}/{path}"]
    )
    assert selected == keys


# ── path handling ─────────────────────────────────────────────────────────────


def test_absolute_paths_are_accepted():
    key = next(iter(MODEL_REGISTRY))
    absolute = Path(inspect.getfile(MODEL_REGISTRY[key])).resolve()
    assert key in changed_models([str(absolute)])


def test_selection_is_independent_of_the_working_directory(monkeypatch, tmp_path):
    """Paths are matched by package name, not resolved against the CWD."""
    key = next(iter(MODEL_REGISTRY))
    monkeypatch.chdir(tmp_path)
    assert key in changed_models([_file_for(key)])


# ── the git wrapper ───────────────────────────────────────────────────────────


def test_empty_diff_selects_everything(monkeypatch):
    """An empty diff must fail safe, not silently test nothing.

    CI only reaches this code when files under the package changed, so an empty
    diff means a wrong base ref or a too-shallow clone.
    """
    import _model_selection as sel

    monkeypatch.setattr(
        sel.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="", stderr=""),
    )
    assert sel.changed_models_since("main") is None


def test_git_failure_selects_everything(monkeypatch):
    import _model_selection as sel

    def boom(*a, **k):
        raise OSError("git not found")

    monkeypatch.setattr(sel.subprocess, "run", boom)
    assert sel.changed_models_since("main") is None
