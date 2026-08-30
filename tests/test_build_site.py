"""The published catalogue must describe the registry it was built from.

The site is generated in CI and deployed without a human looking at it, so
the check that matters is coverage: every registered model reaches the JSON,
carrying the fields the cards render.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from lazyslide_models import MODEL_REGISTRY
from lazyslide_models.base import ModelTask

ROOT = Path(__file__).resolve().parent.parent


def _load_build_site():
    spec = importlib.util.spec_from_file_location(
        "build_site", ROOT / "scripts" / "build_site.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_site = _load_build_site()


@pytest.fixture(scope="module")
def payload(tmp_path_factory):
    out = tmp_path_factory.mktemp("site") / "models.json"
    assert build_site.main(["--out", str(out)]) == 0
    return json.loads(out.read_text())


def test_every_registered_class_is_published(payload):
    published = {m["name"] for m in payload["models"]}
    expected = {cls.__name__ for cls in MODEL_REGISTRY.values()}
    assert published == expected


def test_every_registry_key_is_published(payload):
    published = {key for m in payload["models"] for key in m["keys"]}
    assert published == set(MODEL_REGISTRY)


def test_cards_have_what_they_render(payload):
    for model in payload["models"]:
        assert model["description"], f"{model['name']} has no description"
        assert model["tasks"], f"{model['name']} has no task"
        assert model["api_url"].endswith(f".{model['name']}.html")
        for task in model["tasks"]:
            ModelTask(task)  # raises if the task is not a real registry task


def test_every_model_lands_in_a_display_category(payload):
    known = set(build_site.CATEGORY_ORDER)
    for model in payload["models"]:
        assert model["categories"], f"{model['name']} has no display category"
        assert set(model["categories"]) <= known, model["categories"]


def test_examples_come_only_from_docstrings(payload):
    """No example is invented — a model without one in its docstring has none."""
    from lazyslide_models import MODEL_REGISTRY

    by_name = {cls.__name__: cls for cls in MODEL_REGISTRY.values()}
    for model in payload["models"]:
        example = model["example"]
        if example is None:
            assert build_site.docstring_example(by_name[model["name"]]) is None
            continue
        assert example.strip(), f"{model['name']} has an empty example"
        # Whatever the docstring said, minus the doctest prompts.
        assert ">>>" not in example
        assert not build_site.RST_ROLE.search(example)


def test_descriptions_are_free_of_sphinx_markup(payload):
    for model in payload["models"]:
        text = model["description"]
        assert not build_site.RST_ROLE.search(text), f"{model['name']}: {text}"
        assert "``" not in text


def test_stats_describe_the_published_list(payload):
    models = payload["models"]
    stats = payload["stats"]
    assert stats["n_models"] == len(models)
    assert stats["n_keys"] == sum(len(m["keys"]) for m in models)
    assert stats["n_gated"] == sum(1 for m in models if m["is_gated"])
    assert stats["n_tasks"] == len(stats["by_task"])
    for task, count in stats["by_task"].items():
        assert count == sum(1 for m in models if task in m["tasks"])
    for category, count in stats["by_category"].items():
        assert count == sum(1 for m in models if category in m["categories"])
    # The strip renders by_category in declaration order, including the
    # categories that are declared but not yet filled.
    assert list(stats["by_category"]) == build_site.CATEGORY_ORDER


def test_declared_citations_resolve(payload):
    unresolved = [
        m["name"] for m in payload["models"] if m["bib_key"] and not m["citation"]
    ]
    assert not unresolved, f"bib_key not found in references.bib: {unresolved}"


def test_citations_carry_a_title_and_bibtex(payload):
    for model in payload["models"]:
        citation = model["citation"]
        if citation is None:
            continue
        assert citation["title"], f"{model['name']}: citation has no title"
        assert citation["bibtex"].startswith("@"), f"{model['name']}: bad bibtex"
