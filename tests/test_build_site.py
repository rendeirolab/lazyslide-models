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


@pytest.mark.parametrize(
    ("declared", "family"),
    [
        # "agpl-3.0" contains "gpl", so ordering inside license_family matters.
        ("AGPL-3.0", "AGPL"),
        ("LGPL-2.1", "LGPL"),
        ("GPL-3.0", "GPL"),
        ("Apache 2.0", "Apache 2.0"),
        ("MIT", "MIT"),
        # A dual licence takes the more restrictive half.
        ("Apache 2.0; CC-BY-NC-SA-4.0", "CC BY-NC-SA"),
        ("Owkin non-commercial license", "Custom"),
    ],
)
def test_license_family_is_not_fooled_by_substrings(declared, family):
    assert build_site.license_family(declared) == family
    assert build_site.LICENSE_URLS.get(family) or family == "Custom"


def test_category_tasks_account_for_every_task(payload):
    """The page reads the task off this map, so it must not drop or invent one.

    Two tasks can share a category (cv_feature and tile_prediction both land
    in "Tile prediction"), and the category list is sorted independently of
    the task list — so the pairing has to be recorded, not inferred.
    """
    for model in payload["models"]:
        mapping = model["category_tasks"]
        assert sorted(mapping) == sorted(model["categories"])
        paired = sorted(t for tasks in mapping.values() for t in tasks)
        assert paired == sorted(model["tasks"]), model["name"]


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
