#!/usr/bin/env python3
"""Generate ``site/models.json`` — the data behind the published model zoo.

Reads the live :data:`~lazyslide_models.MODEL_REGISTRY` (so the published
catalogue can never drift from what ``list_models()`` returns) and joins it
with the citation metadata in ``references.bib``.

Run from the repository root::

    python scripts/build_site.py [--out site/models.json]
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
import textwrap
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Where the Sphinx autosummary stubs for each task live. Mirrors
#: ``model_sections`` in LazySlide's ``docs/source/conf.py`` — the generated
#: stub filename is ``lazyslide_models.<module>.<ClassName>.rst``, so these
#: must stay in step or the "API reference" links 404.
SECTION_MODULE = {
    "vision": "vision",
    "multimodal": "multimodal",
    "segmentation": "segmentation",
    "tile_prediction": "tile_prediction",
    "feature_prediction": "feature_prediction",
    "cv_feature": "tile_prediction.cv_features",
    "style_transfer": "style_transfer",
    "image_generation": "image_generation",
    # Slide encoders have no module of their own; conf.py documents them under
    # whichever package they happen to live in.
    "slide_encoder": None,
}

API_BASE = "https://lazyslide.readthedocs.io/en/latest/api/_autogen/"

ENTRY_START = re.compile(r"@(\w+)\s*\{\s*([^,\s}]*)")


# -- references.bib --------------------------------------------------------


def iter_bib_entries(text: str):
    """Yield ``(key, raw_entry)`` for every entry in a BibTeX document.

    Deliberately tolerant: stray top-level braces (this file has had them)
    are skipped, because the structural policing lives in ``check_bib.py``.
    """
    i = 0
    while (match := ENTRY_START.search(text, i)) is not None:
        start = text.index("{", match.start())
        depth, j = 0, start
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        yield match.group(2), text[match.start() : j + 1]
        i = j + 1


def _read_value(text: str, i: int) -> tuple[str, int]:
    """Read one field value starting at *i*; return ``(value, next_index)``."""
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text):
        return "", i
    if text[i] == '"':
        j, depth = i + 1, 0
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            elif text[j] == '"' and depth == 0:
                break
            j += 1
        return text[i + 1 : j], j + 1
    if text[i] == "{":
        j, depth = i, 0
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        return text[i + 1 : j], j + 1
    j = i
    while j < len(text) and text[j] not in ",\n}":
        j += 1
    return text[i:j], j


#: BibTeX escapes the TeX specials; a rendered card wants the characters.
LATEX_ESCAPES = {
    r"\&": "&",
    r"\%": "%",
    r"\$": "$",
    r"\#": "#",
    r"\_": "_",
    r"\{": "{",
    r"\}": "}",
}


def _clean(value: str) -> str:
    """Collapse BibTeX wrapping, casing braces and escapes into plain text."""
    text = value.replace("{", "").replace("}", "")
    for escaped, plain in LATEX_ESCAPES.items():
        text = text.replace(escaped, plain)
    return re.sub(r"\s+", " ", text).strip()


def short_author(author: str | None) -> str | None:
    """``"Chen, Richard J and Ding, Tong and ..."`` -> ``"Chen et al."``.

    A twenty-name author list buries the one thing a card needs to show.
    The full list stays available in the copyable BibTeX.
    """
    if not author:
        return None
    names = [n.split(",")[0].strip() for n in author.split(" and ") if n.strip()]
    if not names:
        return None
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{names[0]} et al."


def parse_bib(path: Path) -> dict[str, dict]:
    """Map citation key -> ``{title, author, venue, year, doi, url, bibtex}``."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    entries: dict[str, dict] = {}
    for key, raw in iter_bib_entries(text):
        fields: dict[str, str] = {}
        # Skip past "@TYPE{key," before reading fields, so the key is not
        # mistaken for a value.
        body = raw[raw.index(",") + 1 :] if "," in raw else ""
        for match in re.finditer(r"(\w+)\s*=", body):
            name = match.group(1).lower()
            if name in fields:
                continue
            value, _ = _read_value(body, match.end())
            fields[name] = _clean(value)
        entries[key] = {
            "title": fields.get("title"),
            "author": short_author(fields.get("author")),
            "venue": fields.get("journal") or fields.get("booktitle"),
            "year": fields.get("year"),
            "doi": fields.get("doi"),
            "url": fields.get("url"),
            "bibtex": raw.strip(),
        }
    return entries


# -- registry --------------------------------------------------------------


def license_family(text: str | None) -> str | None:
    """Collapse 24 free-text licence strings into a facet a reader can use.

    The most restrictive clause wins for dual licences: someone filtering on
    "Apache 2.0" must not be handed a model that is also CC BY-NC-SA.
    """
    if not text:
        return None
    t = text.lower().replace(" ", "-")
    for needle, family in (
        ("cc-by-nc-nd", "CC BY-NC-ND"),
        ("cc-by-nc-sa", "CC BY-NC-SA"),
        ("cc-by-nc", "CC BY-NC"),
        # Before "gpl", which is a substring of it.
        ("agpl", "AGPL"),
        ("lgpl", "LGPL"),
        ("gpl", "GPL"),
        ("apache", "Apache 2.0"),
        ("bsd", "BSD"),
        ("mit", "MIT"),
    ):
        if needle in t:
            return family
    return "Custom"


#: The reader-facing taxonomy. ``ModelTask`` stays the API's vocabulary —
#: ``list_models("vision")`` still works — but "vision" and "cv_feature" are
#: not how someone picks a model, so the site groups them.
CATEGORY = {
    "vision": "Image",
    "multimodal": "Text-Image",
    "slide_encoder": "Slide encoder",
    "segmentation": "Segmentation",
    # Classical CV features run through the same ``zs.tl.tile_prediction``
    # entry point as the learned tile models.
    "tile_prediction": "Tile prediction",
    "cv_feature": "Tile prediction",
    # Both predict molecular readouts from H&E: one as expression tables,
    # one as virtual stains.
    "feature_prediction": "H&E → Spatial omics / IHC",
    "style_transfer": "H&E → Spatial omics / IHC",
    "image_generation": "Image generation",
}

#: Display order. "Chat" is declared but not yet populated — slide-conditioned
#: text generation exists as ``TextResponseModelProtocol`` but no model
#: registers it as its task.
CATEGORY_ORDER = [
    "Image",
    "Text-Image",
    "Slide encoder",
    "Segmentation",
    "Tile prediction",
    "H&E → Spatial omics / IHC",
    "Image generation",
    "Chat",
]

#: Canonical licence texts, so a card can always link somewhere for the
#: licences that carry a standard document. Bespoke licences keep whatever
#: ``license_url`` the model declared, or none.
LICENSE_URLS = {
    "MIT": "https://opensource.org/license/mit",
    "Apache 2.0": "https://www.apache.org/licenses/LICENSE-2.0",
    "BSD": "https://opensource.org/license/bsd-3-clause",
    "AGPL": "https://www.gnu.org/licenses/agpl-3.0.html",
    "LGPL": "https://www.gnu.org/licenses/lgpl-3.0.html",
    "GPL": "https://www.gnu.org/licenses/gpl-3.0.html",
    "CC BY-NC": "https://creativecommons.org/licenses/by-nc/4.0/",
    "CC BY-NC-SA": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "CC BY-NC-ND": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
}

#: Present on nearly every model and never interesting on a card.
BOILERPLATE_PARAMS = {"model_path", "token"}


def clean_annotation(annotation: object) -> str | None:
    """Render a type annotation the way the source wrote it."""
    if annotation is inspect.Parameter.empty:
        return None
    text = (
        annotation
        if isinstance(annotation, str)
        else inspect.formatannotation(annotation)
    )
    text = text.strip()
    # ``from __future__ import annotations`` hands these over already quoted.
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return re.sub(r"\s+", " ", text) or None


#: ``:class:`Virchow2 <lazyslide.models.vision.Virchow2>``` and friends. The
#: docstrings are written for Sphinx; the site has to show what Sphinx would.
RST_ROLE = re.compile(r":[\w:+.-]+:`([^`]+)`")


def strip_rst(text: str | None) -> str | None:
    """Render Sphinx roles and literals as the text a reader should see."""
    if not text:
        return text

    def role(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        # "Title <target>" keeps the title; the target is only for Sphinx.
        if body.endswith(">") and "<" in body:
            body = body[: body.rfind("<")].strip()
        body = body.lstrip("~.")
        # A bare dotted target shows its last component: the class name.
        if "." in body and " " not in body:
            body = body.rsplit(".", 1)[-1]
        return body

    text = RST_ROLE.sub(role, text)
    text = re.sub(r"``([^`]+)``", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _block_after(lines: list[str], start: int, indent: int) -> str | None:
    """Collect the indented body that follows *start*, RST-directive style."""
    body: list[str] = []
    for line in lines[start:]:
        if not line.strip():
            body.append("")
            continue
        if len(line) - len(line.lstrip()) <= indent:
            break
        body.append(line)
    text = textwrap.dedent("\n".join(body)).strip("\n")
    # Doctest prompts read as a transcript; a card wants pasteable code.
    text = re.sub(r"^(>>>|\.\.\.) ?", "", text, flags=re.MULTILINE)
    return text.strip() or None


def docstring_example(cls) -> str | None:
    """Pull a usage example out of the class docstring, if it has one.

    Recognises a numpydoc ``Examples`` section, an RST
    ``.. code-block:: python`` directive, or a bare doctest — whichever the
    model happens to use. The docstring is the source of truth; the
    generated snippet below is only the fallback.
    """
    lines = (cls.__doc__ or "").expandtabs().splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if (
            re.fullmatch(r"Examples?", stripped)
            and i + 1 < len(lines)
            and re.fullmatch(r"-{3,}", lines[i + 1].strip())
            and (found := _block_after(lines, i + 2, indent - 1))
        ):
            return found
        if re.fullmatch(r"\.\.\s+code-block::\s*python", stripped) and (
            found := _block_after(lines, i + 1, indent)
        ):
            return found
    doctest = [ln for ln in lines if ln.strip().startswith((">>>", "..."))]
    if doctest:
        return re.sub(
            r"^\s*(>>>|\.\.\.) ?", "", "\n".join(doctest), flags=re.MULTILINE
        ).strip()
    return None


def init_params(cls) -> list[dict]:
    """The knobs a caller can actually turn, minus the universal plumbing."""
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return []
    params = []
    for name, param in list(signature.parameters.items())[1:]:
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name in BOILERPLATE_PARAMS:
            continue
        params.append(
            {
                "name": name,
                "annotation": clean_annotation(param.annotation),
                "default": (
                    None
                    if param.default is inspect.Parameter.empty
                    else repr(param.default)
                ),
                "required": param.default is inspect.Parameter.empty,
            }
        )
    return params


def api_url(cls, tasks: list[str]) -> str:
    """The Sphinx autosummary page LazySlide generates for *cls*."""
    for task in tasks:
        module = SECTION_MODULE.get(task)
        if module is not None:
            return f"{API_BASE}lazyslide_models.{module}.{cls.__name__}.html"
    # Slide-encoder-only models: documented under their own package.
    package = cls.__module__.split(".")[1]
    return f"{API_BASE}lazyslide_models.{package}.{cls.__name__}.html"


def collect_models(citations: dict[str, dict]) -> list[dict]:
    from lazyslide_models import MODEL_REGISTRY
    from lazyslide_models.base import ModelTask

    records: dict[type, dict] = {}
    for key, cls in MODEL_REGISTRY.items():
        if cls in records:
            records[cls]["keys"].append(key)
            continue
        task = cls.task if isinstance(cls.task, list) else [cls.task]
        tasks = [t.value if isinstance(t, ModelTask) else str(t) for t in task]

        constraint = getattr(cls, "input_constraint", None)
        bib_key = getattr(cls, "bib_key", None)
        params = init_params(cls)
        category_tasks: dict[str, list[str]] = {}
        for t in tasks:
            category = CATEGORY.get(t)
            if category:
                category_tasks.setdefault(category, []).append(t)
        categories = sorted(category_tasks, key=CATEGORY_ORDER.index)
        # No example unless the docstring carries one: a fabricated snippet
        # is a guess about an API this script cannot verify.
        example = docstring_example(cls)
        records[cls] = {
            "keys": [key],
            "name": cls.__name__,
            "module": cls.__module__,
            "tasks": tasks,
            "categories": categories,
            # category -> the ModelTask values that put it there, so the page
            # never has to guess which task a category came from.
            "category_tasks": category_tasks,
            "example": example,
            "is_gated": bool(getattr(cls, "is_gated", False)),
            "commercial": getattr(cls, "commercial", None),
            "license": getattr(cls, "license", None),
            "license_url": getattr(cls, "license_url", None),
            "description": strip_rst(getattr(cls, "description", None)),
            "param_size": getattr(cls, "param_size", None),
            "encode_dim": getattr(cls, "encode_dim", None),
            "flops": getattr(cls, "flops", None),
            "vision_encoder": getattr(cls, "vision_encoder", None),
            "input_constraint": (
                {
                    "min": constraint.min,
                    "max": constraint.max,
                    "divisible_by": constraint.divisible_by,
                }
                if constraint is not None
                else None
            ),
            "classes": list(getattr(cls, "classes", None) or []) or None,
            "params": params,
            "hf_url": getattr(cls, "hf_url", None),
            "github_url": getattr(cls, "github_url", None),
            "paper_url": getattr(cls, "paper_url", None),
            "bib_key": bib_key,
            "citation": citations.get(bib_key) if bib_key else None,
            "api_url": api_url(cls, tasks),
        }

    models = list(records.values())
    # A license may be declared per-variant; normalise to a display string.
    for model in models:
        if isinstance(model["license"], list):
            model["license"] = "; ".join(model["license"])
        if isinstance(model["license_url"], list):
            model["license_url"] = model["license_url"][0]
        model["license_family"] = license_family(model["license"])
        if not model["license_url"]:
            model["license_url"] = LICENSE_URLS.get(model["license_family"])
    models.sort(key=lambda m: m["keys"][0].lower())
    return models


def build_stats(models: list[dict]) -> dict:
    by_task: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    for model in models:
        by_task.update(model["tasks"])
        by_category.update(model["categories"])
    return {
        "n_models": len(models),
        "n_keys": sum(len(m["keys"]) for m in models),
        "n_tasks": len(by_task),
        "n_gated": sum(1 for m in models if m["is_gated"]),
        "n_commercial": sum(1 for m in models if m["commercial"]),
        # Categories overlap — a model can be both Text-Image and a slide
        # encoder — so these deliberately sum to more than n_models. Chat is
        # listed at zero because the category is declared, not yet filled.
        "by_category": {name: by_category.get(name, 0) for name in CATEGORY_ORDER},
        "by_task": dict(by_task.most_common()),
        "by_license": dict(
            Counter(
                m["license_family"] for m in models if m["license_family"]
            ).most_common()
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "site" / "models.json")
    parser.add_argument("--bib", type=Path, default=ROOT / "references.bib")
    args = parser.parse_args(argv)

    from lazyslide_models import __version__

    citations = parse_bib(args.bib)
    models = collect_models(citations)

    uncategorised = sorted({t for m in models for t in m["tasks"] if t not in CATEGORY})
    if uncategorised:
        print(
            f"error: no display category for task(s): {', '.join(uncategorised)} "
            "— add them to CATEGORY and CATEGORY_ORDER",
            file=sys.stderr,
        )
        return 1

    leaked = [
        m["name"]
        for m in models
        if RST_ROLE.search(m["description"] or "") or "``" in (m["description"] or "")
    ]
    if leaked:
        print(
            f"error: raw Sphinx markup survived into: {', '.join(leaked)} "
            "— extend strip_rst()",
            file=sys.stderr,
        )
        return 1

    dangling = [
        f"{m['name']} ({m['bib_key']})"
        for m in models
        if m["bib_key"] and not m["citation"]
    ]
    if dangling:
        print(
            f"error: bib_key not found in {args.bib.name}: {', '.join(dangling)}",
            file=sys.stderr,
        )
        return 1

    missing = [m["name"] for m in models if not m["description"]]
    if not models:
        print("error: the model registry is empty", file=sys.stderr)
        return 1
    if missing:
        print(
            f"error: {len(missing)} model(s) have no description: "
            f"{', '.join(sorted(missing))}",
            file=sys.stderr,
        )
        return 1

    # 0.0.4.post4.dev0+ec2942d is honest but unreadable in a header; keep the
    # exact string for the colophon and show the release it descends from.
    short = re.match(r"\d+\.\d+\.\d+", __version__)
    payload = {
        "version": short.group(0) if short else __version__,
        "version_full": __version__,
        "generated_at": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stats": build_stats(models),
        "models": models,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    stats = payload["stats"]
    out = args.out
    print(
        f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}: "
        f"{stats['n_models']} models, {stats['n_keys']} keys, "
        f"{stats['n_tasks']} tasks, "
        f"{sum(1 for m in models if m['citation'])} with citations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
