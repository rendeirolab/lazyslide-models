# Model zoo site

The searchable catalogue published at
<https://rendeiro.group/lazyslide-models/>.

The organisation has a verified custom domain, so project pages are served
from `rendeiro.group/<repo>/`, not `rendeirolab.github.io/<repo>/` — that host
301s to this one. Use the custom domain in links and in the embed snippet.

| File | What it is |
| --- | --- |
| `index.html` | Standalone shell: masthead, mount point, colophon, web fonts. |
| `embed.js` | The whole browser. Self-mounts into `#lazyslide-models`. |
| `app.css` | Component styles, scoped to `.lsm-root`. |
| `tokens.css` | The design tokens `app.css` imports. |
| `logo.svg` · `favicon.svg` | The LazySlide logo, copied from `assets/logo.svg` in the LazySlide repo — **not** `docs/source/_static/logo.svg`, which is still the old mark. `favicon.svg` is the same file cropped to the illustration so it reads at 16px. |
| `models.json` | **Generated** by `scripts/build_site.py`; not committed. |

The page shows no code it did not get from Python. `scripts/build_site.py`
assigns each model its display category, resolves Sphinx roles in the
description (`:class:`Virchow2 <...>`` renders as `Virchow2`), and lifts the
usage example straight out of the class docstring — an `Examples` section, a
`.. code-block:: python`, or a doctest. **Nothing is generated**: a model with
no example in its docstring shows no example. `embed.js` only renders.

To give a model an example, add one to its docstring; the build picks it up.

No bundler, no dependencies. To work on it:

```bash
uv run task serve-site
```

That rebuilds `models.json` and serves the site at <http://127.0.0.1:8000>.
`uv run task build-site` regenerates the catalogue without serving.

`.github/workflows/pages.yaml` regenerates `models.json` from the live model
registry and deploys on every push to `main` that touches `src/`, `site/`,
`references.bib` or the build script.

> **One-time setup:** the repository's *Settings → Pages → Source* must be set
> to **GitHub Actions** before the first deploy succeeds.

## Embedding it in the LazySlide docs

The same `embed.js` renders inside any host page. In
`docs/source/avail_models.md`, replace the generated model list — the
` ```{eval-rst} ` block containing `.. include:: api/models.rst` — with:

````markdown
```{raw} html
<div id="lazyslide-models"></div>
<noscript><a href="https://rendeiro.group/lazyslide-models/">Browse the model zoo</a></noscript>
<script src="https://rendeiro.group/lazyslide-models/embed.js" defer></script>
```
````

Keep `generate_models_rst()` in `docs/source/conf.py`: it still builds the
per-class API pages that each card's *API reference* link points at.

Embedded, the widget renders in the host page's light DOM, so:

- it adopts the host's body font and paints no background of its own;
- it mirrors `html[data-theme]`, following the docs' own light/dark toggle;
- it never writes to the URL hash unless that hash is already its own, so
  Sphinx anchors survive;
- everything is scoped to `.lsm-root`, so no styles leak in either direction.

GitHub Pages serves `Access-Control-Allow-Origin: *`, which is what lets
`models.json` be fetched from `lazyslide.readthedocs.io`.

## Deep links

- `#lsm-uni` opens that model's panel — the shape to paste into an issue.
- `#lsm:q=segmentation&type=Segmentation&gating=open` restores a search.
