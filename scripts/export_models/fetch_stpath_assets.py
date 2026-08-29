#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28"]
# ///
"""Fetch STPath's gene vocabulary so it can be re-hosted alongside our weights.

STPath's ``symbol2ensembl.json`` is 5.6 MB and lives only in the upstream GitHub
repository, never on Hugging Face. It is too large to vendor in the wheel, so we
mirror it next to the other LazySlide assets and download it at run time -- the
same arrangement Path2Space uses for its gene list.

The file is a factual gene-symbol to Ensembl-ID mapping derived from a public
GRCh38 reference annotation, not the authors' model weights: the weights stay on
``tlhuang/STPath`` and are never redistributed by us.

Run this, then upload with the command printed at the end.
"""

import json
from pathlib import Path

import httpx

SOURCE = (
    "https://raw.githubusercontent.com/Graph-and-Geometric-Learning/"
    "STPath/main/utils_data/symbol2ensembl.json"
)

# Token ids are derived from sorted(set(values)), so the size of the value set
# is load-bearing: a mapping built from a different Ensembl release would shift
# every gene column silently. Must match _vocab.N_UNIQUE_GENES.
EXPECTED_UNIQUE_GENES = 38982
EXPECTED_SYMBOLS = 138560

workdir = Path(__file__).parent
out_dir = workdir / "export_artifacts"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "symbol2ensembl.json"

print(f"Downloading {SOURCE}")
out_file.write_bytes(httpx.get(SOURCE, follow_redirects=True, timeout=120).content)

mapping = json.loads(out_file.read_text(encoding="utf-8"))
n_symbols, n_unique = len(mapping), len(set(mapping.values()))
print(f"  {n_symbols} symbols over {n_unique} unique Ensembl IDs")
if n_unique != EXPECTED_UNIQUE_GENES or n_symbols != EXPECTED_SYMBOLS:
    raise SystemExit(
        f"Unexpected vocabulary size: got {n_symbols}/{n_unique}, "
        f"expected {EXPECTED_SYMBOLS}/{EXPECTED_UNIQUE_GENES}. "
        "Upstream changed the file -- check it against the released stfm.pth "
        "before uploading, the gene columns depend on it."
    )

print(f"\nWrote {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)")
print(
    "\nUpload with:\n"
    "  uv run hf upload RendeiroLab/LazySlide-models \\\n"
    f"    {out_file.relative_to(workdir)} \\\n"
    "    STPath/symbol2ensembl.json"
)
