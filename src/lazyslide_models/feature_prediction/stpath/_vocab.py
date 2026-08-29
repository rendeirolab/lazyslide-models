"""Gene, organ and technology vocabularies for STPath.

The gene vocabulary is a 138,560-entry gene-symbol to Ensembl-ID mapping over
38,982 unique IDs. It is 5.6 MB, so it is hosted alongside the other LazySlide
model assets rather than shipped in the wheel — the same arrangement
:class:`Path2Space <lazyslide_models.feature_prediction.Path2Space>` uses for
its gene list.

Token indices are derived from ``sorted(set(symbol -> ensembl))``, so the file
is load-bearing: a mapping built from a different Ensembl release would shift
every gene column silently. :func:`load_gene_vocab` therefore checks the size of
the value set before returning.
"""

from __future__ import annotations

import json

#: Where the gene vocabulary is hosted.
ASSET_REPO = "RendeiroLab/LazySlide-models"
ASSET_FILE = "STPath/symbol2ensembl.json"

#: Unique Ensembl IDs in the released vocabulary. Two extra token ids are
#: reserved (0 = pad, 1 = mask), giving the 38,984 output columns of ``stfm.pth``.
N_UNIQUE_GENES = 38982
N_GENE_TOKENS = N_UNIQUE_GENES + 2


class GeneVocab:
    """Symbol to Ensembl-ID mapping plus the token ids STPath was trained with."""

    def __init__(self, symbol2ensembl: dict[str, str]):
        self.symbol2ensembl = symbol2ensembl
        ordered = sorted(set(symbol2ensembl.values()))
        # 0 and 1 are reserved for the pad and mask tokens.
        self.gene2id = {gene: i + 2 for i, gene in enumerate(ordered)}
        self.gene_ids = ordered

    @property
    def n_tokens(self) -> int:
        return len(self.gene_ids) + 2

    def symbol2id(self, symbols) -> tuple[list[int], list[int]]:
        """Map gene symbols to output column indices.

        Returns the ids together with the positions of the symbols that were
        found, so callers can report what was dropped.
        """
        ids, valid = [], []
        for i, symbol in enumerate(symbols):
            ensembl = self.symbol2ensembl.get(symbol)
            if ensembl is None:
                continue
            token = self.gene2id.get(ensembl)
            if token is None:
                continue
            ids.append(token)
            valid.append(i)
        return ids, valid


def load_gene_vocab(path: str | None = None, token: str | None = None) -> GeneVocab:
    """Load the gene vocabulary, downloading it from Hugging Face if needed."""
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(ASSET_REPO, ASSET_FILE, token=token)

    with open(path, encoding="utf-8") as handle:
        symbol2ensembl = json.load(handle)

    vocab = GeneVocab(symbol2ensembl)
    if len(vocab.gene_ids) != N_UNIQUE_GENES:
        raise ValueError(
            f"STPath gene vocabulary has {len(vocab.gene_ids)} unique Ensembl IDs, "
            f"expected {N_UNIQUE_GENES}. Gene token indices are derived from this "
            "set, so a mismatched vocabulary would silently mislabel every "
            "predicted gene. Check that the correct file was downloaded."
        )
    return vocab
