"""STPath: spatial gene-expression prediction from tile features."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import torch

from lazyslide_models._model_registry import register
from lazyslide_models._utils import find_stack_level, hf_access
from lazyslide_models.base import FeaturePredictionModel, ModelTask

from ._constants import N_ORGANS, N_TECH, ORGAN_ALIGN, ORGAN_VOC, TECH_ALIGN, TECH_VOC
from ._model import STFM, rescale_coords
from ._vocab import load_gene_vocab

__all__ = ["STPath"]

WEIGHTS_REPO = "tlhuang/STPath"
WEIGHTS_FILE = "stfm.pth"

#: Above this many spots the O(n^2) spatial attention stops being practical.
#: Measured peak RSS on CPU: 1k ~1 GB, 2k ~2 GB, 4k ~4.7 GB, 6k ~9.6 GB.
DEFAULT_MAX_SPOTS = 8000


def _encode_organ(organ_type: str | None) -> int:
    if organ_type is None:
        # Upstream passes "Others" through an aligner that has no "Others" key,
        # so it lands on <pad>. Replicated deliberately: matching the reference
        # implementation matters more than the tidier reading of the intent.
        return ORGAN_VOC.index("<pad>")
    name = ORGAN_ALIGN.get(organ_type, organ_type)
    if name not in ORGAN_VOC:
        options = sorted(v for v in ORGAN_VOC if not v.startswith("<"))
        raise ValueError(
            f"Unknown organ '{organ_type}'. Choose one of {options}, "
            f"or an alias thereof, or leave it as None."
        )
    return ORGAN_VOC.index(name)


def _encode_tech(tech_type: str | None) -> int:
    if tech_type is None:
        return TECH_VOC.index("<pad>")
    name = TECH_ALIGN.get(tech_type, tech_type)
    if name not in TECH_VOC:
        options = sorted(v for v in TECH_VOC if not v.startswith("<"))
        raise ValueError(f"Unknown technology '{tech_type}'. Choose one of {options}.")
    return TECH_VOC.index(name)


@register(
    key="stpath",
    task=ModelTask.feature_prediction,
    is_gated=False,
    # Neither the GitHub repository nor the Hugging Face model declares a
    # licence, which leaves the weights all-rights-reserved. They are fetched
    # from the authors' own repo at run time and never redistributed by us.
    license="Unlicensed",
    license_url="https://github.com/Graph-and-Geometric-Learning/STPath",
    commercial=False,
    description=(
        "Predict spatial gene expression from GigaPath tile features with a "
        "spatially aware transformer."
    ),
    github_url="https://github.com/Graph-and-Geometric-Learning/STPath",
    hf_url="https://huggingface.co/tlhuang/STPath",
    paper_url="https://doi.org/10.1038/s41746-025-02020-3",
    bib_key="Huang2025-stp",
    param_size="49.2M",
    vision_encoder="gigapath",
)
class STPath(FeaturePredictionModel):
    """STPath spatial transcriptomics predictor.

    Consumes 1536-dimensional GigaPath tile features together with tile
    coordinates and predicts log1p-transformed expression for 38,982 genes,
    keyed by Ensembl gene ID.

    Unlike :class:`Path2Space <lazyslide_models.feature_prediction.Path2Space>`,
    predictions are **not** independent per tile: a spatial transformer attends
    over every spot at once, so the model must see the whole slide in one call.
    The runner handles that through ``needs_coords`` and ``whole_slide``.

    .. code-block:: python

        >>> import lazyslide as zs
        >>> zs.tl.feature_extraction(wsi, "gigapath")
        >>> zs.tl.feature_prediction(
        ...     wsi, STPath(organ_type="Kidney", tech_type="Visium",
        ...                 genes=["GATA3", "UBE2C"])
        ... )

    .. note::
        Attention cost is quadratic in the number of spots, and every tile is
        seen in one call — there is no batching, so ``max_spots`` bounds the
        whole slide, not a chunk of it. Past it ``predict`` raises rather than
        thrashing, before producing any output.

        Measured peak memory is roughly 1 GB at 1,000 spots, 4.7 GB at 4,000 and
        9.6 GB at 6,000. STPath is a spot-level model — a Visium section is a
        few thousand spots at 100 µm pitch — so a whole slide at that resolution
        does not fit: a typical slide tiled at 128 µm is around 12,000 tiles.
        Use it on a region of interest, or tile at roughly 256 µm (512 px at
        mpp 0.5, or 256 px at mpp 1.0), accepting that this is coarser than the
        spacing the model was trained on.

    .. note::
        In-context conditioning on observed expression, which upstream exposes
        through ``context_gene_exps``, is not wired up here; every spot is
        predicted from image features alone.

    Parameters
    ----------
    organ_type : str, optional
        Organ the section comes from, e.g. ``"Kidney"``. Aliases from STPath's
        organ vocabulary are accepted. When omitted the organ token is padded,
        matching the reference implementation.
    tech_type : str, optional
        Spatial technology, one of ``"Spatial Transcriptomics"``, ``"Visium"``,
        ``"Xenium"``, ``"Visium HD"``. Padded when omitted.
    genes : sequence of str, optional
        Restrict the output to these gene symbols. Strongly recommended: the
        full panel is 38,982 columns, which is slow to assemble and wide to
        store. When omitted every gene is returned, keyed by Ensembl ID.
    model_path : str, optional
        Local ``stfm.pth`` to load instead of downloading.
    gene_voc_path : str, optional
        Local ``symbol2ensembl.json`` to load instead of downloading.
    token : str, optional
        Hugging Face access token.
    max_spots : int, default: 8000
        Refuse to run above this many tiles. See the note on memory.
    """

    features_model_name = "gigapath"
    needs_coords = True
    whole_slide = True

    def __init__(
        self,
        organ_type: str | None = None,
        tech_type: str | None = None,
        genes: Sequence[str] | None = None,
        model_path: str | None = None,
        gene_voc_path: str | None = None,
        token: str | None = None,
        max_spots: int = DEFAULT_MAX_SPOTS,
    ):
        try:
            import einops  # noqa: F401
        except ImportError:
            raise ImportError(
                "STPath requires einops. You can install it using "
                "`pip install einops`, or `uv sync --group model` for the "
                "full set of optional model dependencies."
            ) from None

        self.organ_id = _encode_organ(organ_type)
        self.tech_id = _encode_tech(tech_type)
        self.max_spots = max_spots
        self._device = torch.device("cpu")

        self.vocab = load_gene_vocab(gene_voc_path, token=token)

        if genes is None:
            # Column i of the head maps to gene_ids[i - 2]; 0 and 1 are pad/mask.
            self.gene_columns = np.arange(2, self.vocab.n_tokens)
            self.gene_names = tuple(self.vocab.gene_ids)
        else:
            genes = list(genes)
            ids, valid = self.vocab.symbol2id(genes)
            if not ids:
                raise ValueError(
                    "None of the requested genes are in STPath's vocabulary."
                )
            if len(ids) != len(genes):
                missing = [genes[i] for i in range(len(genes)) if i not in set(valid)]
                warnings.warn(
                    f"{len(missing)} of {len(genes)} requested genes are not in "
                    f"STPath's vocabulary and will be dropped: {missing[:10]}"
                    + (" ..." if len(missing) > 10 else ""),
                    stacklevel=find_stack_level(),
                )
            self.gene_columns = np.asarray(ids)
            self.gene_names = tuple(genes[i] for i in valid)

        if model_path is None:
            from huggingface_hub import hf_hub_download

            with hf_access(WEIGHTS_REPO):
                model_path = hf_hub_download(WEIGHTS_REPO, WEIGHTS_FILE, token=token)

        self.model = STFM(n_genes=self.vocab.n_tokens, n_tech=N_TECH, n_organs=N_ORGANS)
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self.model.eval()

    def to(self, device):
        self._device = torch.device(device)
        self.model.to(self._device)
        return self

    @torch.inference_mode()
    def predict(self, features, coords=None) -> dict[str, np.ndarray]:
        """Predict log1p expression for every tile of one slide.

        Parameters
        ----------
        features : array-like, shape (n_tiles, 1536)
            GigaPath tile features for the whole slide.
        coords : array-like, shape (n_tiles, 2)
            Tile origins in slide coordinates. Required.
        """
        if coords is None:
            raise ValueError(
                "STPath needs tile coordinates. Call it through "
                "`zs.tl.feature_prediction`, which supplies them, or pass "
                "`coords` explicitly."
            )

        features = np.asarray(features, dtype=np.float32)
        coords = np.asarray(coords, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != 1536:
            raise ValueError(
                "STPath features must have shape [n_tiles, 1536] (GigaPath), "
                f"got {tuple(features.shape)}"
            )
        if coords.shape != (features.shape[0], 2):
            raise ValueError(
                f"coords must have shape [{features.shape[0]}, 2], "
                f"got {tuple(coords.shape)}"
            )

        n_spots = features.shape[0]
        if n_spots > self.max_spots:
            raise ValueError(
                f"STPath attends over every spot at once, which costs O(n^2) "
                f"memory; {n_spots} tiles exceeds max_spots={self.max_spots}. "
                "STPath is a spot-level model, sized for the few thousand spots "
                "of one section. Either tile a region of interest instead of the "
                "whole slide, or tile coarsely -- around 256 um per tile "
                "(512 px at mpp 0.5, or 256 px at mpp 1.0) keeps a typical slide "
                "in range. Raise max_spots only if the machine really has the "
                "memory: roughly 4.7 GB at 4,000 spots and 9.6 GB at 6,000."
            )

        device = self._device
        img = torch.as_tensor(features, device=device)
        xy = rescale_coords(torch.as_tensor(coords, device=device))
        batch_idx = torch.zeros(n_spots, dtype=torch.long, device=device)
        tech = torch.full((n_spots,), self.tech_id, dtype=torch.long, device=device)
        organ = torch.full((n_spots,), self.organ_id, dtype=torch.long, device=device)

        pred = self.model.prediction_head(
            img_tokens=img,
            coords=xy,
            batch_idx=batch_idx,
            tech_tokens=tech,
            organ_tokens=organ,
        )
        pred = pred[:, self.gene_columns].float().cpu().numpy()
        return dict(zip(self.gene_names, pred.T, strict=True))
