from __future__ import annotations

import numpy as np
import torch

from lazyslide_models._model_registry import register
from lazyslide_models._utils import hf_access
from lazyslide_models.base import ModelTask, TilePredictionModel


@register(
    key="deepspotm",
    task=ModelTask.tile_prediction,
    is_gated=True,
    license=["PolyForm-Noncommercial-1.0.0", "CC-BY-NC-SA-4.0"],
    license_url=[
        "https://github.com/ratschlab/DeepSpotM/blob/main/LICENSE",
        "https://github.com/ratschlab/DeepSpotM/blob/main/WEIGHTS_LICENSE.md",
    ],
    description="Transcriptome-wide virtual spatial transcriptomics from H&E images.",
    commercial=False,
    github_url="https://github.com/ratschlab/DeepSpotM",
    hf_url="https://huggingface.co/ratschlab/DeepSpotM",
    paper_url="https://www.medrxiv.org/content/10.64898/2026.06.19.26356060v1",
    bib_key="Nonchev2026-deepspotm",
    vision_encoder="midnight",
)
class DeepSpotM(TilePredictionModel):
    """DeepSpot-M virtual spatial transcriptomics predictor.

    Maps a 224x224 H&E tile to spatial gene expression with a LoRA-adapted
    Midnight backbone and a cross-attention gene decoder. One model predicts a
    transcriptome-wide (~19k gene) panel and, because genes are represented as
    queryable embeddings, genes it never saw during training.

    The weights live in the gated ``ratschlab/DeepSpotM`` Hugging Face repo:
    accept the terms on the model page and authenticate (``huggingface-cli
    login``, the ``HF_TOKEN`` environment variable, or the ``token`` argument)
    before first use.

    ``predict`` returns one array per gene, keyed by gene symbol. By default the
    full panel is predicted; each gene becomes a column on the tile table, which
    is transcriptome-wide but wide to store. Pass ``genes`` to score only a
    marker panel, which is faster (only those gene queries are computed) and
    lean to store::

        from lazyslide_models.tile_prediction import DeepSpotM

        markers = DeepSpotM(genes=["BRAF", "CD37", "COL1A1"])
        zs.tl.tile_prediction(wsi, markers)

    Parameters
    ----------
    source : str, default: "scgpt"
        Gene-embedding source used by the gene router; one of ``evo2``,
        ``orthrus``, ``prott5``, ``scgpt``, ``apertus``.
    genes : str or sequence of str, optional
        Restrict prediction to these gene symbols. When omitted, the full panel
        (``model.gene_names``) is predicted.
    token : str, optional
        Hugging Face access token, used to log in before download.
    """

    # The gene panel depends on the `genes` argument and not known before init
    columns = None

    def __init__(
        self,
        source: str = "scgpt",
        genes=None,
        token: str | None = None,
    ):
        try:
            from deepspotm import DeepSpotM as _DeepSpotM
        except (ImportError, ModuleNotFoundError) as e:
            raise ImportError(
                "To use DeepSpotM, you need to install deepspotm: "
                "pip install git+https://github.com/ratschlab/DeepSpotM.git"
            ) from e

        if token is not None:
            from huggingface_hub import login

            login(token)

        # from_pretrained returns (model, image_processor); the processor is
        # the backbone's own eval transform (center-crop to 224 + Midnight
        # normalization). It is stored on the wrapper so get_transform can hand
        # it to LazySlide's tile DataLoader.
        with hf_access("ratschlab/DeepSpotM"):
            self.model, self._transform = _DeepSpotM.from_pretrained(
                "ratschlab/DeepSpotM", source=source
            )
        self.model.eval()

        if genes is None:
            self.genes = None
        elif isinstance(genes, str):
            self.genes = [genes]
        else:
            self.genes = list(genes)

        # Now that the panel is known, record the concrete columns. Both
        # branches are exact: `self.genes` when the caller picked a panel, the
        # backbone's own list otherwise.
        self.columns = (
            tuple(self.genes)
            if self.genes is not None
            else tuple(self.model.gene_names)
        )

    def get_transform(self):
        # Return DeepSpot-M's own transform, not LazySlide's default. The
        # backbone is normalized with mean/std 0.5 (Midnight), so applying the
        # default ImageNet transform here would silently degrade predictions.
        return self._transform

    @torch.inference_mode()
    def predict(self, image):
        """Predict expression for a batch of tiles.

        ``image`` is a preprocessed ``[B, 3, 224, 224]`` tensor. Returns a dict
        mapping gene symbol to a length-``B`` NumPy array (one column per gene).
        """
        image = image.to(next(self.model.parameters()).device)
        if self.genes is not None:
            expression = self.model.predict_genes(image, self.genes)
        else:
            expression, _, _ = self.model(image)

        expression = expression.float().cpu().numpy()
        # `columns` was built from the same panel in __init__, so it is the
        # single source of truth for column order.
        return dict(zip(self.columns, np.asarray(expression).T, strict=True))
