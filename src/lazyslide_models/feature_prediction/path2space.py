import warnings

import numpy as np
import torch

from lazyslide_models._model_registry import register
from lazyslide_models.base import FeaturePredictionModel, ModelTask


@register(
    key="path2space",
    task=ModelTask.feature_prediction,
    license="Apache-2.0",
    license_url="https://github.com/eldadshulman/path2space-companion/blob/main/LICENSE",
    description="Predict spatial gene expression from CTransPath tile features.",
    commercial=True,
    github_url="https://github.com/eldadshulman/path2space-companion",
    paper_url="https://doi.org/10.1016/j.cell.2026.04.023",
    bib_key="Shulman2026-p2s",
    param_size="1.76B",
    vision_encoder="ctranspath",
)
class Path2Space(FeaturePredictionModel):
    """Path2Space gene-expression predictor.

    The model consumes raw 768-dimensional CTransPath tile features and
    returns one NumPy array per gene. Gene names and output ordering are read
    from the gene list published alongside the exported ensemble.
    """

    features_model_name = "ctranspath"

    def __init__(self, model_path=None, token=None):
        from huggingface_hub import hf_hub_download

        repo_id = "RendeiroLab/LazySlide-models"
        model_file = model_path or hf_hub_download(
            repo_id,
            "Path2Space/Path2Space_exported.pt2",
            token=token,
        )
        genes_file = hf_hub_download(
            repo_id,
            "Path2Space/Path2Space_genes.txt",
            token=token,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given buffer is not writable"
            )
            self.model = torch.export.load(model_file).module()

        with open(genes_file, encoding="utf-8") as handle:
            self.genes = tuple(line.strip() for line in handle if line.strip())

    @torch.inference_mode()
    def predict(self, features):
        """Predict expression for an ``[n_tiles, 768]`` feature matrix."""
        device = next(self.model.parameters()).device
        features = torch.as_tensor(features, dtype=torch.float32, device=device)
        if features.ndim != 2 or features.shape[1] != 768:
            raise ValueError(
                "Path2Space features must have shape [n_tiles, 768], "
                f"got {tuple(features.shape)}"
            )

        output = self.model(features).detach().cpu().numpy()
        return dict(zip(self.genes, np.asarray(output).T, strict=True))
