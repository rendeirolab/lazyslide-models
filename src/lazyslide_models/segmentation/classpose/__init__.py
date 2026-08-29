"""Classpose: foundation-model-driven cell phenotyping in H&E."""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
import torch

from lazyslide_models._model_registry import register
from lazyslide_models._utils import hf_access
from lazyslide_models.base import (
    InputConstraint,
    ModelTask,
    SegmentationModel,
    SegmentationOutput,
)

from ._blocks import build_class_transformer

__all__ = ["Classpose"]

HF_REPO = "classpose/classpose"

#: Per-variant training resolution and cell-type vocabulary, copied from
#: upstream ``src/classpose/model_configs.py``. Class index 0 is background, so
#: the registered class tuple is ``("Background", *cell_types)``.
VARIANTS: dict[str, dict] = {
    "conic": {
        "mpp": 0.5,
        "cell_types": (
            "Neutrophil",
            "Epithelial",
            "Lymphocyte",
            "Plasma cell",
            "Eosinophil",
            "Connective",
        ),
    },
    "consep": {
        "mpp": 0.25,
        "cell_types": (
            "Other",
            "Inflammatory",
            "Healthy epithelial",
            "Malignant epithelial",
            "Stroma",
            "Muscle",
        ),
    },
    "glysac": {
        "mpp": 0.25,
        # Upstream's model_configs.py lists a fourth "Ambiguous" type here, which
        # makes its own `len(cell_types) == n_classes - 1` check reject the
        # checkpoint: glysac.pt has 4 output classes, i.e. background + 3. The
        # Hugging Face model card agrees the GlySAC model has three classes
        # ("lymphocytes, epithelial ... and other cell types"), so the trailing
        # entry is dropped. Reported upstream.
        "cell_types": ("Other", "Lymphocyte", "Epithelial"),
    },
    "monusac": {
        "mpp": 0.25,
        "cell_types": ("Epithelial", "Lymphocyte", "Macrophage", "Neutrophil"),
    },
    "nucls": {
        "mpp": 0.2,
        "cell_types": (
            "Tumor",
            "Stroma",
            "Lymphocyte",
            "Plasma cell",
            "Macrophage",
            "Other",
        ),
    },
    "puma": {
        "mpp": 0.22,
        "cell_types": (
            "Apoptosis",
            "Tumor",
            "Endothelial",
            "Stroma",
            "Lymphocyte",
            "Histocyte",
            "Epithelial",
            "Melanophage",
            "Other",
        ),
    },
}

_UNET_BLOCK_KEY = re.compile(r"out_class\.encoder_blocks\.[0-9]+\.block\.conv1\.weight")


def _infer_head(state: dict) -> tuple[int, list[int] | None]:
    """Read the head shape out of a checkpoint, as upstream's ``infer_structure`` does.

    Returns ``(n_classes, feature_transformation_structure)`` where
    ``n_classes`` counts the background class and the structure is ``None`` for
    the single-``Conv2d`` head.
    """
    structure = [state[k].shape[0] for k in state if _UNET_BLOCK_KEY.search(k)] or None
    return int(state["W3"].shape[1]), structure


@register(
    key="classpose",
    task=ModelTask.segmentation,
    is_gated=False,
    # Upstream is self-contradictory: the repository LICENSE file is
    # CC BY-NC 4.0 while the Hugging Face model card states MIT. We record the
    # stricter of the two until the authors clarify.
    license="CC-BY-NC-4.0",
    license_url="https://github.com/sohmandal/classpose/blob/main/LICENSE",
    commercial=False,
    description="Foundation-model-driven whole-slide-scale cell phenotyping in H&E.",
    github_url="https://github.com/sohmandal/classpose",
    hf_url="https://huggingface.co/classpose/classpose",
    paper_url="https://doi.org/10.64898/2025.12.18.695211",
    bib_key="Mandal2025-cp",
    param_size="304M",
    input_constraint=InputConstraint(min=256, max=256),
)
class Classpose(SegmentationModel):
    """Cellpose-SAM with a semantic head, for joint cell segmentation and typing.

    One checkpoint per training dataset; each carries its own resolution and
    cell-type vocabulary:

    ==========  ====  ==================================================
    variant     mpp   cell types (class 0 is always Background)
    ==========  ====  ==================================================
    conic       0.5   Neutrophil, Epithelial, Lymphocyte, Plasma cell,
                      Eosinophil, Connective
    consep      0.25  Other, Inflammatory, Healthy epithelial,
                      Malignant epithelial, Stroma, Muscle
    glysac      0.25  Other, Lymphocyte, Epithelial, Ambiguous
    monusac     0.25  Epithelial, Lymphocyte, Macrophage, Neutrophil
    nucls       0.2   Tumor, Stroma, Lymphocyte, Plasma cell,
                      Macrophage, Other
    puma        0.22  Apoptosis, Tumor, Endothelial, Stroma, Lymphocyte,
                      Histocyte, Epithelial, Melanophage, Other
    ==========  ====  ==================================================

    .. code-block:: python

        >>> import lazyslide as zs
        >>> zs.pp.tile_tissues(wsi, 256, mpp=0.5)
        >>> zs.seg.cells(wsi, model="classpose", model_kwargs={"variant": "conic"})

    Upstream warns that the ``nucls`` checkpoint is trained largely on bounding
    box annotations and performs worse than the others.

    Parameters
    ----------
    variant : str, default: "conic"
        Which trained checkpoint to load. See the table above.
    model_path : str, optional
        Local checkpoint to load instead of downloading from Hugging Face.
    token : str, optional
        Hugging Face access token.
    flow_threshold, cellprob_threshold, min_size : float, float, int
        Passed through to Cellpose's mask dynamics.
    """

    def __init__(
        self,
        variant: Literal[
            "conic", "consep", "glysac", "monusac", "nucls", "puma"
        ] = "conic",
        model_path: str | None = None,
        token: str | None = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 15,
    ):
        if variant not in VARIANTS:
            raise ValueError(
                f"Unknown Classpose variant '{variant}'. "
                f"Choose one of {sorted(VARIANTS)}."
            )
        self.variant = variant
        self.mpp = VARIANTS[variant]["mpp"]
        cell_types = VARIANTS[variant]["cell_types"]
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.min_size = min_size
        self._device = torch.device("cpu")

        if model_path is None:
            from huggingface_hub import hf_hub_download

            with hf_access(HF_REPO):
                model_path = hf_hub_download(HF_REPO, f"{variant}.pt", token=token)

        state = torch.load(model_path, map_location="cpu", weights_only=True)
        n_classes, structure = _infer_head(state)
        if n_classes != len(cell_types) + 1:
            raise ValueError(
                f"Checkpoint for '{variant}' declares {n_classes} classes but "
                f"{len(cell_types)} cell types are registered (+1 background)."
            )
        self.classes = ("Background", *cell_types)

        self.model = build_class_transformer(
            n_cell_classes=n_classes,
            feature_transformation_structure=structure,
        )
        self.model.load_state_dict(state)
        self.model.eval()

    def to(self, device):
        self._device = torch.device(device)
        self.model.to(self._device)
        return self

    def get_transform(self):
        # Cellpose normalises per image by the 1st/99th percentile, which is not
        # expressible as a torchvision transform. Done inside `segment` instead.
        return None

    @torch.inference_mode()
    def segment(self, image) -> SegmentationOutput:
        from cellpose import dynamics, transforms
        from cellpose.models import normalize_default

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        if image.ndim == 3:
            image = image[np.newaxis, ...]

        # Tiles arrive channel-last (B, H, W, C), which is Cellpose's layout.
        x = np.stack(
            [transforms.normalize_img(img, **normalize_default) for img in image]
        )
        x = torch.as_tensor(
            x.transpose(0, 3, 1, 2), dtype=torch.float32, device=self._device
        )

        out = self.model(x)
        n = len(self.classes)
        prob = torch.softmax(out[:, :n], dim=1).float().cpu().numpy()
        flows = out[:, n:].float().cpu().numpy()

        masks = np.stack(
            [
                dynamics.resize_and_compute_masks(
                    flows[i, :2],
                    flows[i, 2],
                    flow_threshold=self.flow_threshold,
                    cellprob_threshold=self.cellprob_threshold,
                    min_size=self.min_size,
                    device=self._device,
                )
                for i in range(len(flows))
            ]
        )

        return SegmentationOutput(
            instance_map=masks,
            probability_map=prob,
            classes=self.classes,
        )
