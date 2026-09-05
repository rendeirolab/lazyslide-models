import warnings

import numpy as np
import torch
from torch import nn

from lazyslide_models._model_registry import register
from lazyslide_models._utils import hf_access
from lazyslide_models.base import ModelTask, TilePredictionModel

#: The 50 protein markers ROSIE predicts, in output order.
ROSIE_MARKERS = (
    "DAPI",
    "CD45",
    "CD68",
    "CD14",
    "PD1",
    "FoxP3",
    "CD8",
    "HLA-DR",
    "PanCK",
    "CD3e",
    "CD4",
    "aSMA",
    "CD31",
    "Vimentin",
    "CD45RO",
    "Ki67",
    "CD20",
    "CD11c",
    "Podoplanin",
    "PDL1",
    "GranzymeB",
    "CD38",
    "CD141",
    "CD21",
    "CD163",
    "BCL2",
    "LAG3",
    "EpCAM",
    "CD44",
    "ICOS",
    "GATA3",
    "Gal3",
    "CD39",
    "CD34",
    "TIGIT",
    "ECad",
    "CD40",
    "VISTA",
    "HLA-A",
    "MPO",
    "PCNA",
    "ATM",
    "TP63",
    "IFNg",
    "Keratin8/18",
    "IDO1",
    "CD79a",
    "HLA-E",
    "CollagenIV",
    "CD66",
)


@register(
    key="rosie",
    task=ModelTask.style_transfer,
    is_gated=True,
    license="CC-BY-NC-4.0",
    description="AI generation of multiplex immunofluorescence staining from histopathology images",
    commercial=False,
    github_url="https://gitlab.com/enable-medicine-public/rosie",
    paper_url="https://doi.org/10.1038/s41467-025-62346-0",
    bib_key="Wu2025-kv",
    param_size="50M",
    flops="17.37G",
)
class ROSIE(TilePredictionModel):
    # One value per marker per tile, not a dense map — so this is a tile
    # prediction, despite being registered under the `style_transfer` task.
    columns = ROSIE_MARKERS

    def __init__(self, model_path: str | None = None, token: str | None = None):
        from huggingface_hub import hf_hub_download
        from torchvision import models

        with hf_access("ericwu09/ROSIE"):
            weights_file = hf_hub_download(
                repo_id="ericwu09/ROSIE",
                filename="best_model_single.pth",
            )

        model = models.convnext_small(weights="IMAGENET1K_V1")
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 50)
        model = nn.DataParallel(model)
        weights = torch.load(weights_file, map_location="cpu", weights_only=False)
        model.load_state_dict(weights["model_state_dict"])
        model.eval()
        self.model = model

    @torch.inference_mode()
    def predict(self, image):
        """Predict all 50 marker intensities for a batch of tiles.

        ``image`` is a preprocessed ``[B, 3, 224, 224]`` tensor. Returns a dict
        mapping marker name to a length-``B`` NumPy array.
        """
        expression = self.model(image).float().cpu().numpy()
        return dict(zip(self.columns, np.asarray(expression).T, strict=True))

    def get_transform(self):
        import torch
        from torchvision.transforms.v2 import (
            Compose,
            Normalize,
            Resize,
            ToDtype,
            ToImage,
        )

        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Resize(size=(224, 224), antialias=False),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def check_input_tile(self, mpp, size_x=None, size_y=None) -> bool:
        # The x and y must be divisible by 8
        if mpp != 0.25:
            warnings.warn(
                "The ROSIE model is trained on image size 128x128 at mpp=0.25."
            )
        return True
