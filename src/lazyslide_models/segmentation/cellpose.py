import numpy as np

from lazyslide_models._model_registry import register
from lazyslide_models.base import ModelTask, SegmentationModel, SegmentationOutput


@register(
    key="cellpose",
    task=ModelTask.segmentation,
    license="BSD-3-Clause",
    description="Cell segmentation model",
    commercial=True,
    github_url="https://github.com/MouseLand/cellpose",
    hf_url="https://huggingface.co/mouseland/cellpose-sam",
    paper_url="https://doi.org/10.1038/s41592-020-01018-x",
    bib_key="Stringer2021-cx",
)
class Cellpose(SegmentationModel):
    """
    Cellpose-SAM cell segmentation. Needs cellpose>=4.2.0.

    ``pretrained_model`` picks the weights. The default, ``"cpsam_v2"``, is
    upstream's default since cellpose 4.2: the same SAM ViT-L as the original
    ``"cpsam"``, retrained to predict fewer spurious masks in low-contrast
    regions. ``"cpsam"`` is still available. ``"cpdino"`` and ``"cpdino-vitb"``
    also need ``facebookresearch/dinov3`` installed and come under the DINOv3
    licence.

    .. code-block:: python

       >>> zs.seg.cells(wsi, model="cellpose", pretrained_model="cpsam")

    If you want to fine-tune the cellpose model, please take a look at the following resources:

    - https://github.com/MouseLand/cellpose/blob/main/notebooks/train_Cellpose-SAM.ipynb
    - https://cellpose.readthedocs.io/en/latest/train.html

    To run a fine-tuned model, pass the `model_path` argument pointing to the fine-tuned weights.

    .. code-block:: python

       >>> zs.seg.cells(wsi, model="cellpose", model_path="fine-tuned-checkpoint.pth")

    """

    def __init__(
        self,
        diam_mean=None,
        model_path=None,
        pretrained_model="cpsam_v2",
        **eval_kwargs,
    ):
        import os

        try:
            from cellpose import models
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install cellpose>=4.2.0")

        weights = model_path if model_path is not None else pretrained_model
        # cellpose only logs a warning for an unknown name or a missing file,
        # then quietly runs its default model instead
        if weights not in models.MODEL_NAMES and not os.path.exists(weights):
            raise ValueError(
                f"{weights!r} is neither a file nor a model this cellpose knows "
                f"({', '.join(models.MODEL_NAMES)}). cpsam_v2 needs cellpose>=4.2.0."
            )

        self.model = models.CellposeModel(
            pretrained_model=weights,
            diam_mean=diam_mean,
            gpu=True,
        )
        self.eval_kwargs = eval_kwargs

    def to(self, device):
        import torch

        device = torch.device(device)
        # cellpose moves inputs to self.model.device, so the network must follow
        self.model.device = device
        self.model.net.to(device)
        return self

    def get_transform(self):
        return None

    def segment(self, image):
        if image.ndim == 4:
            # If the image is a batch, we need to make it into a list of images
            image = [img.detach().cpu().numpy() for img in image]

        masks, _, _ = self.model.eval(image, batch_size=len(image), **self.eval_kwargs)
        if isinstance(masks, list):
            # If the masks are a list, we need to convert them to a numpy array
            masks = np.array(masks)
        elif masks.ndim == 2:
            # If the masks are a single image, we need to add a batch dimension
            masks = masks[np.newaxis, ...]
        return SegmentationOutput(instance_map=masks)
