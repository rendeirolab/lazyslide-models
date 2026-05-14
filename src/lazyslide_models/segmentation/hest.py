import warnings

import torch

from lazyslide_models._model_registry import register
from lazyslide_models.base import ModelTask, SegmentationModel, SegmentationOutput


@register(
    key="hest-tissue-segmentation",
    task=ModelTask.segmentation,
    license="CC-BY-NC-SA-4.0",
    description="DeepLabV3 model finetuned on HEST-1k and Acrobat for IHC/H&E tissue segmentation.",
    commercial=False,
    hf_url="https://huggingface.co/MahmoodLab/hest-tissue-seg",
    param_size="39.6M",
    flops="62.61G",
)
class HESTTissueSegmentation(SegmentationModel):
    """
    Tissue segmentation model from HEST.

    512x512 with mpp=1 or 2
    """

    def __init__(self, model_path=None, token=None):
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download(
            "RendeiroLab/LazySlide-models", "HEST/HEST_tissue_seg_exported.pt2"
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given buffer is not writable"
            )
            self.model = torch.export.load(model_file).module()

    def get_transform(self):
        from torchvision.transforms.v2 import (
            Compose,
            Normalize,
            ToDtype,
            ToImage,
        )

        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @torch.inference_mode()
    def segment(self, image):
        return SegmentationOutput(
            probability_map=self.model(image)["out"].softmax(1),
            classes=("Background", "Tissue"),
        )
