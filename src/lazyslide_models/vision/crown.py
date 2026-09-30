from typing import ClassVar

import torch

from lazyslide_models._model_registry import register
from lazyslide_models._utils import hf_access
from lazyslide_models.base import ModelTask, TimmViTModel

_HF_HUB_ID = "LexieK/Crown"


def _dinov2_to_timm(state_dict: dict) -> dict:
    """Rename a DINOv2 ``vit_large`` state dict (``block_chunks=4``,
    ``ffn_layer="swiglufused"``) to timm's ViT names."""
    renamed = {}
    for key, value in state_dict.items():
        if key == "mask_token":  # only used in DINOv2 training
            continue
        parts = key.split(".")
        if parts[0] == "blocks":  # blocks.<chunk>.<i>.* -> blocks.<i>.*
            key = ".".join(["blocks", *parts[2:]])
        key = key.replace(".mlp.w12.", ".mlp.fc1.").replace(".mlp.w3.", ".mlp.fc2.")
        renamed[key] = value
    return renamed


@register(
    key="crown",
    is_gated=True,
    task=ModelTask.vision,
    license="Apache-2.0",
    description="A universal visual foundation model for computational cytopathology",
    commercial=True,
    hf_url="https://huggingface.co/LexieK/Crown",
    github_url="https://github.com/LexieK7/CROWN",
    paper_url="https://doi.org/10.1038/s43018-026-01240-0",
    bib_key="Zheng2026-crown",
    encode_dim=1024,
)
class CROWN(TimmViTModel):
    """CROWN tile encoder for cytology.

    A DINOv2 ViT-L/16 pretrained on more than 10 million cytology image patches.
    It is the first zoo encoder trained on cytology rather than histology.
    Returns the normalised class token (1024-d).
    """

    # eval scripts: Resize(224) (bilinear), CenterCrop(224), ImageNet stats.
    # The README example leaves out Normalize, but the eval code normalises.
    transform_kws: ClassVar[dict] = {"interpolation": "bilinear"}

    def __init__(self, model_path=None, token=None):
        from huggingface_hub import hf_hub_download
        from timm.layers import SwiGLUPacked

        super().__init__(
            "vit_large_patch16_224",
            pretrained=False,
            img_size=224,
            init_values=1.0,
            # DINOv2's fused SwiGLU hidden size: 2 x 2736 for dim 1024
            mlp_ratio=5472 / 1024,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        with hf_access(_HF_HUB_ID):
            weights = model_path or hf_hub_download(
                _HF_HUB_ID, "CROWN.pth", token=token
            )
        state_dict = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(_dinov2_to_timm(state_dict))
        self.model.eval()
