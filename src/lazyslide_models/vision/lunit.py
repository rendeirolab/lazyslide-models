from typing import ClassVar

from lazyslide_models._model_registry import register
from lazyslide_models.base import ModelTask, TimmModel, TimmViTModel

shared_info = {
    "task": ModelTask.vision,
    "license": "lunit-non-commercial",
    "license_url": "https://github.com/lunit-io/benchmark-ssl-pathology/blob/main/LICENSE",
    "description": "Benchmarking Self-Supervised Learning on Diverse Pathology Datasets",
    "commercial": False,
    "github_url": "https://github.com/lunit-io/benchmark-ssl-pathology",
    "paper_url": "https://doi.org/10.48550/arXiv.2212.04690",
    "bib_key": "Kang2023-bp",
}

# Per-channel RGB statistics from the upstream release notes
# (github.com/lunit-io/benchmark-ssl-pathology/releases/tag/pretrained-weights).
# The 1aurent timm ports we load declare ImageNet stats instead.
LUNIT_MEAN = (0.70322989, 0.53606487, 0.66096631)
LUNIT_STD = (0.21716536, 0.26081574, 0.20723464)


class LunitResNet50(TimmModel):
    """Base class for Lunit ResNet50 SSL variants."""

    _hf_hub_id: str
    transform_kws: ClassVar[dict] = {"mean": LUNIT_MEAN, "std": LUNIT_STD}

    def __init__(self, model_path=None, token=None):
        super().__init__(f"hf-hub:{self._hf_hub_id}", token=token)


class LunitViTSmall(TimmViTModel):
    """Base class for Lunit ViT-Small DINO variants."""

    _hf_hub_id: str
    transform_kws: ClassVar[dict] = {"mean": LUNIT_MEAN, "std": LUNIT_STD}

    def __init__(self, model_path=None, token=None):
        super().__init__(f"hf-hub:{self._hf_hub_id}", token=token)


@register(
    key="lunit-bt",
    **shared_info,
    hf_url="https://huggingface.co/1aurent/resnet50.lunit_bt",
    param_size="23.6M",
    encode_dim=2048,
)
class LunitResNet50BT(LunitResNet50):
    _hf_hub_id = "1aurent/resnet50.lunit_bt"


@register(
    key="lunit-mocov2",
    **shared_info,
    hf_url="https://huggingface.co/1aurent/resnet50.lunit_mocov2",
    param_size="23.6M",
    encode_dim=2048,
)
class LunitResNet50MoCoV2(LunitResNet50):
    _hf_hub_id = "1aurent/resnet50.lunit_mocov2"


@register(
    key="lunit-swav",
    **shared_info,
    hf_url="https://huggingface.co/1aurent/resnet50.lunit_swav",
    param_size="23.6M",
    encode_dim=2048,
)
class LunitResNet50SwAV(LunitResNet50):
    _hf_hub_id = "1aurent/resnet50.lunit_swav"


@register(
    key="lunit-dino-s8",
    **shared_info,
    hf_url="https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino",
    param_size="21.7M",
    encode_dim=384,
)
class LunitDINOPatch8(LunitViTSmall):
    _hf_hub_id = "1aurent/vit_small_patch8_224.lunit_dino"


@register(
    key="lunit-dino-s8-moozy",
    **{
        **shared_info,
        "description": "Lunit DINO ViT-S/8 with the preprocessing MOOZY was trained on",
    },
    hf_url="https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino",
    param_size="21.7M",
    encode_dim=384,
)
class LunitDINOPatch8Moozy(LunitDINOPatch8):
    """``lunit-dino-s8`` preprocessed like MOOZY's feature extractor.

    MOOZY's AtlasPatch pipeline builds the transform from the timm config of
    the 1aurent port: ImageNet stats, bicubic, ``crop_pct`` 0.9. Use this key,
    not ``lunit-dino-s8``, for features that feed ``moozy``.
    """

    transform_kws: ClassVar[dict] = {"crop_pct": 0.9}


@register(
    key="lunit-dino-s16",
    **shared_info,
    hf_url="https://huggingface.co/1aurent/vit_small_patch16_224.lunit_dino",
    param_size="21.1M",
    encode_dim=384,
)
class LunitDINOPatch16(LunitViTSmall):
    _hf_hub_id = "1aurent/vit_small_patch16_224.lunit_dino"
