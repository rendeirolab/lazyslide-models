"""
Weight-free checks that preprocessing matches each model's upstream recipe.

Every reference below is the upstream eval pipeline (PIL + torchvision v1),
compared against our tensor pipeline on a noise tile, which exaggerates
resampling differences. PIL-vs-tensor rounding and float bicubic overshoot
stay below 6e-3 in mean; the smallest real divergence (bicubic instead of
bilinear, 0.5/0.5 stats) is 3.6e-2.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms as T

from lazyslide_models import MODEL_REGISTRY
from lazyslide_models.base import TimmModel

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
OPENAI_CLIP = (
    (0.48145466, 0.4578275, 0.40821073),
    (0.26862954, 0.26130258, 0.27577711),
)
LUNIT = (
    (0.70322989, 0.53606487, 0.66096631),
    (0.21716536, 0.26081574, 0.20723464),
)
HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
BILINEAR, BICUBIC = T.InterpolationMode.BILINEAR, T.InterpolationMode.BICUBIC

TILE = np.random.default_rng(0).integers(0, 256, (512, 512, 3), dtype=np.uint8)


def _pil(resize, crop, stats, interpolation=BILINEAR):
    return T.Compose(
        [
            T.Resize(resize, interpolation=interpolation),
            T.CenterCrop(crop),
            T.ToTensor(),
            T.Normalize(*stats),
        ]
    )


# Sources are cited in the audit PR (#29 follow-up).
UPSTREAM = {
    "uni": _pil(224, 224, IMAGENET),  # README Resize(224); hub cfg bilinear
    "uni2": _pil(224, 224, IMAGENET),
    "mstar": _pil(224, 224, IMAGENET),  # README Resize(224)
    "gigapath": _pil(256, 224, IMAGENET, BICUBIC),  # README Resize(256) + crop
    "gigapath-flash": _pil(256, 224, IMAGENET, BICUBIC),
    "virchow": _pil(224, 224, IMAGENET, BICUBIC),  # hub pretrained_cfg
    "virchow2": _pil(224, 224, IMAGENET, BICUBIC),
    # Lunit release notes give the stats; resize is unspecified upstream
    "lunit-bt": _pil(224, 224, LUNIT, BICUBIC),
    "lunit-mocov2": _pil(224, 224, LUNIT, BICUBIC),
    "lunit-swav": _pil(224, 224, LUNIT, BICUBIC),
    "lunit-dino-s8": _pil(224, 224, LUNIT, BICUBIC),
    "lunit-dino-s16": _pil(224, 224, LUNIT, BICUBIC),
    # AtlasPatch (MOOZY's extractor): timm cfg of the 1aurent port, crop_pct 0.9
    "lunit-dino-s8-moozy": _pil(248, 224, IMAGENET, BICUBIC),
    "ctranspath": _pil(224, 224, IMAGENET),  # Resize(224) on PIL
    "chief": _pil(224, 224, IMAGENET),
    "conch": _pil(448, 448, OPENAI_CLIP, BICUBIC),  # conch image_transform
    "conch-madeleine": _pil(224, 224, OPENAI_CLIP, BICUBIC),  # force_image_size
    "omiclip": _pil(224, 224, OPENAI_CLIP, BICUBIC),  # open_clip coca_ViT-L-14
    "biomedclip": _pil(224, 224, OPENAI_CLIP, BICUBIC),  # open_clip hub cfg
    "musk": _pil(384, 384, HALF, BICUBIC),  # IMAGENET_INCEPTION_MEAN/STD
    "titan": _pil(448, 448, IMAGENET),  # conch_v1_5.py Resize(BILINEAR)
    "mascaret": _pil(224, 224, HALF),  # README T.Resize(224), config stats
    "phaet": _pil(224, 224, IMAGENET),
    # ROSIE evaluate.py resizes the tensor after ToTensor
    "rosie": T.Compose(
        [T.ToTensor(), T.Resize(224, antialias=True), T.Normalize(*IMAGENET)]
    ),
}

# The Waiv encoders read their normalisation off the loaded config.
_WAIV_STATS = {"mascaret": HALF, "phaet": IMAGENET}


def _our_transform(key):
    """``get_transform()`` without loading weights."""
    cls = MODEL_REGISTRY[key]
    model = cls.__new__(cls)
    model.img_size = (224, 224)  # set from the checkpoint by TimmModel
    if key in _WAIV_STATS:
        mean, std = _WAIV_STATS[key]
        config = SimpleNamespace(pixel_mean=mean, pixel_std=std)
        model.model = SimpleNamespace(config=config)
    return model.get_transform()


@pytest.mark.parametrize("key", sorted(UPSTREAM))
def test_transform_matches_upstream(key: str) -> None:
    ours = _our_transform(key)(TILE)
    ref = UPSTREAM[key](Image.fromarray(TILE))
    assert ours.shape == ref.shape
    err = (ours - ref).abs().mean().item()
    assert err < 1e-2, f"{key}: mean |ours - upstream| = {err:.4f}"


def test_plain_timm_name_uses_its_pretrained_cfg() -> None:
    import timm

    model = TimmModel("resnet50", pretrained=False)  # cfg: bicubic, crop 0.95
    cfg = timm.data.resolve_model_data_config(model.model)
    ref = timm.data.create_transform(**cfg)(Image.fromarray(TILE))
    ours = model.get_transform()(TILE)
    assert ours.shape == ref.shape
    assert (ours - ref).abs().mean().item() < 1e-2


def test_moozy_patch_size_is_level0_tile_spacing() -> None:
    from lazyslide_models.vision.moozy import _tile_spacing

    # 448-px level-0 tiles (224 px at 0.5 mpp on a 40x scan) with gaps
    xs, ys = np.meshgrid([0, 448, 1344], [896, 1344])
    coords = torch.tensor(np.stack([xs.ravel(), ys.ravel()], axis=1))[None]
    assert _tile_spacing(coords) == 448


def test_haralick_keeps_white_pixels() -> None:
    from lazyslide_models.tile_prediction.cv_features import HaralickTexture

    # 254 and 255 share the top gray level, so the features must agree
    white, near_white = (np.full((1, 32, 32, 3), v, np.uint8) for v in (255, 254))
    ref = HaralickTexture().predict(near_white)
    out = HaralickTexture().predict(white)
    for col in HaralickTexture.columns:
        np.testing.assert_allclose(out[col], ref[col], err_msg=col)
