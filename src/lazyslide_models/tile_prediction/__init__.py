from .cv_features import (
    Brightness,
    Canny,
    ColorDeconvolution,
    ColorDeconvolutionMap,
    Contrast,
    Entropy,
    HaralickTexture,
    Saturation,
    Sharpness,
    Sobel,
    SplitRGB,
)
from .deepspotm import DeepSpotM
from .focuslitenn import FocusLiteNN
from .pathprofiler_qc import PathProfilerQC
from .spider import (
    Spider,
    SpiderBreast,
    SpiderColorectal,
    SpiderSkin,
    SpiderThorax,
)

# Tile-level models only: tile_prediction runs these on the CPU for their
# columns. A dense cv_feature such as ColorDeconvolutionMap goes through
# virtual_stain instead.
CV_FEATURES = {
    "split_rgb": SplitRGB,
    "brightness": Brightness,
    "contrast": Contrast,
    "sobel": Sobel,
    "canny": Canny,
    "sharpness": Sharpness,
    "entropy": Entropy,
    "saturation": Saturation,
    "haralick_texture": HaralickTexture,
    "color_deconvolution": ColorDeconvolution,
}


__all__ = [
    "CV_FEATURES",
    "Brightness",
    "Canny",
    "ColorDeconvolution",
    "ColorDeconvolutionMap",
    "Contrast",
    "DeepSpotM",
    "Entropy",
    "FocusLiteNN",
    "HaralickTexture",
    "PathProfilerQC",
    "Saturation",
    "Sharpness",
    "Sobel",
    "Spider",
    "SpiderBreast",
    "SpiderColorectal",
    "SpiderSkin",
    "SpiderThorax",
    "SplitRGB",
]
