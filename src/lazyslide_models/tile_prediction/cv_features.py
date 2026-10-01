from __future__ import annotations

import math
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
from skimage.color import hdx_from_rgb, hed_from_rgb
from skimage.util import dtype_limits

from lazyslide_models._model_registry import register
from lazyslide_models.base import MarkerMapModel, ModelTask, TilePredictionModel

__all__ = [
    "Brightness",
    "Canny",
    "ColorDeconvolution",
    "ColorDeconvolutionMap",
    "Contrast",
    "Entropy",
    "HaralickTexture",
    "Saturation",
    "Sharpness",
    "Sobel",
    "SplitRGB",
]


def _as_batch(image):
    """Tiles as a contiguous uint8 NumPy ``[B, H, W, 3]`` batch."""
    if torch.is_tensor(image):
        # A runner may have moved the batch to an accelerator: LazySlide does
        # for a model instance, as opposed to a registry key.
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 3:
        image = image[None]
    if image.shape[1] == 3 and image.shape[-1] == 3:
        raise ValueError(
            f"Cannot tell BHWC from BCHW for a batch of shape {image.shape}; "
            "pass tiles larger than 3 pixels."
        )
    if image.shape[1] == 3:  # BCHW
        image = image.transpose(0, 2, 3, 1)
    if image.dtype != np.uint8:
        # cv2 and the uint8 casts below would otherwise return garbage silently
        raise TypeError(
            f"cv features take uint8 RGB tiles, got {image.dtype}; "
            "drop any transform that converts them to float."
        )
    return np.ascontiguousarray(image)


class _CVFeatures(TilePredictionModel, ABC):
    #: Subclasses whose ``_func`` returns a dict set this explicitly; for the
    #: rest it is derived below from the class name, which is the same rule
    #: ``_process_batch`` uses to key its result.
    columns: tuple[str, ...] | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "columns" not in cls.__dict__:
            cls.columns = (cls.__name__.lower(),)

    def to(self, device):
        return self

    def get_transform(self):
        return None

    @abstractmethod
    def _func(self, image):
        """
        Abstract method to be implemented by subclasses.
        This method should process a single image and return the feature.
        """

    def _process_batch(self, images):
        results = [self._func(image) for image in images]

        if isinstance(results[0], dict):
            batch_results = {
                key: np.array([r[key] for r in results]) for key in results[0]
            }
            return batch_results
        else:
            return {self.__class__.__name__.lower(): np.array(results)}

    def predict(self, image):
        return self._process_batch(_as_batch(image))


class CVCompose(_CVFeatures):
    """
    Compose multiple CV features into a single feature.

    This class allows you to combine multiple CV features into a single feature.
    It is useful for creating a composite feature that includes multiple aspects
    of the image, such as brightness, contrast, and color information.

    Parameters
    ----------
    *models : list of _CVFeatures
        List of CV feature instances to be composed.
    """

    # Composed from the children, so the class-name default would be wrong.
    columns = None

    def __init__(self, *models):
        self.models = models
        self.columns = tuple(n for m in models for n in m.columns)

    def _func(self, image):
        pass

    def predict(self, image):
        image = _as_batch(image)

        results = {}
        for model in self.models:
            model_results = model.predict(image)
            results.update(model_results)
        return results


@register(
    key="split_rgb",
    task=ModelTask.cv_feature,
    description="Mean red, green and blue intensity of a tile",
)
class SplitRGB(_CVFeatures):
    """
    Calculate the red, green and blue intensity of a tile, on a 0-255 scale.

    Parameters
    ----------
    method : str
        Name of the NumPy reduction applied to each channel, e.g. ``"mean"``,
        ``"median"`` or ``"std"``. Default is ``"mean"``.
    """

    columns = ("red", "green", "blue")

    def __init__(
        self,
        method: str = "mean",
    ):
        self.method = method

    def _func(self, image):
        if self.method == "mean":
            # Same values, ~20x faster than ndarray.mean over a strided uint8 axis
            c_int = cv2.mean(image)
        else:
            c_int = getattr(np, self.method)(image, axis=(0, 1))
        return {"red": c_int[0], "green": c_int[1], "blue": c_int[2]}


@register(
    key="brightness",
    task=ModelTask.cv_feature,
    description="Mean pixel intensity of a tile",
)
class Brightness(_CVFeatures):
    """
    Calculate the brightness of a tile.

    The tile can be in shape (H, W, C) for a single image or (B, C, H, W) for a batch of images.
    """

    def _func(self, image):
        return image.mean()


@register(
    key="contrast",
    task=ModelTask.cv_feature,
    description="Spread between the 1st and 99th gray-level percentiles",
)
class Contrast(_CVFeatures):
    """
    Calculate the contrast of a tile.

    Contrast is the gray-level range between ``lower_percentile`` and
    ``upper_percentile``, as a fraction of the dtype range.

    The tile can be in shape (H, W, C) for a single image or (B, C, H, W) for a batch of images.

    Parameters
    ----------
    lower_percentile : float
        Lower percentile for contrast calculation.
    upper_percentile : float
        Upper percentile for contrast calculation.
    """

    def __init__(
        self,
        lower_percentile: float = 1,
        upper_percentile: float = 99,
    ):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def _func(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        dlimits = dtype_limits(image, clip_negative=False)
        limits = np.percentile(image, [self.lower_percentile, self.upper_percentile])
        ratio = (limits[1] - limits[0]) / (dlimits[1] - dlimits[0])
        return ratio


@register(
    key="sharpness",
    task=ModelTask.cv_feature,
    description="Variance of the Laplacian; higher means a sharper tile",
)
class Sharpness(_CVFeatures):
    """
    Calculate the sharpness of a tile.

    Sharpness is calculated as the variance of the Laplacian of the pixel values.
    The Laplacian operator is used to measure the second derivative of an image,
    which highlights regions of rapid intensity change and is therefore often used
    for edge detection. High variance in the Laplacian indicates a sharper image.

    The tile can be in shape (H, W, C) for a single image or (B, C, H, W) for a batch of images.
    """

    def _func(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Apply Laplacian operator
        laplacian = cv2.Laplacian(gray_image.astype(np.float32), cv2.CV_32F)
        return laplacian.var()


@register(
    key="sobel",
    task=ModelTask.cv_feature,
    description="Variance of the Sobel gradient, an edge-strength measure",
)
class Sobel(_CVFeatures):
    """
    Calculate the sobel of a tile.

    Sobel is calculated as the variance of the Sobel of the pixel values.

    The Sobel operator calculates the gradient of the image intensity at each pixel,
    giving the direction of the largest possible increase from light to dark and the
    rate of change in that direction.

    The tile can be in shape (H, W, C) for a single image or (B, C, H, W) for a batch of images.
    """

    def __init__(self, ksize=3):
        self.ksize = ksize

    def _func(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Calculate Sobel in x and y directions
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.ksize)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.ksize)

        # Calculate the magnitude of gradients
        magnitude = np.sqrt(sobelx**2 + sobely**2)

        # Calculate variance of the magnitude
        return magnitude.var()


@register(
    key="canny",
    task=ModelTask.cv_feature,
    description="Fraction of pixels on a Canny edge",
)
class Canny(_CVFeatures):
    """
    Calculate the Canny edge density of a tile.

    The Canny edge detector is an edge detection operator that uses a multi-stage
    algorithm to detect a wide range of edges in images. The score is the
    fraction of pixels that lie on an edge, between 0 and 1.

    The tile can be in shape (H, W, C) for a single image or (B, C, H, W) for a batch of images.

    Parameters
    ----------
    low_threshold : int
        Lower threshold for the hysteresis procedure in Canny edge detection.
    high_threshold : int
        Higher threshold for the hysteresis procedure in Canny edge detection.
    """

    def __init__(self, low_threshold=100, high_threshold=200):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def _func(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, self.low_threshold, self.high_threshold)
        return cv2.countNonZero(edges) / edges.size


@register(
    key="entropy",
    task=ModelTask.cv_feature,
    description="Shannon entropy of the gray-level histogram, in bits",
)
class Entropy(_CVFeatures):
    """
    Calculate the entropy of a tile.

    The Shannon entropy, in bits, of the tile's 256-bin gray-level histogram:
    higher means the intensities are spread over more levels. It depends on the
    histogram alone, not on where the pixels are, so shuffling a tile leaves it
    unchanged; for texture, use ``HaralickTexture``.

    The tile can be in shape (H, W, C) for a single image or (B, C, H, W) for a batch of images.
    """

    def _func(self, image):
        # Convert to grayscale if the image is in color
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Calculate histogram
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

        # Normalize histogram to get probability distribution
        hist = hist / hist.sum()

        # Remove zero probabilities (log(0) is undefined)
        hist = hist[hist > 0]

        # Calculate entropy
        entropy_value = -np.sum(hist * np.log2(hist))

        return entropy_value


@register(
    key="saturation",
    task=ModelTask.cv_feature,
    description="Mean saturation of the HSV saturation channel",
)
class Saturation(_CVFeatures):
    """
    Calculate the color saturation of a tile.

    Saturation measures the colorfulness of an image. It is calculated by converting
    the image to HSV color space and taking the mean of the saturation channel.
    Higher values indicate more vibrant colors.

    The tile can be in shape (H, W, C) for a single image or (B, C, H, W) for a batch of images.
    """

    def _func(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Extract saturation channel (second channel in HSV)
        saturation_channel = hsv_image[:, :, 1]

        # Calculate mean saturation
        mean_saturation = saturation_channel.mean() / 255.0  # Normalize to [0, 1]

        return mean_saturation


@register(
    key="haralick_texture",
    task=ModelTask.cv_feature,
    description="Haralick texture features from the gray-level co-occurrence matrix",
)
class HaralickTexture(_CVFeatures):
    """
    Calculate texture features using Gray Level Co-occurrence Matrix (GLCM).

    This class implements Haralick texture features which are derived from the GLCM.
    These features provide information about the texture of an image and are widely
    used in image analysis.

    The co-occurrence matrices of all offsets, each counting only pixel pairs
    inside the tile, are averaged before the features are computed. As in
    scikit-image's ``graycoprops``, ``texture_energy`` is the angular second
    moment (its ``"ASM"``; its ``"energy"`` is the square root). Correlation is
    0 on a flat tile, where it is undefined.

    The tile can be in shape (H, W, C) for a single image or (B, C, H, W) for a batch of images.

    Parameters
    ----------
    distances : list of int
        List of pixel pair distance offsets.
    angles : list of float
        List of pixel pair angles in radians.
    levels : int
        Number of gray levels to use in the GLCM, from 2 to 256.
    """

    columns = (
        "texture_energy",
        "texture_contrast",
        "texture_homogeneity",
        "texture_correlation",
        "texture_entropy",
    )

    def __init__(self, distances=None, angles=None, levels=8):
        self.distances = distances if distances is not None else [1]
        self.angles = (
            angles if angles is not None else [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        )
        if not 2 <= levels <= 256:
            # 8-bit gray has 256 values, and the quantised tile is uint8
            raise ValueError(f"levels must be between 2 and 256, got {levels}")
        self.levels = levels

    def _calculate_glcm(self, image):
        """Calculate the normalised Gray Level Co-occurrence Matrix per offset."""
        levels = self.levels
        # Quantize the image to reduce the number of intensity values;
        # digitize puts 255 past the last edge, so clip it into the top level
        bins = np.linspace(0, 255, levels + 1)
        lut = np.clip(np.digitize(np.arange(256), bins) - 1, 0, levels - 1)
        quantized = cv2.LUT(image, lut.astype(np.uint8))
        rows, cols = quantized.shape

        glcm = np.zeros((levels, levels, len(self.distances), len(self.angles)))
        for i, distance in enumerate(self.distances):
            for j, angle in enumerate(self.angles):
                dx = round(distance * np.cos(angle))
                dy = round(distance * np.sin(angle))
                # Pairs (q[r, c], q[r - dy, c - dx]) with both pixels in the tile
                ys, ye = max(dy, 0), rows + min(dy, 0)
                xs, xe = max(dx, 0), cols + min(dx, 0)
                pairs = [
                    quantized[ys:ye, xs:xe],
                    quantized[ys - dy : ye - dy, xs - dx : xe - dx],
                ]
                counts = cv2.calcHist(
                    pairs, [0, 1], None, [levels, levels], [0, levels, 0, levels]
                ).astype(np.float64)
                glcm[:, :, i, j] = counts / counts.sum()

        return glcm

    def _calculate_haralick_features(self, glcm):
        """Calculate Haralick features from GLCM."""
        features = {}

        # Average over all GLCMs
        mean_glcm = glcm.mean(axis=(2, 3))

        # Calculate features
        # 1. Energy (Angular Second Moment)
        features["energy"] = np.sum(mean_glcm**2)

        # 2. Contrast
        indices = np.arange(self.levels)
        i, j = np.meshgrid(indices, indices)
        features["contrast"] = np.sum(mean_glcm * ((i - j) ** 2))

        # 3. Homogeneity (Inverse Difference Moment)
        features["homogeneity"] = np.sum(mean_glcm / (1 + (i - j) ** 2))

        # 4. Correlation
        pi = mean_glcm.sum(axis=1)
        pj = mean_glcm.sum(axis=0)

        mu_i = np.sum(indices * pi)
        mu_j = np.sum(indices * pj)

        sigma_i = np.sqrt(np.sum(pi * ((indices - mu_i) ** 2)))
        sigma_j = np.sqrt(np.sum(pj * ((indices - mu_j) ** 2)))

        if sigma_i > 0 and sigma_j > 0:
            corr_num = np.sum(mean_glcm * np.outer(indices - mu_i, indices - mu_j))
            features["correlation"] = corr_num / (sigma_i * sigma_j)
        else:
            features["correlation"] = 0

        # 5. Entropy
        non_zero = mean_glcm > 0
        if np.any(non_zero):
            features["entropy"] = -np.sum(
                mean_glcm[non_zero] * np.log2(mean_glcm[non_zero])
            )
        else:
            features["entropy"] = 0

        return features

    def _func(self, image):
        # Convert to grayscale if the image is in color
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Calculate GLCM
        glcm = self._calculate_glcm(gray_image)

        # Calculate Haralick features
        features = self._calculate_haralick_features(glcm)

        # Prefix feature names with 'texture_'
        scores = {f"texture_{k}": v for k, v in features.items()}

        return scores


#: Stain presets from scikit-image, as ``(RGB-to-stain matrix, stain names)``:
#: ``hed`` is Ruifrok and Johnston's; ``hdx`` is Landini's hematoxylin and DAB,
#: with their cross product as the residual.
# ponytail: scikit-image's other presets (fgx, bex, rbd, ...) are one line each
_STAINS = {
    "hed": (hed_from_rgb, ("hematoxylin", "eosin", "dab")),
    "hdx": (hdx_from_rgb, ("hematoxylin", "dab", "residual")),
}

#: The optical density ``skimage.color.separate_stains`` gives each uint8 value
_OD = np.log(np.maximum(np.arange(256) / 255, 1e-6)) / np.log(1e-6)


def _unmix(stain):
    """``(RGB-to-stain matrix, names)`` for a preset or a dict of stain vectors."""
    if isinstance(stain, str):
        if stain not in _STAINS:
            raise ValueError(
                f"Unknown stain {stain!r}: use one of {list(_STAINS)}, or a dict "
                "of three {name: RGB optical-density vector} entries."
            )
        return _STAINS[stain]
    if len(stain) != 3:
        raise ValueError(
            "Unmixing needs three stain vectors; for two stains, add a residual "
            "as the third, e.g. np.cross(first, second)."
        )
    names, vectors = zip(*stain.items())
    # The vectors are rows like scikit-image's rgb_from_hed; unmixing inverts them
    return np.linalg.inv(np.asarray(vectors, dtype=np.float64)), names


@register(
    key="color_deconvolution",
    task=ModelTask.cv_feature,
    description=(
        "Mean amount of each stain in a tile (hematoxylin, eosin, DAB), "
        "by color deconvolution"
    ),
    bib_key="Ruifrok2001-cd",
    paper_url="https://pubmed.ncbi.nlm.nih.gov/11531144/",
)
class ColorDeconvolution(_CVFeatures):
    """
    Measure the stains in a tile by color deconvolution.

    Each pixel's optical density is unmixed into the amounts of three stains,
    given their RGB absorption vectors, and each amount is averaged over the
    tile. On an IHC slide, ``dab`` measures the chromogen.

    The amounts are in the units of ``skimage.color.separate_stains``, so the
    default equals ``skimage.color.rgb2hed`` averaged over the tile: a pixel
    holding amounts ``s`` of stains with RGB vectors ``V`` transmits
    ``10 ** (-6 * s @ V)`` of the light. Negative amounts are clipped to zero
    in each pixel.

    An absent stain does not read zero, since real pixels are never exact
    mixtures of the stain vectors: an H&E slide reads a ``dab`` of about 0.02
    to 0.05, so compare tiles with each other or with a negative control.
    White background adds zero, so a tile with less tissue has a lower mean;
    for statistics over the tissue alone, use ``ColorDeconvolutionMap``. The
    presets are published stain vectors. Stain colors vary with the staining
    batch and the scanner, so pass measured vectors when the numbers matter.

    .. code-block:: python

       >>> zs.tl.tile_prediction(wsi, "color_deconvolution")
       >>> tiles = wsi["tiles"]
       >>> tiles[tiles["dab"] > tiles["dab"].quantile(0.9)]

    Parameters
    ----------
    stain : str or dict, default: "hed"
        The stains to unmix. ``"hed"`` gives ``hematoxylin``, ``eosin`` and
        ``dab``; ``"hdx"`` gives ``hematoxylin``, ``dab`` and a ``residual``,
        for IHC without eosin. A dict of exactly three ``{name: vector}``
        entries unmixes custom stains: each vector is the stain's optical
        density in red, green and blue, like a row of
        ``skimage.color.rgb_from_hed``. The names become the columns.
    """

    columns = ("hematoxylin", "eosin", "dab")

    def __init__(self, stain: str | dict = "hed"):
        self._unmixing, self.columns = _unmix(stain)

    def _func(self, image):
        # separate_stains through lookups: per pixel, od @ unmixing (~8x faster)
        stains = cv2.transform(cv2.LUT(image, _OD), self._unmixing.T)
        return dict(zip(self.columns, cv2.mean(np.maximum(stains, 0))[:3]))


class _Unmix(torch.nn.Module):
    """``skimage.color.separate_stains`` as tensor ops, so it runs on any device."""

    def __init__(self, unmixing):
        super().__init__()
        self.register_buffer("unmixing", torch.as_tensor(unmixing, dtype=torch.float32))

    def forward(self, image):
        # image: [B, 3, H, W] in [0, 1]. Optical density (black clamped so its
        # log is finite), then unmixing, with negative amounts clipped
        od = torch.log(image.clamp_min(1e-6)) / math.log(1e-6)
        return torch.einsum("bchw,cs->bshw", od, self.unmixing).clamp_min(0)


@register(
    key="color_deconvolution_map",
    task=ModelTask.cv_feature,
    description="Per-pixel amount of each stain, by color deconvolution",
    bib_key="Ruifrok2001-cd",
    paper_url="https://pubmed.ncbi.nlm.nih.gov/11531144/",
)
class ColorDeconvolutionMap(MarkerMapModel):
    """
    Map the stains of a tile by color deconvolution, one value per pixel.

    The per-pixel amounts that ``ColorDeconvolution`` averages over a tile, in
    the same units. Run it with ``virtual_stain`` to store the stains as one
    image, a channel each, as float32 at the tile resolution: use a coarse tile
    set for a whole-slide map. Parts of the slide no tile covers stay zero.

    .. code-block:: python

       >>> zs.tl.virtual_stain(wsi, "color_deconvolution_map", image_key="stains")
       >>> wsi.images["stains"]

    Parameters
    ----------
    stain : str or dict, default: "hed"
        The stains to unmix, as for ``ColorDeconvolution``.
    """

    channel_names = ("hematoxylin", "eosin", "dab")

    def __init__(self, stain: str | dict = "hed"):
        unmixing, self.channel_names = _unmix(stain)
        self.model = _Unmix(unmixing).eval()

    def get_transform(self):
        from torchvision.transforms.v2 import Compose, ToDtype, ToImage

        return Compose([ToImage(), ToDtype(torch.float32, scale=True)])

    @torch.inference_mode()
    def predict(self, image):
        return self.model(image)
