from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    NamedTuple,
    Protocol,
    Self,
    Tuple,
    TypedDict,
    runtime_checkable,
)

import torch
from torch.utils.flop_counter import FlopCounterMode

from ._repr import model_repr_html
from ._utils import get_default_transform, hf_access

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class ModelTask(Enum):
    vision = "vision"
    segmentation = "segmentation"
    multimodal = "multimodal"
    slide_encoder = "slide_encoder"
    tile_prediction = "tile_prediction"
    feature_prediction = "feature_prediction"
    style_transfer = "style_transfer"
    cv_feature = "cv_feature"
    image_generation = "image_generation"


@dataclass(frozen=True)
class InputConstraint:
    """Declarative spatial input constraints for a model.

    Attach via ``@register(input_constraint=InputConstraint(...))`` so that
    tests and downstream tooling can inspect requirements at class level
    without instantiating the model.

    Parameters
    ----------
    min : int, optional
        Minimum spatial size (inclusive).
    max : int, optional
        Maximum spatial size (inclusive).
    divisible_by : int, optional
        Spatial size must be divisible by this value (e.g. ViT patch size).
    """

    min: int | None = None
    max: int | None = None
    divisible_by: int | None = None

    def validate(self, size: int, model_name: str = "Model") -> None:
        """Raise ``ValueError`` if *size* violates any constraint."""
        if self.min is not None and size < self.min:
            raise ValueError(
                f"{model_name} requires input size >= {self.min}, got {size}"
            )
        if self.max is not None and size > self.max:
            raise ValueError(
                f"{model_name} requires input size <= {self.max}, got {size}"
            )
        if self.divisible_by is not None and size % self.divisible_by != 0:
            raise ValueError(
                f"{model_name} requires input size divisible by "
                f"{self.divisible_by}, got {size}"
            )

    def filter_sizes(self, sizes: list[int]) -> list[int]:
        """Return the subset of *sizes* that satisfy all constraints."""
        valid = []
        for s in sizes:
            if self.min is not None and s < self.min:
                continue
            if self.max is not None and s > self.max:
                continue
            if self.divisible_by is not None and s % self.divisible_by != 0:
                continue
            valid.append(s)
        return valid

    @property
    def default_size(self) -> int:
        """A reasonable default size for input generation."""
        return self.min or 224


class DenseTokens(NamedTuple):
    """Dense output from a ViT encoder.

    Attributes
    ----------
    cls_token : torch.Tensor
        CLS token embedding, shape ``[B, D]``.
    patch_tokens : torch.Tensor
        Patch token embeddings, shape ``[B, N_patches, D]``.
    """

    cls_token: Any  # torch.Tensor [B, D]
    patch_tokens: Any  # torch.Tensor [B, N_patches, D]


class SegmentationOutput(NamedTuple):
    """Segmentation output from a segmentation model.
    Covers both semantic and instance segmentation.

    Attributes
    ----------
    probability_map : torch.Tensor
        Per-class probabilities, shape ``[B, C, H, W]`` (float).
    instance_map : torch.Tensor
        Instance ID map, shape ``[B, H, W]`` (int).
    patch_token_map : torch.Tensor
        Vision token map, shape ``[B, D, Patch_H, Patch_W]``.
        Only set when the model is a ViT.
    classes : tuple of str
        Class names in index order.

    """

    probability_map: Any = None
    instance_map: Any = None
    patch_token_map: Any = None
    classes: Tuple | None = None


class SlideEncodeOutput(TypedDict):
    """Base structured output for slide encoders.

    Runtime value is a plain ``dict``. Model-specific outputs should
    subclass with ``total=False`` to declare optional extra fields, e.g.::

        class PrismSlideEncodeOutput(SlideEncodeOutput, total=False):
            latents: Any

    Attributes
    ----------
    embeddings : torch.Tensor
        Slide-level embedding, shape ``[B, D]`` or ``[D]``.
    """

    embeddings: Any  # torch.Tensor [B, D] or [D]


@runtime_checkable
class ModelBaseProtocol(Protocol):
    model: Any
    name: str

    def get_transform(self) -> Callable | None: ...

    def to(self, device) -> Self: ...

    def try_compile(self, **compile_kws: Any): ...


@runtime_checkable
class ImageModelProtocol(ModelBaseProtocol, Protocol):
    def get_transform(self) -> Callable: ...

    def encode_image(self, image, *args, **kwargs) -> ArrayLike: ...


@runtime_checkable
class ViTModelProtocol(ModelBaseProtocol, Protocol):
    grid_size: Tuple[int, int]
    patch_size: Tuple[int, int]

    def encode_image_dense(self, image, *args, **kwargs) -> DenseTokens: ...


@runtime_checkable
class ImageTextModelProtocol(ImageModelProtocol, Protocol):
    def encode_text(self, text, *args, **kwargs) -> ArrayLike: ...


@runtime_checkable
class SlideEncoderModelProtocol(ModelBaseProtocol, Protocol):
    def encode_slide(
        self, embeddings, coords=None, *args, **kwargs
    ) -> SlideEncodeOutput: ...


@runtime_checkable
class ZeroShotModelProtocol(ModelBaseProtocol, Protocol):
    def score(self, embeddings, prompts, *args, **kwargs) -> ArrayLike: ...


@runtime_checkable
class SegmentationModelProtocol(ModelBaseProtocol, Protocol):
    def segment(self, image, *args, **kwargs) -> SegmentationOutput: ...


@runtime_checkable
class TilePredictionModelProtocol(ModelBaseProtocol, Protocol):
    def predict(self, image, *args, **kwargs) -> Dict[str, Any]: ...


@runtime_checkable
class FeaturePredictionModelProtocol(ModelBaseProtocol, Protocol):
    def predict(self, features, *args, **kwargs) -> Dict[str, Any]: ...


@runtime_checkable
class StyleTransferModelProtocol(ModelBaseProtocol, Protocol):
    def predict(self, image, *args, **kwargs): ...

    def get_channel_names(self) -> Tuple[str, ...]: ...


@runtime_checkable
class ImageGenerationModelProtocol(ModelBaseProtocol, Protocol):
    def generate(self, *args, **kwargs): ...

    def generate_conditionally(self, *args, **kwargs): ...


class ModelBase(ABC):
    model: Any

    def _repr_html_(self) -> str:
        return model_repr_html(self)

    def get_transform(self) -> Callable | None:
        return None

    def to(self, device) -> Self:
        self.model.to(device)
        return self

    def try_compile(self, **compile_kws: Any):
        try:
            self.model = torch.compile(self.model, **compile_kws)
        except Exception:  # noqa
            pass

    def estimate_param_size(self) -> int | None:
        """Count the number of parameters in a model."""
        model = self.model
        if not isinstance(model, torch.nn.Module):
            try:
                # If it's a Coco model, get the underlying PyTorch model
                model = model.model
            except (AttributeError, TypeError):
                return None
        return sum(p.numel() for p in model.parameters())

    def _resolve_method(
        self, model: torch.nn.Module, method: str
    ) -> Tuple[Any, torch.nn.Module] | None:
        """Resolve method path and return (callable, target_model) for FLOPS counting."""
        if "." in method:
            parts = method.split(".")
            obj, target = model, model
            for part in parts[:-1]:
                obj = getattr(obj, part, None)
                if obj is None:
                    return None
                if isinstance(obj, torch.nn.Module):
                    target = obj
            method_obj = getattr(obj, parts[-1], None)
            return (method_obj, target) if method_obj else None

        method_obj = getattr(model, method, None) or getattr(self, method, None)
        return (method_obj, model) if method_obj else None

    def estimate_flops(
        self, *args: Any, method: str = "forward", **kwargs: Any
    ) -> int | None:
        """Count the number of flops in a model."""
        model = self.model
        if not isinstance(model, torch.nn.Module):
            try:
                model = model.model
            except (AttributeError, TypeError):
                return None
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        result = self._resolve_method(model, method)
        if result is None:
            return None

        method_obj, target = result
        is_training = model.training
        model.eval()
        with FlopCounterMode(target, display=False, depth=None) as flop_counter:
            method_obj(*args, **kwargs)
        model.train(is_training)
        return flop_counter.get_total_flops()

    @property
    def name(self) -> str:
        return self.__class__.__name__


class ImageModel(ModelBase):
    def get_transform(self) -> Callable:
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

    @abstractmethod
    def encode_image(self, image: ArrayLike, *args, **kwargs) -> ArrayLike:
        raise NotImplementedError

    def __call__(self, image: ArrayLike, *args, **kwargs):
        return self.encode_image(image)


class TimmModel(ModelBase):
    def __init__(self, name, token=None, compile=False, compile_kws=None, **kwargs):
        import timm
        from huggingface_hub import login

        if token is not None:
            login(token)

        default_kws = {"pretrained": True, "num_classes": 0}
        default_kws.update(kwargs)

        with hf_access(name):
            self.model = timm.create_model(name, **default_kws)
            try:
                self.model.eval()
            except AttributeError:
                pass

        if compile:
            self.try_compile(**(compile_kws or {}))

        if hasattr(self.model, "default_cfg"):
            self.img_size = self.model.default_cfg.get("input_size", (3, 224, 224))[1:]
        else:
            self.img_size = (224, 224)

    def get_transform(self):
        return get_default_transform(self.img_size)

    @torch.inference_mode()
    def encode_image(self, image: torch.Tensor, *args, **kwargs) -> ArrayLike:
        return self.model(image)


class TimmViTModel(TimmModel):
    def __init__(self, name, token=None, compile=False, compile_kws=None, **kwargs):
        super().__init__(
            name, token=token, compile=compile, compile_kws=compile_kws, **kwargs
        )
        from timm.models import VisionTransformer

        self.is_timm_vit = isinstance(self.model, VisionTransformer)
        if not self.is_timm_vit:
            raise ValueError(f"Model {name} is not a timm VisionTransformer")

        patch_embed = self.model.patch_embed
        self.img_size: Tuple[int, int] = patch_embed.img_size
        self.patch_size: Tuple[int, int] = patch_embed.patch_size
        self.grid_size: Tuple[int, int] = patch_embed.grid_size
        self.num_prefix_tokens: int = int(self.model.num_prefix_tokens)

    @torch.inference_mode()
    def encode_image_dense(self, image: torch.Tensor, *args, **kwargs) -> DenseTokens:
        out = self.model.forward_features(image)
        return DenseTokens(
            cls_token=out[:, 0],
            patch_tokens=out[:, self.num_prefix_tokens :],
        )


class SlideEncoderModel(ModelBase):
    """Base class for slide-level encoders.

    ``encode_slide`` must return a :class:`SlideEncodeOutput` (a ``TypedDict``)
    with at least an ``"embeddings"`` key containing the primary slide
    embedding tensor. Models may declare a subclass with ``total=False`` to
    add extra fields (e.g. ``"latents"`` for captioning-ready
    representations).
    """

    @abstractmethod
    def encode_slide(self, embeddings, coords=None, **kwargs) -> SlideEncodeOutput:
        raise NotImplementedError


class ImageTextModel(ImageModel):
    @abstractmethod
    def encode_text(self, text, *args, **kwargs) -> ArrayLike:
        raise NotImplementedError

    def tokenize(self, text, *args, **kwargs):
        raise NotImplementedError


class SegmentationModel(ModelBase):
    def get_transform(self):
        import torch
        from torchvision.transforms.v2 import Compose, Normalize, ToDtype, ToImage

        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @abstractmethod
    def segment(self, image) -> SegmentationOutput:
        raise NotImplementedError


class TilePredictionModel(ModelBase):
    @abstractmethod
    def predict(self, image) -> Dict[str, Any]:
        """The output should always be a dict of numpy arrays
        to allow multiple outputs.
        """
        raise NotImplementedError


class FeaturesPredictionModel(ModelBase):
    features_model_name: str | None = None

    @abstractmethod
    def predict(self, features) -> Dict[str, Any]:
        raise NotImplementedError


class StyleTransferModel(ModelBase):
    @abstractmethod
    def predict(self, image):
        raise NotImplementedError

    @abstractmethod
    def get_channel_names(self) -> Tuple[str, ...]:
        raise NotImplementedError


class ImageGenerationModel(ModelBase):
    def generate(self, *args, **kwargs):
        raise NotImplementedError

    def generate_conditionally(self, *args, **kwargs):
        raise NotImplementedError
