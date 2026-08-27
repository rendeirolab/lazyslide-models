import inspect
import os
from contextlib import contextmanager
from types import FrameType

import torch


def require_transformers_below_5(model_name: str, reason: str) -> None:
    """Raise if the installed ``transformers`` is 5.0 or newer.

    Some upstream HuggingFace repos ship ``trust_remote_code`` modules that
    import APIs removed in transformers 5.0. Because the failure happens
    inside third-party remote code, it cannot be patched from here — the only
    fix is to pin transformers. Raise up front with the real cause rather than
    letting a bare ``ModuleNotFoundError`` surface from deep inside the
    downloaded module.

    Parameters
    ----------
    model_name : str
        Registry key, used in the error message.
    reason : str
        One sentence naming what the remote code needs, e.g. which removed
        module it imports.
    """
    from importlib.metadata import version

    from packaging.version import Version

    installed = version("transformers")
    if Version(installed) >= Version("5.0"):
        raise ImportError(
            f"'{model_name}' requires transformers < 5.0 (installed: "
            f"{installed}). {reason} This is upstream remote code and cannot "
            f"be patched by lazyslide-models. Pin with: "
            f"pip install 'transformers<5'"
        )


def _fake_class(name, deps, inject=""):
    def init(self, *args, **kwargs):
        raise ImportError(
            f"To use {name}, you need to install {', '.join(deps)}."
            f"{inject}"
            "Please restart the kernel after installation."
        )

    # Dynamically create the class
    new_class = type(name, (object,), {"__init__": init})

    return new_class


@contextmanager
def hf_access(name: str):
    """
    Context manager for Hugging Face access.
    """
    from huggingface_hub.errors import GatedRepoError

    try:
        yield
    except GatedRepoError as e:
        raise GatedRepoError(
            f"You don't have access to {name}. Please request access to the model on HuggingFace. "
            "After access granted, please login to HuggingFace with huggingface-cli on this machine "
            "with a token that has access to this model. "
            "You may also pass token as an argument in LazySlide, however, this is not recommended."
        ) from e


def get_default_transform(img_size=(224, 224)):
    """The default transform for the model."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import (
        CenterCrop,
        Compose,
        Normalize,
        Resize,
        ToDtype,
        ToImage,
    )

    transforms = [
        ToImage(),
        ToDtype(dtype=torch.float32, scale=True),
        Resize(
            size=img_size,
            interpolation=InterpolationMode.BICUBIC,
            max_size=None,
            antialias=True,
        ),
        CenterCrop(img_size),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return Compose(transforms)


def find_stack_level() -> int:
    """
    Find the first place in the stack that is not inside pandas
    (tests notwithstanding).
    """

    import pandas as pd

    pkg_dir = os.path.dirname(pd.__file__)
    test_dir = os.path.join(pkg_dir, "tests")

    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame: FrameType | None = inspect.currentframe()
    try:
        n = 0
        while frame:
            filename = inspect.getfile(frame)
            if filename.startswith(pkg_dir) and not filename.startswith(test_dir):
                frame = frame.f_back
                n += 1
            else:
                break
    finally:
        # See note in
        # https://docs.python.org/3/library/inspect.html#inspect.Traceback
        del frame
    return n
