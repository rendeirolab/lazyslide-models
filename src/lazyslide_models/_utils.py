import inspect
import os
import sys
from contextlib import contextmanager
from types import FrameType, ModuleType

import torch


def ensure_transformers_compat():
    """Shim removed ``transformers`` sub-modules so that third-party remote
    code (loaded via ``trust_remote_code=True``) does not crash on import.

    ``transformers.onnx`` was removed in transformers 5.x.  Several HuggingFace
    model repos (e.g. histai/hibou-L) still import ``OnnxConfig`` from it at
    module level.  Injecting a lightweight stub into ``sys.modules`` lets those
    imports succeed without vendoring the remote code.
    """
    if "transformers.onnx" not in sys.modules:
        try:
            # If the real module exists (transformers < 5), leave it alone.
            import transformers.onnx  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            stub = ModuleType("transformers.onnx")
            # Provide a dummy OnnxConfig so `from transformers.onnx import OnnxConfig` works.
            stub.OnnxConfig = type("OnnxConfig", (), {})
            sys.modules["transformers.onnx"] = stub


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
