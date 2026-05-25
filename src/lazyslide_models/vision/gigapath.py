from lazyslide_models._model_registry import register
from lazyslide_models.base import (
    ModelTask,
    SlideEncodeOutput,
    SlideEncoderModel,
    TimmViTModel,
)


@register(
    key="gigapath",
    is_gated=True,
    task=ModelTask.vision,
    license="Apache 2.0 with conditions",
    description="A whole-slide foundation model for digital pathology",
    commercial=False,
    hf_url="https://huggingface.co/prov-gigapath/prov-gigapath",
    github_url="https://github.com/prov-gigapath/prov-gigapath",
    paper_url="https://doi.org/10.1038/s41586-024-07441-w",
    bib_key="Xu2024-td",
    param_size="1.13B",
    encode_dim=1536,
)
class GigaPath(TimmViTModel):
    def __init__(self, model_path=None, token=None):
        # Version check
        import timm

        try:
            from packaging import version

            timm_version = version.parse(timm.__version__)
            minimum_version = version.parse("1.0.3")
            if timm_version < minimum_version:
                raise ImportError(
                    f"Gigapath needs timm >= 1.0.3. You have version {timm_version}."
                    f"Run `pip install --upgrade timm` to install the latest version."
                )
        # If packaging is not installed, skip the version check
        except ModuleNotFoundError:
            pass

        super().__init__("hf_hub:prov-gigapath/prov-gigapath", token=token)


@register(
    key="gigapath-slide-encoder",
    is_gated=True,
    task=ModelTask.slide_encoder,
    license="Apache 2.0 with conditions",
    description="A whole-slide foundation model for digital pathology",
    commercial=False,
    hf_url="https://huggingface.co/prov-gigapath/prov-gigapath",
    github_url="https://github.com/prov-gigapath/prov-gigapath",
    paper_url="https://doi.org/10.1038/s41586-024-07441-w",
    bib_key="Xu2024-td",
    vision_encoder="gigapath",
)
class GigaPathSlideEncoder(SlideEncoderModel):
    def __init__(self, model_path=None, token=None):
        import warnings

        import torch
        from huggingface_hub import constants, login

        super().__init__()

        # Upstream DilatedAttention hard-asserts `flash_attention=True` and
        # has no CPU/MPS code path — only NVIDIA GPUs with `flash-attn`
        # installed can run the forward pass.
        if not torch.cuda.is_available():
            warnings.warn(
                "gigapath-slide-encoder requires a CUDA GPU with the "
                "`flash-attn` package installed; CPU/MPS execution is not "
                "supported by the upstream model.",
                RuntimeWarning,
                stacklevel=2,
            )

        if token is not None:
            login(token)

        try:
            # Monkey-patch: upstream `get_optimal_segment_length` builds the
            # segment list via numpy then `str(list(...))`. Under NumPy >= 2,
            # the repr becomes `[np.int64(1024), ...]`, which the downstream
            # `eval(self.segment_length)` in torchscale cannot resolve
            # (NameError: name 'np' is not defined). Cast to plain ints.
            import numpy as np
            from gigapath.slide_encoder import LongNetViT, create_model

            def _patched_get_optimal_segment_length(
                self, max_wsi_size: int = 262144, tile_size: int = 256
            ) -> str:
                max_seq_len = (max_wsi_size // tile_size) ** 2
                segment_length = np.linspace(
                    np.log2(1024), int(np.log2(max_seq_len)), 5
                )
                segment_length = np.power(2, segment_length).astype(int).tolist()
                return str([int(x) for x in segment_length])

            LongNetViT.get_optimal_segment_length = _patched_get_optimal_segment_length

            model = create_model(
                "hf_hub:prov-gigapath/prov-gigapath",
                "gigapath_slide_enc12l768d",
                1536,
                local_dir=constants.HF_HOME,
            )
            self.model = model
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install gigapath to use this GigaPathSlideEncoder."
                "Try pip install git+https://github.com/prov-gigapath/prov-gigapath"
            )

    def encode_slide(self, embeddings, coords=None, **kwargs) -> SlideEncodeOutput:
        return {"embeddings": self.model(embeddings, coords).squeeze()}
