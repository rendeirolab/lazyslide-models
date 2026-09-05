import os

import torch

from lazyslide_models._model_registry import register
from lazyslide_models._utils import hf_access
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
    key="gigapath-flash",
    is_gated=True,
    task=ModelTask.vision,
    license="Apache 2.0",
    license_url="https://huggingface.co/prov-gigapath/prov-gigapath-flash/blob/main/LICENSE",
    description="An efficient whole-slide foundation model for digital pathology",
    commercial=False,
    hf_url="https://huggingface.co/prov-gigapath/prov-gigapath-flash",
    github_url="https://github.com/prov-gigapath/prov-gigapath",
    paper_url="https://doi.org/10.48550/arXiv.2607.18218",
    bib_key="Usuyama2026-gf",
    param_size="21.7M",
    encode_dim=384,
    flops="8.48G",
)
class GigaPathFlash(TimmViTModel):
    """GigaPath-Flash tile encoder: a DINOv2-small ViT-S/16 with SwiGLU FFN."""

    _hf_hub_id = "prov-gigapath/prov-gigapath-flash"

    def __init__(self, model_path=None, token=None):
        from huggingface_hub import hf_hub_download
        from timm.layers import SwiGLUPacked

        # The checkpoint is a timm-style `pytorch_model.bin`, but its
        # `config.json` names the custom architecture
        # `gigapath_tile_enc_dinov2s`, whose upstream `@register_model` shim
        # ignores the `pretrained` flag. Loading via
        # `timm.create_model("hf_hub:...", pretrained=True)` would therefore
        # silently return a randomly initialised model. Build the backbone
        # explicitly and load the released weights ourselves instead.
        super().__init__(
            "vit_small_patch16_224",
            pretrained=False,
            img_size=224,
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=2048 / 384.0,  # SwiGLU: fc1 -> 2048, fc2 <- 1024
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            init_values=1e-5,  # LayerScale
            global_pool="token",
            class_token=True,
            reg_tokens=0,
        )

        with hf_access(self._hf_hub_id):
            weights_file = model_path or hf_hub_download(
                self._hf_hub_id,
                "pytorch_model.bin",
                token=token,
            )

        self.model.load_state_dict(torch.load(weights_file, map_location="cpu"))
        self.model.eval()


class _GigaPathSlideEncoder(SlideEncoderModel):
    """Shared loader for the LongNet slide encoders of the GigaPath family.

    Subclasses set ``_hf_hub_id``, ``_slide_arch`` and ``_in_chans``.
    """

    _hf_hub_id: str
    _slide_arch: str
    _in_chans: int

    def __init__(self, model_path=None, token=None):
        import warnings

        from huggingface_hub import constants, login

        super().__init__()

        # Upstream DilatedAttention hard-asserts `flash_attention=True` and
        # has no CPU/MPS code path — only NVIDIA GPUs with `flash-attn`
        # installed can run the forward pass.
        if not torch.cuda.is_available():
            warnings.warn(
                f"{type(self).__name__} requires a CUDA GPU with the "
                "`flash-attn` package installed; CPU/MPS execution is not "
                "supported by the upstream model.",
                RuntimeWarning,
                stacklevel=2,
            )

        if token is not None:
            login(token)

        try:
            from gigapath.slide_encoder import create_model
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"Please install gigapath to use {type(self).__name__}. "
                "Try pip install git+https://github.com/prov-gigapath/prov-gigapath"
            )

        # `create_model` downloads to `<local_dir>/slide_encoder.pth` with
        # `force_download=True`, so give every variant its own directory —
        # otherwise the GigaPath and GigaPath-Flash checkpoints clobber each
        # other when both are used in one session.
        with hf_access(self._hf_hub_id):
            self.model = create_model(
                f"hf_hub:{self._hf_hub_id}",
                self._slide_arch,
                self._in_chans,
                local_dir=os.path.join(
                    constants.HF_HOME, self._hf_hub_id.replace("/", "--")
                ),
            )

    def encode_slide(self, embeddings, coords=None, **kwargs) -> SlideEncodeOutput:
        # Upstream DilatedAttention always calls flash_attn, which only
        # accepts fp16/bf16. Tests and callers pass fp32 tile embeddings.
        # Same pattern as Prism2: autocast on CUDA, no-op elsewhere.
        if torch.is_tensor(embeddings) and embeddings.is_cuda:
            with torch.autocast("cuda", torch.bfloat16):
                outcomes = self.model(embeddings, coords)
        else:
            outcomes = self.model(embeddings, coords)
        # `LongNetViT.forward` returns a *list* of per-layer outcomes; with
        # `all_layer_embed=False` (the default) it holds a single [B, D]
        # tensor.
        return {"embeddings": outcomes[0].squeeze()}


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
class GigaPathSlideEncoder(_GigaPathSlideEncoder):
    _hf_hub_id = "prov-gigapath/prov-gigapath"
    _slide_arch = "gigapath_slide_enc12l768d"
    _in_chans = 1536


@register(
    key="gigapath-flash-slide-encoder",
    is_gated=True,
    task=ModelTask.slide_encoder,
    license="Apache 2.0",
    license_url="https://huggingface.co/prov-gigapath/prov-gigapath-flash/blob/main/LICENSE",
    description="An efficient whole-slide foundation model for digital pathology",
    commercial=False,
    hf_url="https://huggingface.co/prov-gigapath/prov-gigapath-flash",
    github_url="https://github.com/prov-gigapath/prov-gigapath",
    paper_url="https://doi.org/10.48550/arXiv.2607.18218",
    bib_key="Usuyama2026-gf",
    param_size="21.5M",
    vision_encoder="gigapath-flash",
)
class GigaPathFlashSlideEncoder(_GigaPathSlideEncoder):
    _hf_hub_id = "prov-gigapath/prov-gigapath-flash"
    _slide_arch = "gigapath_slide_enc12l384d"
    _in_chans = 384
