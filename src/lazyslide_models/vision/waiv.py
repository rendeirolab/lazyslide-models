import torch

from lazyslide_models._model_registry import register
from lazyslide_models._utils import hf_access
from lazyslide_models.base import DenseTokens, ImageModel, ModelTask


class WaivEncoder(ImageModel):
    """Base class for the Waiv robustness-finetuned pathology encoders.

    Both models ship the same ``modeling_finetuned_encoder.py`` remote-code
    wrapper (``FinetunedEncoderModel``), which exposes an inner DINOv2 through
    a uniform ``last_hidden_state`` / ``pooler_output`` interface and stores
    its expected input normalisation in ``config.pixel_mean`` /
    ``config.pixel_std``.
    """

    _hf_hub_id: str

    def __init__(self, model_path=None, token=None):
        from transformers import AutoModel

        with hf_access(self._hf_hub_id):
            self.model = AutoModel.from_pretrained(
                self._hf_hub_id,
                trust_remote_code=True,
                token=token,
            )
        self.model.eval()

    def get_transform(self):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import (
            CenterCrop,
            Compose,
            Normalize,
            Resize,
            ToDtype,
            ToImage,
        )

        # Read the normalisation off the config rather than hardcoding it —
        # mascaret uses 0.5/0.5/0.5 while phaet uses the ImageNet statistics.
        config = self.model.config
        return Compose(
            [
                ToImage(),
                Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
                CenterCrop(224),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=tuple(config.pixel_mean), std=tuple(config.pixel_std)),
            ]
        )

    @torch.inference_mode()
    def encode_image_dense(self, image) -> DenseTokens:
        hidden = self.model(pixel_values=image).last_hidden_state
        return DenseTokens(cls_token=hidden[:, 0], patch_tokens=hidden[:, 1:])

    @torch.inference_mode()
    def encode_image(self, image):
        # `pooler_output` is the L2-normalised CLS token — the feature vector
        # recommended by both model cards.
        return self.model(pixel_values=image).pooler_output


shared_info = dict(
    is_gated=True,
    task=ModelTask.vision,
    license="Waiv custom non-commercial license",
    description="Robustifying pathology foundation models via fine-tuning",
    commercial=False,
    paper_url="https://doi.org/10.48550/arXiv.2607.22861",
    bib_key="Filiot2026-rb",
)


@register(
    key="mascaret",
    **shared_info,
    license_url="https://huggingface.co/wearewaiv/mascaret/blob/main/LICENSE.pdf",
    hf_url="https://huggingface.co/wearewaiv/mascaret",
    param_size="1.14B",
    encode_dim=1536,
    flops="582.55G",
)
class Mascaret(WaivEncoder):
    """Mascaret: DINOv2-giant robustness-finetuned from ``kaiko-ai/midnight``."""

    _hf_hub_id = "wearewaiv/mascaret"


@register(
    key="phaet",
    **shared_info,
    license_url="https://huggingface.co/wearewaiv/phaet/blob/main/LICENSE.pdf",
    hf_url="https://huggingface.co/wearewaiv/phaet",
    param_size="303.4M",
    encode_dim=1024,
    flops="119.29G",
)
class Phaet(WaivEncoder):
    """Phaet: DINOv2-large robustness-finetuned from ``owkin/phikon-v2``."""

    _hf_hub_id = "wearewaiv/phaet"
