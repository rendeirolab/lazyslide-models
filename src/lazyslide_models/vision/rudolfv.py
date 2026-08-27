import torch

from lazyslide_models._model_registry import register
from lazyslide_models._utils import hf_access
from lazyslide_models.base import DenseTokens, ImageModel, ModelTask


class RudolfV2Base(ImageModel):
    """Base class for the RudolfV-2 family of pathology foundation models.

    All three variants ship the same ``modeling_rudolfv.py`` remote code and a
    standard ``AutoImageProcessor``, so preprocessing is delegated to the
    processor (``get_transform()`` returns ``None``).
    """

    _hf_hub_id: str

    def __init__(self, model_path=None, token=None):
        from transformers import AutoImageProcessor, AutoModel

        with hf_access(self._hf_hub_id):
            self.model = AutoModel.from_pretrained(
                self._hf_hub_id,
                trust_remote_code=True,
                token=token,
            )
            self.img_processor = AutoImageProcessor.from_pretrained(
                self._hf_hub_id,
                token=token,
            )
        self.model.eval()

        # ViT-*/8 at 224 px yields a 28x28 = 784 patch grid, plus 1 CLS and
        # 8 register tokens = 793 output tokens. All three variants carry the
        # registers; the count is read off the config rather than hardcoded.
        self.num_prefix_tokens = 1 + getattr(
            self.model.config, "num_register_tokens", 0
        )

    def get_transform(self):
        return None

    @torch.inference_mode()
    def encode_image_dense(self, image) -> DenseTokens:
        inputs = self.img_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        hidden = self.model(**inputs).last_hidden_state
        return DenseTokens(
            cls_token=hidden[:, 0],
            patch_tokens=hidden[:, self.num_prefix_tokens :],
        )

    @torch.inference_mode()
    def encode_image(self, image):
        return self.encode_image_dense(image).cls_token


shared_info = {
    "is_gated": True,
    "task": ModelTask.vision,
    "license": "CC-BY-NC-ND-4.0 with supplementary terms",
    "description": (
        "A family of robust and efficient open-weights pathology foundation models"
    ),
    "commercial": False,
    "bib_key": "Milbich2026-rv",
}


@register(
    key="rudolfv2-s",
    **shared_info,
    hf_url="https://huggingface.co/Aignostics/RudolfV-2-S",
    param_size="21.4M",
    encode_dim=384,
    flops="33.79G",
)
class RudolfV2S(RudolfV2Base):
    """RudolfV-2-S: ViT-S/8 distilled from RudolfV-2."""

    _hf_hub_id = "Aignostics/RudolfV-2-S"


@register(
    key="rudolfv2-b",
    **shared_info,
    hf_url="https://huggingface.co/Aignostics/RudolfV-2-B",
    param_size="85.2M",
    encode_dim=768,
    flops="134.94G",
)
class RudolfV2B(RudolfV2Base):
    """RudolfV-2-B: ViT-B/8 distilled from RudolfV-2."""

    _hf_hub_id = "Aignostics/RudolfV-2-B"


@register(
    key="rudolfv2",
    **shared_info,
    hf_url="https://huggingface.co/Aignostics/RudolfV-2",
    param_size="1.13B",
    encode_dim=1536,
    flops="1796.55G",
)
class RudolfV2(RudolfV2Base):
    """RudolfV-2: ViT-g/8 pretrained on 300,000 whole slide images."""

    _hf_hub_id = "Aignostics/RudolfV-2"
