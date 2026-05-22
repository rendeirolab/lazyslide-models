import torch

from lazyslide_models._model_registry import register
from lazyslide_models.base import DenseTokens, ImageTextModel, ModelTask

# CLIP/OpenAI normalization stats — shared by all QuiltNet variants
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_CLIP_SIZE = 224


class QuiltNet(ImageTextModel):
    """Base class for QuiltNet CLIP variants."""

    _hf_hub_id: str

    def __init__(self, model_path=None, token=None):
        try:
            from open_clip import create_model_from_pretrained, get_tokenizer
        except ImportError:
            raise ImportError(
                "open_clip is not installed. You can install it using "
                "`pip install open_clip_torch`."
            )

        self.model, _ = create_model_from_pretrained(f"hf-hub:{self._hf_hub_id}")
        self.model.eval()
        self.tokenizer = get_tokenizer(f"hf-hub:{self._hf_hub_id}")

    def get_transform(self):
        from torchvision.transforms.v2 import (
            CenterCrop,
            Compose,
            InterpolationMode,
            Normalize,
            Resize,
            ToDtype,
            ToImage,
        )

        return Compose(
            [
                ToImage(),
                Resize(
                    _CLIP_SIZE,
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                CenterCrop(_CLIP_SIZE),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
            ]
        )

    @torch.inference_mode()
    def encode_image_dense(self, image):
        visual = self.model.visual

        if hasattr(visual, "trunk"):
            # TimmModel backend (e.g. QuiltNet-B-16-PMB)
            trunk = visual.trunk
            x = trunk.forward_features(image)
            n = trunk.num_prefix_tokens
        else:
            # open_clip VisionTransformer backend
            x = visual._embeds(image)
            x = visual.transformer(x)
            x = visual.ln_post(x)
            n = 1

        return DenseTokens(cls_token=x[:, 0], patch_tokens=x[:, n:])

    @torch.inference_mode()
    def encode_image(self, image):
        return self.model.encode_image(image, normalize=False)

    @torch.inference_mode()
    def encode_text(self, text):
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(text).to(device)
        return self.model.encode_text(tokens, normalize=True)


shared_info = dict(
    task=ModelTask.multimodal,
    license="MIT",
    description="Quilt-1M: histopathology vision-language model trained on 1M image-text pairs",
    commercial=True,
    github_url="https://github.com/wisdomikezogwo/quilt1m",
    paper_url="https://doi.org/10.48550/arXiv.2306.11207",
    bib_key="Ikezogwo2023-qn",
    encode_dim=512,
)


@register(
    key="quiltnet-b32",
    **shared_info,
    hf_url="https://huggingface.co/wisdomik/QuiltNet-B-32",
)
class QuiltNetB32(QuiltNet):
    _hf_hub_id = "wisdomik/QuiltNet-B-32"


@register(
    key="quiltnet-b16",
    **shared_info,
    hf_url="https://huggingface.co/wisdomik/QuiltNet-B-16",
)
class QuiltNetB16(QuiltNet):
    _hf_hub_id = "wisdomik/QuiltNet-B-16"


@register(
    key="quiltnet-b16-pmb",
    **shared_info,
    hf_url="https://huggingface.co/wisdomik/QuiltNet-B-16-PMB",
)
class QuiltNetB16PMB(QuiltNet):
    """ViT-B/16 image tower with PubMedBERT text tower."""

    _hf_hub_id = "wisdomik/QuiltNet-B-16-PMB"
