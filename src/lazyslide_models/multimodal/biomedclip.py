import numpy as np
import torch
from PIL import Image

from lazyslide_models._model_registry import register
from lazyslide_models.base import DenseTokens, ImageTextModel, ModelTask

_HF_HUB_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


@register(
    key="biomedclip",
    task=ModelTask.multimodal,
    license="MIT",
    description="A biomedical VLP foundation model pretrained on PMC-15M image-text pairs",
    commercial=True,
    hf_url=f"https://huggingface.co/{_HF_HUB_ID}",
    github_url="https://github.com/microsoft/BiomedCLIP_data_pipeline",
    paper_url="https://doi.org/10.1056/AIoa2400640",
    bib_key="Zhang2024-bc",
    encode_dim=512,
)
class BiomedCLIP(ImageTextModel):
    def __init__(self, model_path=None, token=None):
        try:
            from open_clip import create_model_from_pretrained, get_tokenizer
        except ImportError:
            raise ImportError(
                "open_clip is not installed. You can install it using "
                "`pip install open_clip_torch`."
            )

        self.model, self.processor = create_model_from_pretrained(
            f"hf-hub:{_HF_HUB_ID}"
        )
        self.model.eval()
        self.tokenizer = get_tokenizer(f"hf-hub:{_HF_HUB_ID}")
        self._context_length = 256

    def get_transform(self):
        return None

    def _prepare_image(self, image):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image = self.processor(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        device = next(self.model.parameters()).device
        return image.to(device)

    @torch.inference_mode()
    def encode_image_dense(self, image):
        image = self._prepare_image(image)
        trunk = self.model.visual.trunk
        # forward_features returns all tokens (CLS + patches) before the head
        x = trunk.forward_features(image)
        n = trunk.num_prefix_tokens  # typically 1 (CLS)
        return DenseTokens(cls_token=x[:, 0], patch_tokens=x[:, n:])

    @torch.inference_mode()
    def encode_image(self, image):
        image = self._prepare_image(image)
        return self.model.encode_image(image, normalize=False)

    @torch.inference_mode()
    def encode_text(self, text):
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(text, context_length=self._context_length).to(device)
        text_features = self.model.encode_text(tokens, normalize=True)
        return text_features
