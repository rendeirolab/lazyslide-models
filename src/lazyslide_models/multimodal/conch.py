import warnings

import torch

from lazyslide_models._model_registry import register
from lazyslide_models._utils import find_stack_level, hf_access
from lazyslide_models.base import DenseTokens, ImageTextModel, ModelTask


@register(
    key="conch",
    is_gated=True,
    task=ModelTask.multimodal,
    license="CC-BY-NC-ND-4.0",
    description="CONtrastive learning from Captions for Histopathology (CONCH)",
    commercial=False,
    hf_url="https://huggingface.co/MahmoodLab/conch",
    github_url="https://github.com/mahmoodlab/CONCH",
    paper_url="https://doi.org/10.1038/s41591-024-02856-4",
    bib_key="Lu2024-nu",
    param_size="395.2M",
    encode_dim=512,
    flops="35.08G",
)
class CONCH(ImageTextModel):
    def __init__(self, model_path=None, token=None):
        warnings.warn(
            "As from v0.8.2, Normalization will not be applied to image embedding of CONCH model anymore."
            "A `normalize=True` argument is added to the `text_image_similarity` method."
            "If you only use the image embedding for text image similarity, you can safely ignore this warning.",
            stacklevel=find_stack_level(),
        )
        try:
            from conch.open_clip_custom import (
                create_model_from_pretrained,
                get_tokenizer,
            )
        except ImportError:
            raise ImportError(
                "Conch is not installed. You can install it using "
                "`pip install git+https://github.com/mahmoodlab/CONCH.git`."
            )

        if model_path is None:
            model_path = "hf_hub:MahmoodLab/conch"

        with hf_access(model_path):
            self.model, self.processor = create_model_from_pretrained(
                "conch_ViT-B-16", model_path, hf_auth_token=token
            )
            self.model.eval()
            self.tokenizer = get_tokenizer()

    @torch.inference_mode()
    def encode_image_dense(self, image):
        if not isinstance(image, torch.Tensor):
            image = self.processor(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        device = next(self.model.parameters()).device
        image = image.to(device)
        # trunk is a timm VisionTransformer — forward_features returns [B, 1+N, D]
        out = self.model.visual.trunk.forward_features(image)
        return DenseTokens(cls_token=out[:, 0], patch_tokens=out[:, 1:])

    @torch.inference_mode()
    def encode_image(self, image):
        if not isinstance(image, torch.Tensor):
            image = self.processor(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Move image to the same device as the model
        # Get the model device
        try:
            device = next(self.model.parameters()).device
        except Exception:
            device = torch.device("cpu")
        image = image.to(device)

        image_feature = self.model.encode_image(
            image, normalize=False, proj_contrast=True
        )
        return image_feature

    def tokenize(self, text, *args, **kwargs):
        import torch.nn.functional as F

        # Inline tokenization to avoid conch's batch_encode_plus call
        # which was removed in transformers >= 5.0.
        tokens = self.tokenizer(
            text,
            max_length=127,
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = F.pad(tokens["input_ids"], (0, 1), value=self.tokenizer.pad_token_id)
        return tokens

    @torch.inference_mode()
    def encode_text(self, text):
        encode_texts = self.tokenize(text)
        # Move tokenized text to the same device as the model
        # Get the model device
        try:
            device = next(self.model.parameters()).device
        except Exception:
            device = torch.device("cpu")
        encode_texts = encode_texts.to(device)
        text_feature = self.model.encode_text(encode_texts, normalize=True)
        return text_feature
