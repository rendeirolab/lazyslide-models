import torch
import torch.nn.functional as F

from lazyslide_models._model_registry import register
from lazyslide_models._utils import hf_access
from lazyslide_models.base import ImageTextModel, InputConstraint, ModelTask

_HF_HUB_ID = "JerrryNie/ConceptCLIP"


@register(
    key="conceptclip",
    is_gated=True,
    task=ModelTask.multimodal,
    license="MIT",
    description=(
        "An explainable biomedical foundation model via large-scale "
        "concept-enhanced vision-language pretraining"
    ),
    commercial=True,
    hf_url="https://huggingface.co/JerrryNie/ConceptCLIP",
    github_url="https://github.com/JerrryNie/ConceptCLIP",
    paper_url="https://doi.org/10.1038/s41551-026-01764-x",
    bib_key="Nie2026-conceptclip",
    encode_dim=1152,
    input_constraint=InputConstraint(min=384),
)
class ConceptCLIP(ImageTextModel):
    """ConceptCLIP biomedical image-text model.

    A SigLIP so400m image encoder and a PubMedBERT text encoder, pretrained on
    23 million biomedical image-text-concept triplets across ten imaging
    modalities, pathology included.

    The Hugging Face repo is gated only to have users acknowledge that this is
    a research model, not a medical device. The weights are MIT licensed.
    """

    def __init__(self, model_path=None, token=None):
        from transformers import AutoModel, BertTokenizer

        with hf_access(_HF_HUB_ID):
            self.model = AutoModel.from_pretrained(
                model_path or _HF_HUB_ID, trust_remote_code=True, token=token
            ).eval()
            self.tokenizer = BertTokenizer.from_pretrained(_HF_HUB_ID, token=token)

    def get_transform(self):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import (
            Compose,
            Normalize,
            Resize,
            ToDtype,
            ToImage,
        )

        # the repo's SiglipImageProcessor: squash to 384 x 384, bicubic, 0.5/0.5
        return Compose(
            [
                ToImage(),
                Resize(
                    (384, 384), interpolation=InterpolationMode.BICUBIC, antialias=True
                ),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    @torch.inference_mode()
    def encode_image(self, image):
        return self.model.encode_image(image)[0]

    @torch.inference_mode()
    def encode_text(self, text):
        if isinstance(text, str):
            text = [text]
        device = next(self.model.parameters()).device
        # the 77-token context open_clip trained the text tower with
        tokens = self.tokenizer(
            text,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        features = self.model.encode_text(tokens["input_ids"].to(device))[0]
        return F.normalize(features, dim=-1)
