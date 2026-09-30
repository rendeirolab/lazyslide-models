import torch
import torch.nn.functional as F
from torch import nn

from lazyslide_models._model_registry import register
from lazyslide_models.base import ImageTextModel, ModelTask

_HF_HUB_ID = "Astaxanthin/KEEP"


class _KEEPNet(nn.Module):
    """The layout of KEEP's ``modeling_keep.KEEPModel``, so its weights load."""

    def __init__(self, config: dict):
        import timm
        from transformers import BertConfig, BertModel

        super().__init__()
        vision = config["vision_config"]
        self.visual = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            img_size=vision["img_size"],
            patch_size=vision["patch_size"],
            init_values=vision["init_values"],
            num_classes=vision["num_classes"],
        )
        dim = config["projection_dim"]
        self.visual_head = nn.Sequential(
            nn.Linear(self.visual.num_features, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.text = BertModel(BertConfig(**config["text_config"]))
        self.logit_scale = nn.Parameter(torch.ones([]))


@register(
    key="keep",
    task=ModelTask.multimodal,
    license="MIT",
    description=(
        "Knowledge-enhanced pretraining for vision-language pathology "
        "foundation model on cancer diagnosis"
    ),
    commercial=True,
    hf_url="https://huggingface.co/Astaxanthin/KEEP",
    github_url="https://github.com/MAGIC-AI4Med/KEEP",
    paper_url="https://doi.org/10.1016/j.ccell.2026.01.019",
    bib_key="Zhou2026-keep",
    encode_dim=768,
)
class KEEP(ImageTextModel):
    """KEEP (KnowledgE-Enhanced Pathology) image-text model.

    A ViT-L/16 image encoder and a BERT text encoder, aligned using a disease
    knowledge graph. It is strong at zero-shot cancer detection and subtyping.

    The model is built here from its config and weights instead of its remote
    code. That code replaces timm's ``LayerScale`` for the whole process, so
    every timm ViT built after it gets a renamed parameter, and on current timm
    it does not load at all: its replacement rejects the ``device`` and
    ``dtype`` arguments timm now passes.
    """

    def __init__(self, model_path=None, token=None):
        import json

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import BertTokenizer

        with open(hf_hub_download(_HF_HUB_ID, "config.json", token=token)) as f:
            config = json.load(f)
        model = _KEEPNet(config)

        weights = model_path or hf_hub_download(
            _HF_HUB_ID, "model.safetensors", token=token
        )
        # KEEP renamed timm's LayerScale parameter from ``gamma`` to ``weight``
        state_dict = {
            k.replace(".ls1.weight", ".ls1.gamma").replace(
                ".ls2.weight", ".ls2.gamma"
            ): v
            for k, v in load_file(weights).items()
        }
        model.load_state_dict(state_dict)
        self.model = model.eval()
        # AutoTokenizer would read config.json and ask to run KEEP's remote code
        self.tokenizer = BertTokenizer.from_pretrained(_HF_HUB_ID, token=token)

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

        # README: Resize(224, BICUBIC), CenterCrop(224), ImageNet stats
        return Compose(
            [
                ToImage(),
                Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
                CenterCrop(224),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @torch.inference_mode()
    def encode_image(self, image):
        return self.model.visual_head(self.model.visual(image))

    @torch.inference_mode()
    def encode_text(self, text):
        if isinstance(text, str):
            text = [text]
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(
            text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        return F.normalize(self.model.text(**tokens).pooler_output, dim=-1)
