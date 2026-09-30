import torch

from lazyslide_models._model_registry import register
from lazyslide_models._utils import hf_access
from lazyslide_models.base import ModelTask, TimmViTModel
from lazyslide_models.vision.virchow import Virchow2

_HF_HUB_ID = "FTZhou/CRISP"
# peft LoRA settings from CRISP's crisp.py: r=8, lora_alpha=16
_LORA_SCALE = 16 / 8


def _merge_lora(state_dict: dict, scale: float) -> dict:
    """Fold a peft LoRA state dict into plain weights: ``W + scale * B @ A``."""
    prefix = "base_model.model."
    merged = {}
    for key, value in state_dict.items():
        key = key.removeprefix(prefix)
        if ".lora_" in key:
            continue
        if ".base_layer." in key:
            module, param = key.split(".base_layer.")
            if param == "weight":
                a = state_dict[f"{prefix}{module}.lora_A.default.weight"]
                b = state_dict[f"{prefix}{module}.lora_B.default.weight"]
                value = value + scale * (b @ a)
            key = f"{module}.{param}"
        merged[key] = value
    return merged


@register(
    key="crisp",
    is_gated=True,
    task=ModelTask.vision,
    license=["MIT", "CC-BY-NC-ND-4.0"],
    license_url=[
        "https://huggingface.co/FTZhou/CRISP",
        "https://huggingface.co/paige-ai/Virchow2",
    ],
    description="A clinically-oriented foundation model for intraoperative pathology",
    commercial=False,
    hf_url="https://huggingface.co/FTZhou/CRISP",
    github_url="https://github.com/FT-ZHOU-ZZZ/CRISP",
    paper_url="https://doi.org/10.1038/s41591-026-04703-0",
    bib_key="Zhao2026-crisp",
    param_size="631.2M",
    encode_dim=2560,
    flops="328.97G",
)
class CRISP(Virchow2):
    """CRISP tile encoder for frozen sections.

    Virchow2 adapted with LoRA on more than 100,000 frozen section slides and
    validated prospectively in intraoperative diagnosis. Like Virchow2, it
    returns the class token concatenated with the mean patch token (2560-d).

    CRISP's card says MIT, but its checkpoint ships Virchow2's weights plus the
    LoRA, and Virchow2 is CC-BY-NC-ND-4.0, so it is marked non-commercial. The
    LoRA is merged into the Virchow2 weights at load time, so peft is not
    needed.
    """

    def __init__(self, model_path=None, token=None):
        from huggingface_hub import hf_hub_download
        from timm.layers import SwiGLUPacked

        # Virchow2's hub config, built here without its weights
        TimmViTModel.__init__(
            self,
            "vit_huge_patch14_224",
            pretrained=False,
            img_size=224,
            init_values=1e-5,
            num_classes=0,
            reg_tokens=4,
            mlp_ratio=5.3375,
            global_pool="",
            dynamic_img_size=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        with hf_access(_HF_HUB_ID):
            weights = model_path or hf_hub_download(
                _HF_HUB_ID, "CRISP.pth", token=token
            )
        state_dict = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(_merge_lora(state_dict, _LORA_SCALE))
        self.model.eval()
