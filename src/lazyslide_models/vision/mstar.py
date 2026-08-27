from lazyslide_models._model_registry import register
from lazyslide_models.base import ModelTask, TimmViTModel


@register(
    key="mstar",
    is_gated=True,
    task=ModelTask.vision,
    license="CC-BY-NC-ND-4.0",
    description="A multimodal knowledge-enhanced whole-slide pathology foundation model",
    commercial=False,
    hf_url="https://huggingface.co/Wangyh/mSTAR",
    github_url="https://github.com/Innse/mSTAR",
    paper_url="https://doi.org/10.1038/s41467-025-66220-x",
    bib_key="Xu2025-ms",
    param_size="303.4M",
    encode_dim=1024,
    flops="119.29G",
)
class MSTAR(TimmViTModel):
    """mSTAR tile encoder: a ViT-L/16 distilled with pathology reports and RNA-Seq."""

    def __init__(self, model_path=None, token=None):
        super().__init__(
            "hf-hub:Wangyh/mSTAR",
            token=token,
            init_values=1e-5,
            dynamic_img_size=True,
        )
