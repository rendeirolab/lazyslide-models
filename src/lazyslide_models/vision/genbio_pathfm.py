import torch

from lazyslide_models._model_registry import register
from lazyslide_models._utils import check_transformers_version
from lazyslide_models.base import DenseTokens, ImageModel, ModelTask


@register(
    key="genbio-pathfm",
    task=ModelTask.vision,
    license="GenBio AI Community License",
    license_url="https://huggingface.co/genbio-ai/genbio-pathfm/blob/main/LICENSE.txt",
    description="A state-of-the-art histopathology foundation model trained with JEDI (JEPA + DINO)",
    commercial=False,
    hf_url="https://huggingface.co/genbio-ai/genbio-pathfm",
    github_url="https://github.com/genbio-ai/genbio-pathfm",
    paper_url="https://doi.org/10.1101/2026.03.17.712534",
    bib_key="Kapse2026-gp",
    param_size="1.1B",
    encode_dim=4608,
)
class GenBioPathFM(ImageModel):
    def __init__(self, model_path=None, token=None):
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "transformers is not installed. You can install it using "
                "`pip install transformers`."
            )
        check_transformers_version("genbio-pathfm")

        self.model = AutoModel.from_pretrained(
            "genbio-ai/genbio-pathfm",
            trust_remote_code=True,
            token=token,
        )
        self.model.eval()

        self.img_size = (224, 224)
        self.patch_size = (16, 16)
        self.grid_size = (14, 14)
        self.num_prefix_tokens: int = 5  # 1 CLS + 4 storage tokens

    def get_transform(self):
        from torchvision.transforms.v2 import (
            CenterCrop,
            Compose,
            Normalize,
            Resize,
            ToDtype,
            ToImage,
        )

        return Compose(
            [
                ToImage(),
                Resize(224),
                CenterCrop(224),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(
                    mean=(0.697, 0.575, 0.728),
                    std=(0.188, 0.240, 0.187),
                ),
            ]
        )

    @torch.inference_mode()
    def encode_image_dense(self, image):
        tokens, (h, w) = self.model.backbone.prepare_tokens(image)
        rope = self.model.backbone.rope_embed(H=h, W=w)
        for blk in self.model.backbone.blocks:
            tokens = blk(tokens, rope)
        tokens = self.model.backbone.norm(tokens)
        return DenseTokens(
            cls_token=tokens[:, 0],
            patch_tokens=tokens[:, self.num_prefix_tokens :],
        )

    @torch.inference_mode()
    def encode_image(self, image):
        return self.encode_image_dense(image).cls_token
