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
        # GenBioPathFM splits RGB into 3 single-channel images and encodes each
        b, _c, h, w = image.shape
        features = self.model._encode(image.view(b * 3, 1, h, w))
        # cls: [B*3, D] -> [B, 3, D] -> [B, 3*D]
        cls = features["x_norm_clstoken"].view(b, 3, -1)
        cls_token = torch.cat([cls[:, 0], cls[:, 1], cls[:, 2]], dim=-1)
        # patches: [B*3, N, D] -> [B, 3, N, D] -> [B, N, 3*D]
        patches = features["x_norm_patchtokens"]
        n, d = patches.shape[1], patches.shape[2]
        patches = patches.view(b, 3, n, d).permute(0, 2, 1, 3).reshape(b, n, 3 * d)
        return DenseTokens(cls_token=cls_token, patch_tokens=patches)

    @torch.inference_mode()
    def encode_image(self, image):
        return self.encode_image_dense(image).cls_token
