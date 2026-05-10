import torch

from lazyslide_models._model_registry import register
from lazyslide_models._utils import hf_access
from lazyslide_models.base import DenseTokens, ImageModel, ModelTask


@register(
    key="open-midnight",
    is_gated=True,
    task=ModelTask.vision,
    license="Apache 2.0",
    description="Open replication of Midnight, a state-of-the-art pathology foundation model trained on 12K slides",
    commercial=True,
    hf_url="https://huggingface.co/SophontAI/OpenMidnight",
    github_url="https://github.com/MedARC-AI/OpenMidnight",
    bib_key="kaplan2025openmidnight",
    param_size="1.1B",
    encode_dim=1536,
)
class OpenMidnight(ImageModel):
    def __init__(self, model_path=None, token=None):
        from huggingface_hub import hf_hub_download

        with hf_access("SophontAI/OpenMidnight"):
            download_location = hf_hub_download(
                repo_id="SophontAI/OpenMidnight",
                filename="teacher_checkpoint_load.pt",
                token=token,
            )
            model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitg14_reg", pretrained=False
            )
            checkpoint = torch.load(download_location, map_location="cpu")

            # OpenMidnight is trained at 224 resolution (baseline dinov2 is 392)
            pos_embed = checkpoint["pos_embed"]
            model.pos_embed = torch.nn.parameter.Parameter(pos_embed)
            model.load_state_dict(checkpoint)
            model.eval()

        self.model = model
        self.img_size = (224, 224)
        self.patch_size = (14, 14)
        self.grid_size = (16, 16)
        self.num_prefix_tokens: int = 5  # 1 CLS + 4 register tokens

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
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    @torch.inference_mode()
    def encode_image_dense(self, image):
        out = self.model.forward_features(image)
        return DenseTokens(
            cls_token=out["x_norm_clstoken"],
            patch_tokens=out["x_norm_patchtokens"],
        )

    @torch.inference_mode()
    def encode_image(self, image):
        output = self.model(image)  # CLS token output, shape: [B, 1536]
        return output
