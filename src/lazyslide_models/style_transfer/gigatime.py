import torch
from torch import nn

from lazyslide_models._model_registry import register
from lazyslide_models._utils import hf_access
from lazyslide_models.base import (
    InputConstraint,
    MarkerMapModel,
    ModelTask,
)

GIGATIME_CHANNELS = (
    "DAPI",
    "TRITC",  # background channel not used in analysis
    "Cy5",  # background channel not used in analysis
    "PD-1",
    "CD14",
    "CD4",
    "T-bet",
    "CD34",
    "CD68",
    "CD16",
    "CD11c",
    "CD138",
    "CD20",
    "CD3",
    "CD8",
    "PD-L1",
    "CK",
    "Ki67",
    "Tryptase",
    "Actin-D",
    "Caspase3-D",
    "PHH3-B",
    "Transgelin",
)


@register(
    key="gigatime",
    task=ModelTask.style_transfer,
    is_gated=True,
    license="PROV-GIGATIME LICENSE",
    license_url="https://github.com/prov-gigatime/GigaTIME/blob/main/LICENSE",
    description="Multimodal AI generates virtual population for tumor microenvironment modeling",
    commercial=False,
    github_url="https://github.com/prov-gigatime/GigaTIME",
    paper_url="https://doi.org/10.1016/j.cell.2025.11.016",
    bib_key="Valanarasu2025-md",
    param_size="9M",
    flops="52.88G",
)
class GigaTIME(MarkerMapModel):
    channel_names = GIGATIME_CHANNELS

    def __init__(self, model_path: str | None = None, token: str | None = None):
        from huggingface_hub import hf_hub_download

        with hf_access("prov-gigatime/GigaTIME"):
            weights_file = hf_hub_download(
                repo_id="prov-gigatime/GigaTIME",
                filename="model.pth",
            )

        self.model = GigaTIMEModel(num_classes=23)
        self.model.load_state_dict(torch.load(weights_file, map_location="cpu"))
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image):
        return self.model(image)

    def get_transform(self):
        import torch
        from torchvision.transforms.v2 import (
            Compose,
            Normalize,
            ToDtype,
            ToImage,
        )

        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def check_input_tile(self, mpp, size_x=None, size_y=None) -> bool:
        return True


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class GigaTIMEModel(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(
            nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0]
        )
        self.conv1_2 = VGGBlock(
            nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1]
        )
        self.conv2_2 = VGGBlock(
            nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2]
        )

        self.conv0_3 = VGGBlock(
            nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0]
        )
        self.conv1_3 = VGGBlock(
            nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1]
        )

        self.conv0_4 = VGGBlock(
            nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0]
        )

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        output = self.final(x0_4)
        return output


# GigaTIME-flash channel order, verbatim from the repo's `config.json`.
# `TRITC` and `Cy5` are background channels not used in downstream analysis.
GIGATIME_FLASH_CHANNELS = (
    "DAPI",
    "TRITC",
    "Cy5",
    "PD-1_1:200",
    "CD14",
    "CD4",
    "T-bet",
    "CD34",
    "CD68_1:100",
    "CD16",
    "CD11c",
    "CD138",
    "CD20",
    "CD3_1:1000",
    "CD8",
    "PD-L1",
    "CK_1:150",
    "Ki67_1:150",
    "Tryptase",
    "Actin-D",
    "Caspase3-D",
    "PHH3-B",
    "Transgelin",
)


@register(
    key="gigatime-flash",
    task=ModelTask.style_transfer,
    is_gated=True,
    license="Apache 2.0",
    license_url="https://huggingface.co/prov-gigatime/gigatime-flash/blob/main/LICENSE",
    description="Efficient spatial proteomics prediction from H&E",
    commercial=False,
    hf_url="https://huggingface.co/prov-gigatime/gigatime-flash",
    github_url="https://github.com/prov-gigatime/GigaTIME",
    paper_url="https://doi.org/10.48550/arXiv.2607.18218",
    bib_key="Usuyama2026-gf",
    param_size="23.8M",
    flops="13.68G",
    input_constraint=InputConstraint(min=256, max=256),
)
class GigaTIMEFlash(MarkerMapModel):
    """GigaTIME-flash: 23-channel virtual mIF maps from 256x256 H&E tiles.

    The HuggingFace repo ships only ``model.pth`` and ``config.json`` — the
    architecture lives in the upstream notebook
    ``scripts/gigatime_flash_testing.ipynb`` and is reproduced below.
    """

    channel_names = GIGATIME_FLASH_CHANNELS

    _hf_hub_id = "prov-gigatime/gigatime-flash"

    def __init__(self, model_path: str | None = None, token: str | None = None):
        from huggingface_hub import hf_hub_download

        with hf_access(self._hf_hub_id):
            weights_file = model_path or hf_hub_download(
                repo_id=self._hf_hub_id,
                filename="model.pth",
                token=token,
            )

        self.model = GigaTIMEFlashModel(num_classes=len(GIGATIME_FLASH_CHANNELS))

        checkpoint = torch.load(weights_file, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        self.model.load_state_dict(_remap_flash_state_dict(self.model, checkpoint))
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image):
        # `config.json` sets `apply_sigmoid: true`; the forward pass returns
        # logits. (GigaTIME v1 has no such flag and returns raw logits.)
        return torch.sigmoid(self.model(image))

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

        return Compose(
            [
                ToImage(),
                Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
                CenterCrop(256),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )


def _remap_flash_state_dict(model: nn.Module, checkpoint: dict) -> dict:
    """Map a released GigaTIME-flash checkpoint onto :class:`GigaTIMEFlashModel`.

    The checkpoint was saved from a ``DataParallel``-wrapped, PEFT-adapted
    model, so keys may carry a ``module.`` prefix, a ``.base_layer.`` LoRA
    segment, or a bare ``encoder.`` prefix. Try each candidate spelling and
    keep the first that matches a parameter of the same shape.
    """
    model_state = model.state_dict()
    loaded = {}
    for key, value in checkpoint.items():
        candidates = [key]
        if key.startswith("module."):
            candidates.append(key[len("module.") :])
        if ".base_layer." in key:
            candidates.append(key.replace(".base_layer.", "."))
        if key.startswith("encoder.") and not key.startswith(
            "encoder.base_model.model."
        ):
            candidates.append(key.replace("encoder.", "encoder.base_model.model.", 1))
        for candidate in candidates:
            if candidate in model_state and model_state[candidate].shape == value.shape:
                loaded[candidate] = value
                break
    return loaded


# ---------------------------------------------------------------------------
# GigaTIME-flash architecture
#
# Vendored from `scripts/gigatime_flash_testing.ipynb` in the upstream
# GigaTIME repo. The LoRA layers are reimplemented inline (rather than pulled
# from `peft`) so that the parameter names match the released checkpoint
# without depending on peft's internal naming.
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Minimal PEFT-compatible LoRA wrapper around an ``nn.Linear``."""

    def __init__(self, base_layer, r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.base_layer = base_layer
        self.scaling = {"default": lora_alpha / r}
        self.lora_dropout = nn.ModuleDict({"default": nn.Dropout(lora_dropout)})
        self.lora_A = nn.ModuleDict(
            {"default": nn.Linear(base_layer.in_features, r, bias=False)}
        )
        self.lora_B = nn.ModuleDict(
            {"default": nn.Linear(r, base_layer.out_features, bias=False)}
        )

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        result = self.base_layer(x)
        update = self.lora_B["default"](
            self.lora_A["default"](self.lora_dropout["default"](x))
        )
        return result + update * self.scaling["default"]


def apply_lora(module: nn.Module, target_modules=("qkv", "proj")) -> None:
    """Recursively replace the named ``nn.Linear`` children with LoRA layers."""
    for child_name, child in list(module.named_children()):
        if child_name in target_modules and isinstance(child, nn.Linear):
            setattr(module, child_name, LoRALinear(child))
        else:
            apply_lora(child, target_modules)


class BaseModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


class PeftFallbackModel(nn.Module):
    """Reproduces peft's ``base_model.model.*`` key nesting without peft."""

    def __init__(self, model):
        super().__init__()
        self.base_model = BaseModelWrapper(model)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model.model, name)

    def forward(self, *args, **kwargs):
        return self.base_model.model(*args, **kwargs)


class GigaTIMEFlashModel(nn.Module):
    """DINOv2-small (+LoRA) encoder with a convolutional mIF decoder."""

    def __init__(self, num_classes=23):
        super().__init__()
        from timm.layers import SwiGLUPacked
        from timm.models.vision_transformer import _create_vision_transformer

        vit_encoder = _create_vision_transformer(
            "vit_small_patch14_dinov2",
            pretrained=False,
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            init_values=1e-5,
            mlp_ratio=2.66667 * 2,
            mlp_layer=SwiGLUPacked,
            act_layer=nn.SiLU,
            img_size=224,
        )
        apply_lora(vit_encoder)
        self.encoder = PeftFallbackModel(vit_encoder)
        self.num_classes = num_classes

        self.decoder4 = self._decoder_block(384, 192)
        self.decoder3 = self._decoder_block(192, 96)
        self.decoder2 = self._decoder_block(96, 48)
        self.decoder1 = self._decoder_block(48, 24)

        self.skip1 = self._skip_block(384, 48, 3)
        self.skip2 = self._skip_block(384, 96, 2)
        self.skip3 = self._skip_block(384, 192, 1)
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)

        # The released weights were trained on 256x256 tiles.
        self.encoder.patch_embed.img_size = (256, 256)
        self.encoder.patch_embed.grid_size = (16, 16)
        self.encoder.patch_embed.num_patches = 256

    @staticmethod
    def _decoder_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

    @staticmethod
    def _skip_block(in_channels, out_channels, times):
        layers = []
        for _ in range(times):
            layers.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    @staticmethod
    def _tokens_to_patch(x):
        batch_size, num_tokens, channels = x.shape
        grid = int(num_tokens**0.5)
        if grid * grid != num_tokens:
            raise ValueError(f"Expected square patch tokens, got {num_tokens}")
        return x.permute(0, 2, 1).contiguous().view(batch_size, channels, grid, grid)

    @staticmethod
    def _resize_pos_embed(pos_embed, new_grid_size):
        cls_tok, grid_tok = pos_embed[:, :1], pos_embed[:, 1:]
        old_num = int(grid_tok.shape[1] ** 0.5)
        grid_tok = grid_tok.reshape(1, old_num, old_num, -1).permute(0, 3, 1, 2)
        grid_tok = torch.nn.functional.interpolate(
            grid_tok, size=new_grid_size, mode="bicubic", align_corners=False
        )
        grid_tok = grid_tok.permute(0, 2, 3, 1).reshape(
            1, new_grid_size[0] * new_grid_size[1], -1
        )
        return torch.cat((cls_tok, grid_tok), dim=1)

    def forward(self, x):
        enc_outs = []
        x = self.encoder.patch_embed(x)
        _, num_tokens, _ = x.shape
        grid = int(num_tokens**0.5)

        pos_emb = self._resize_pos_embed(self.encoder.pos_embed, (grid, grid))
        x = x + pos_emb[:, 1:]
        x = self.encoder.patch_drop(x)
        x = self.encoder.norm_pre(x)

        for idx, block in enumerate(self.encoder.blocks):
            x = block(x)
            if idx in {3, 5, 8, 11}:
                enc_outs.append(x)

        x = self.decoder4(self._tokens_to_patch(enc_outs[3]))
        x = self.decoder3(x + self.skip3(self._tokens_to_patch(enc_outs[2])))
        x = self.decoder2(x + self.skip2(self._tokens_to_patch(enc_outs[1])))
        x = self.decoder1(x + self.skip1(self._tokens_to_patch(enc_outs[0])))
        x = self.final_conv(x)

        if x.shape[-1] != 256:
            x = nn.functional.interpolate(
                x, size=(256, 256), mode="bilinear", align_corners=True
            )
        return x
