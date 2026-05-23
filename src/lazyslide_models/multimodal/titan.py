import numpy as np
import torch
import torch.nn.functional as F

from lazyslide_models._model_registry import register
from lazyslide_models._utils import hf_access
from lazyslide_models.base import (
    ImageModel,
    InputConstraint,
    ModelTask,
    SlideEncodeOutput,
)


@register(
    key=["titan", "conch_v1.5"],
    is_gated=True,
    task=[ModelTask.multimodal, ModelTask.slide_encoder],
    license="CC-BY-NC-ND-4.0",
    description="Multimodal whole slide foundation model for pathology",
    commercial=False,
    github_url="https://github.com/mahmoodlab/TITAN",
    hf_url="https://huggingface.co/MahmoodLab/TITAN",
    paper_url="https://doi.org/10.48550/arXiv.2411.19666",
    bib_key="Ding2024-pk",
    param_size="158.9M",
    encode_dim=768,
    vision_encoder="titan",
    input_constraint=InputConstraint(min=448),
)
class Titan(
    ImageModel,
):
    TEMPLATES = [
        "CLASSNAME.",
        "an image of CLASSNAME.",
        "the image shows CLASSNAME.",
        "the image displays CLASSNAME.",
        "the image exhibits CLASSNAME.",
        "an example of CLASSNAME.",
        "CLASSNAME is shown.",
        "this is CLASSNAME.",
        "I observe CLASSNAME.",
        "the pathology image shows CLASSNAME.",
        "a pathology image shows CLASSNAME.",
        "the pathology slide shows CLASSNAME.",
        "shows CLASSNAME.",
        "contains CLASSNAME.",
        "presence of CLASSNAME.",
        "CLASSNAME is present.",
        "CLASSNAME is observed.",
        "the pathology image reveals CLASSNAME.",
        "a microscopic image of showing CLASSNAME.",
        "histology shows CLASSNAME.",
        "CLASSNAME can be seen.",
        "the tissue shows CLASSNAME.",
        "CLASSNAME is identified.",
    ]

    def __init__(self, model_path=None, token=None):
        from transformers import AutoModel, PreTrainedTokenizerFast

        with hf_access(model_path):
            # Pre-fetch the remote Titan class and patch it for
            # transformers >= 5.0 compatibility. Upstream `Titan.__init__`
            # never calls `self.post_init()`, so the attributes set in
            # `PreTrainedModel.post_init()` (notably `all_tied_weights_keys`)
            # are missing. `_finalize_model_loading` in transformers 5.x
            # reads `model.all_tied_weights_keys.keys()` and crashes with
            # `AttributeError: 'Titan' object has no attribute
            # 'all_tied_weights_keys'`.
            self._patch_titan_post_init(token)

            self.model = AutoModel.from_pretrained(
                "MahmoodLab/TITAN",
                add_pooling_layer=False,
                token=token,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
            )
            self.model.eval()
            self.conch, self.conch_transform = self.model.return_conch()
            self.conch.eval()
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained("MahmoodLab/TITAN")
            self.tokenizer.context_length = 128

    @staticmethod
    def _patch_titan_post_init(token=None):
        """Patch ``MahmoodLab/TITAN`` Titan class for transformers >= 5.0.

        Upstream ``Titan.__init__`` does ``super().__init__(config)`` but
        never calls ``self.post_init()``. In transformers 5.x, attributes
        such as ``all_tied_weights_keys`` are only set inside ``post_init``
        and are read during ``_finalize_model_loading``. Without the call,
        ``from_pretrained`` raises ``AttributeError``.

        We wrap the upstream ``__init__`` to invoke ``post_init`` at the
        end. Safe to call multiple times.
        """
        try:
            from transformers.dynamic_module_utils import (
                get_class_from_dynamic_module,
            )

            titan_cls = get_class_from_dynamic_module(
                "modeling_titan.Titan",
                "MahmoodLab/TITAN",
                token=token,
            )
            if getattr(titan_cls, "_lazyslide_post_init_patched", False):
                return

            original_init = titan_cls.__init__

            def patched_init(self, config, *args, **kwargs):
                original_init(self, config, *args, **kwargs)
                # Idempotent: post_init is safe to re-run if already called.
                self.post_init()

            titan_cls.__init__ = patched_init
            titan_cls._lazyslide_post_init_patched = True
        except Exception:
            # If patching fails, let the normal error path surface it.
            pass

    def to(self, device):
        super().to(device)
        self.conch.to(device)
        return self

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
                Resize(448, interpolation=InterpolationMode.BICUBIC, antialias=True),
                CenterCrop(448),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @torch.inference_mode()
    def encode_image(self, image):
        image_feature = self.conch(image)
        return image_feature

    @torch.inference_mode()
    def encode_text(self, text):
        # transformers 5.x removed `batch_encode_plus` from the new
        # `TokenizersBackend`. The tokenizer's `__call__` accepts the same
        # arguments and works on both 4.x and 5.x.
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
        try:
            device = next(self.model.parameters()).device
        except Exception:
            device = torch.device("cpu")
        encode_texts = tokens.to(device)
        text_feature = self.model.encode_text(encode_texts, normalize=True)
        return text_feature

    @torch.inference_mode()
    def encode_slide(
        self, embeddings, coords=None, base_tile_size=None, **kwargs
    ) -> SlideEncodeOutput:
        # Cast base_tile_size to numpy integer if it's not already
        slide_embeddings = self.model.encode_slide_from_patch_features(
            embeddings, coords, np.int64(base_tile_size)
        )
        return {"embeddings": slide_embeddings}

    @torch.inference_mode()
    def score(
        self, slide_embeddings, prompts: list[str], template: str = None, **kwargs
    ):
        if template is None:
            template = self.TEMPLATES

        classifier = self.model.zero_shot_classifier(
            prompts, template, device=slide_embeddings.device
        )
        scores = self.model.zero_shot(slide_embeddings, classifier)
        return scores
