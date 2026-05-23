import warnings
from importlib.util import find_spec
from typing import Any

import torch

from lazyslide_models._model_registry import register
from lazyslide_models._utils import hf_access
from lazyslide_models.base import ModelBase, ModelTask, SlideEncodeOutput


class PrismSlideEncodeOutput(SlideEncodeOutput, total=False):
    """Prism slide-encoder output.

    Adds an optional ``latents`` field used downstream for captioning and
    zero-shot scoring against the text decoder.
    """

    latents: Any  # torch.Tensor, captioning-ready latents


@register(
    key="prism",
    is_gated=True,
    task=[ModelTask.multimodal, ModelTask.slide_encoder],
    license="CC-BY-NC-ND-4.0",
    description=(
        "A multi-modal generative foundation model for slide-level histopathology, "
        "the Prism models encode slide-level embeddings "
        "from :class:`Virchow <lazyslide.models.vision.Virchow>`."
    ),
    commercial=False,
    hf_url="https://huggingface.co/paige-ai/Prism",
    paper_url="https://doi.org/10.48550/arXiv.2405.10254",
    bib_key="Shaikovski2024-kd",
    param_size="557.7M",
    vision_encoder="virchow",
)
class Prism(ModelBase):
    def __init__(self, model_path=None, token=None):
        from transformers import AutoModel

        environs = find_spec("environs")
        sacremoses = find_spec("sacremoses")

        if environs is None or sacremoses is None:
            raise ModuleNotFoundError(
                "To run PRISM model, 'environs' and 'sacremoses' must be installed, try "
                "`pip install environs sacremoses`."
            )

        # Suppress warnings from transformers
        with warnings.catch_warnings(), hf_access(model_path):
            warnings.simplefilter("ignore")

            # Pre-fetch the remote code so we can patch it before model init.
            # In transformers >= 5.0, _tied_weights_keys must be a dict, but
            # the upstream Prism repo still uses a list.
            self._patch_tied_weights_keys(token)

            self.model = AutoModel.from_pretrained(
                "paige-ai/Prism",
                trust_remote_code=True,
                token=token,
                low_cpu_mem_usage=False,
            )
            self.model.eval()

    @staticmethod
    def _patch_tied_weights_keys(token=None):
        """Patch Prism's BioGptForCausalLM for transformers >= 5.0 compatibility.

        transformers 5.x changed ``_tied_weights_keys`` from a list to a dict
        mapping tied parameter names to their source. The upstream Prism repo
        still uses a list, which crashes during ``post_init()``.
        """
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            biogpt_cls = get_class_from_dynamic_module(
                "biogpt_hf.BioGptForCausalLM",
                "paige-ai/Prism",
                token=token,
            )
            tied = getattr(biogpt_cls, "_tied_weights_keys", None)
            if isinstance(tied, list):
                # Convert list to dict: tied weight -> source weight
                biogpt_cls._tied_weights_keys = {
                    k: "biogpt.embed_tokens.weight" for k in tied
                }
        except Exception:
            pass  # If patching fails, let the normal error path handle it

    @torch.inference_mode()
    def encode_slide(self, embeddings, coords=None, **kwargs) -> PrismSlideEncodeOutput:
        out = self.model.slide_representations(embeddings)
        return {
            "embeddings": out["image_embedding"],
            "latents": out["image_latents"],
        }

    @torch.inference_mode()
    def score(
        self,
        slide_embedding,
        prompts: list[list[str]],
    ):
        if len(prompts):
            pass

        device = self.model.device

        # Flatten all prompts and track indices for class reconstruction
        flat_prompts = []
        group_lengths = []
        for group in prompts:
            flat_prompts.extend(group)
            group_lengths.append(len(group))

        token_ids = self.model.tokenize(flat_prompts)[:, :-1].to(device)

        dummy_image_latents = torch.empty(
            (len(flat_prompts), 1, self.model.text_decoder.context_dim), device=device
        )
        decoder_out = self.model.text_decoder(token_ids, dummy_image_latents)

        text_proj = self.model.text_to_latents(decoder_out["text_embedding"])
        image_proj = self.model.img_to_latents(slide_embedding)

        sim = torch.einsum("i d, j d -> i j", image_proj, text_proj)  # (image, prompt)
        sim = sim * self.model.temperature.exp()
        zero_shot_probs = torch.softmax(
            sim.to(torch.float), dim=-1
        )  # (Bi, total_prompts)

        # Sum probabilities per group (class)
        class_probs = []
        start = 0
        for length in group_lengths:
            end = start + length
            class_probs.append(zero_shot_probs[:, start:end].sum(dim=-1, keepdim=True))
            start = end

        probs = torch.cat(class_probs, dim=-1)
        return probs

    @torch.inference_mode()
    def caption(
        self,
        img_latents,
        prompt: list[str],
        max_length: int = 100,
    ):
        genned_ids = self.model.generate(
            self.model.tokenize(prompt).to(self.model.device),
            key_value_states=img_latents,
            do_sample=False,
            num_beams=5,
            num_beam_groups=1,
            max_length=max_length,
        )
        genned_caption = self.model.untokenize(genned_ids)

        return genned_caption
