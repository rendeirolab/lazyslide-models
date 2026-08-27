from contextlib import contextmanager
from importlib.util import find_spec
from typing import Any

import torch

from lazyslide_models._model_registry import register
from lazyslide_models._utils import hf_access
from lazyslide_models.base import ModelBase, ModelTask, SlideEncodeOutput


class Prism2SlideEncodeOutput(SlideEncodeOutput, total=False):
    """PRISM2 slide-encoder output.

    Adds the ``diagnostic_embeddings`` field: the Phi-3 hidden state at the
    ``<|assistant|>`` position for a fixed image-only prompt, which the model
    card recommends for linear-probe evaluation.
    """

    diagnostic_embeddings: Any  # torch.Tensor [B, 3072]


@register(
    key="prism2",
    is_gated=True,
    task=[ModelTask.multimodal, ModelTask.slide_encoder],
    license="CC-BY-NC-ND-4.0",
    description=(
        "An end-to-end multimodal pathology foundation model with clinical "
        "dialogue, PRISM2 encodes slide-level embeddings and generates "
        "free-text responses from "
        ":class:`Virchow2 <lazyslide.models.vision.Virchow2>` tile embeddings."
    ),
    commercial=False,
    hf_url="https://huggingface.co/paige-ai/Prism2",
    paper_url="https://doi.org/10.1038/s41591-026-04521-4",
    bib_key="Vorontsov2026-p2",
    param_size="4.4B",
    encode_dim=2560,
    vision_encoder="virchow2",
    # PRISM2 consumes the Virchow2 *CLS token only* (1280-d), not the
    # 2560-d CLS+mean vector that `virchow2.encode_dim` advertises.
    slide_input_dim=1280,
)
class Prism2(ModelBase):
    """PRISM2 slide encoder and single-turn text generator.

    The model consumes pre-extracted Virchow2 tile embeddings, **not** raw
    slides. Per slide, pass an ``(N, 1280)`` tensor (or ``(B, N, 1280)`` for a
    batch) extracted with:

    - the class token only (not class + mean),
    - 20x magnification (0.5 mpp),
    - 224x224 tiles,
    - background/glass tiles removed.

    Requires ``flash-attn``, and therefore a CUDA GPU: the upstream
    ``modeling_prism2.py`` imports ``flash_attn_varlen_func`` at module level.
    """

    def __init__(self, model_path=None, token=None):
        from transformers import AutoModel, AutoProcessor

        if find_spec("flash_attn") is None:
            raise ModuleNotFoundError(
                "To run the PRISM2 model, 'flash-attn' must be installed. The "
                "upstream modeling code imports `flash_attn_varlen_func` at "
                "module level, so a CUDA GPU is required. Try "
                "`pip install flash-attn --no-build-isolation`."
            )

        with hf_access("paige-ai/Prism2"):
            self.model = AutoModel.from_pretrained(
                "paige-ai/Prism2",
                trust_remote_code=True,
                torch_dtype="auto",
                token=token,
            )
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(
                "paige-ai/Prism2",
                trust_remote_code=True,
                token=token,
            )

    @contextmanager
    def _autocast(self):
        """Run the wrapped block under bf16 autocast on CUDA.

        The perceiver's cross-attention calls ``flash_attn_varlen_func``, whose
        kernels only accept fp16/bf16 — every example in the model card wraps
        its calls in ``torch.autocast("cuda", torch.bfloat16)``. Doing it here
        means callers get a working model instead of a dtype error. No-op off
        CUDA.
        """
        if self.model.device.type == "cuda":
            with torch.autocast("cuda", torch.bfloat16):
                yield
        else:
            yield

    def _batch(self, embeddings, attention_mask=None):
        """Pad/mask tile embeddings into the batch dict the model expects.

        The upstream processor only accepts a ``list`` of per-slide tensors or
        an already-batched ``(B, N, D)`` tensor. Coerce anything else — NumPy
        arrays in particular, which would otherwise be iterated row-by-row as
        if each tile were a slide.
        """
        if not isinstance(embeddings, (list, tuple)):
            embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
            if embeddings.ndim == 2:  # single slide (N, D) -> (1, N, D)
                embeddings = embeddings.unsqueeze(0)

        batch = self.processor(
            tile_embeddings=embeddings, attention_mask=attention_mask
        )
        device = self.model.device
        return {k: v.to(device) for k, v in batch.items()}

    @torch.inference_mode()
    def encode_slide(
        self, embeddings, coords=None, attention_mask=None, **kwargs
    ) -> Prism2SlideEncodeOutput:
        """Encode tile embeddings into base and diagnostic slide embeddings.

        ``coords`` is accepted for interface compatibility and ignored —
        PRISM2's perceiver is position-free.
        """
        batch = self._batch(embeddings, attention_mask)
        with self._autocast():
            return {
                "embeddings": self.model.get_base_embedding(**batch),
                "diagnostic_embeddings": self.model.get_diagnostic_embedding(**batch),
            }

    @torch.inference_mode()
    def respond(
        self, embeddings, prompt: str, attention_mask=None, **gen_kwargs
    ) -> list[str]:
        """Generate a free-text response per slide for a single-turn prompt.

        Parameters
        ----------
        embeddings : array-like
            Virchow2 tile embeddings, ``(N, 1280)`` per slide or a batched
            ``(B, N, 1280)`` tensor.
        prompt : str
            The user text. Use ``"Write a report"`` for report generation; any
            other single-turn instruction or question also works. The
            ``<|image_1|>`` placeholder is added by the model.
        **gen_kwargs
            Forwarded to HuggingFace ``generate`` (``max_new_tokens``,
            ``num_beams``, ``do_sample``, ...).

        Returns
        -------
        list of str
            One decoded response per slide.
        """
        batch = self._batch(embeddings, attention_mask)
        with self._autocast():
            return self.model.get_response(**batch, prompt=prompt, **gen_kwargs)

    @torch.inference_mode()
    def yes_no_score(
        self,
        embeddings,
        question: str,
        attention_mask=None,
        return_log_probs: bool = False,
    ):
        """Score one Yes/No question against every slide in the batch.

        Returns ``P('Yes' | answer in {'Yes', 'No'})`` as a ``(B,)`` tensor, or
        ``(B, 2)`` log-probabilities when ``return_log_probs=True``.

        This is deliberately *not* named ``score``: unlike the
        ``score(embeddings, prompts) -> (B, n_classes)`` zero-shot contract
        implemented by :class:`Titan` and :class:`Prism`, it takes tile
        embeddings and a single question.
        """
        batch = self._batch(embeddings, attention_mask)
        with self._autocast():
            return self.model.yes_no_score(
                **batch,
                question=question,
                return_log_probs=return_log_probs,
            )
