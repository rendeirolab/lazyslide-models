import math

import torch

from lazyslide_models._model_registry import register
from lazyslide_models.base import ModelTask, SlideEncodeOutput, SlideEncoderModel


@register(
    key="moozy",
    task=ModelTask.slide_encoder,
    license="CC-BY-NC-SA-4.0",
    description="A patient-first foundation model for computational pathology",
    commercial=False,
    hf_url="https://huggingface.co/AtlasAnalyticsLab/MOOZY",
    github_url="https://github.com/AtlasAnalyticsLab/MOOZY",
    paper_url="https://doi.org/10.48550/arXiv.2603.27048",
    bib_key="Kotp2026-mz",
    param_size="85.77M",
    encode_dim=768,
    vision_encoder="lunit-dino-s8",
)
class Moozy(SlideEncoderModel):
    """MOOZY slide and case encoder.

    The slide encoder requires spatial coordinates and patch sizes for its
    ALiBi position bias. Pass ``coords`` (xy positions) and ``patch_sizes``
    as keyword arguments to :meth:`encode_slide`.

    The case transformer aggregates multiple slide embeddings into a single
    patient-level representation via :meth:`encode_case`.
    """

    def __init__(self, model_path=None, token=None):
        try:
            from moozy.hf_hub import ensure_checkpoint
            from moozy.models.factory import load_stage2_inference_model
        except ImportError:
            raise ImportError(
                "moozy is not installed. You can install it using `pip install moozy`."
            )

        device = torch.device("cpu")
        checkpoint_path = ensure_checkpoint()
        self.model = load_stage2_inference_model(checkpoint_path, device=device)

    @torch.inference_mode()
    def encode_slide(
        self,
        embeddings,
        coords=None,
        **kwargs,
    ) -> SlideEncodeOutput:
        """Encode patch features into a slide-level embedding.

        Parameters
        ----------
        embeddings : torch.Tensor
            Patch features. Accepted shapes:

            - ``[B, H, W, 384]`` — spatial grid layout (native format).
            - ``[H, W, 384]`` — single slide spatial grid (will be unsqueezed).
            - ``[B, T, 384]`` — flat sequence; will be reshaped to a square
              grid (T must be a perfect square or will be zero-padded).
            - ``[T, 384]`` — single slide flat sequence.

        coords : torch.Tensor
            Spatial coordinates for each patch token. Must match the spatial
            layout of ``embeddings``. Shape ``[B, H, W, 2]`` or ``[H, W, 2]``
            for grid inputs, or ``[B, T, 2]`` / ``[T, 2]`` for flat inputs.
            **Required** — the ALiBi position bias needs real-space positions.

        **kwargs
            ``patch_sizes`` : float or torch.Tensor, optional
                Patch size in level-0 pixels. Defaults to 224.
            ``invalid_mask`` : torch.Tensor, optional
                Boolean mask ``[B, H, W]`` where True = invalid/background.

        Returns
        -------
        dict
            ``{"embeddings": cls_output}`` where ``cls_output`` is ``[B, 768]``.
        """
        if coords is None:
            raise ValueError(
                "MOOZY slide encoder requires spatial coordinates (coords). "
                "Pass coords as xy positions matching the spatial layout of embeddings."
            )

        patch_sizes = kwargs.get("patch_sizes", 224)
        invalid_mask = kwargs.get("invalid_mask", None)

        # Handle dimensionality
        if embeddings.dim() == 2:
            # [T, 384] -> [1, T, 384]
            embeddings = embeddings.unsqueeze(0)
        if embeddings.dim() == 3:
            if embeddings.shape[-1] == 384:
                # [B, T, 384] -> reshape to [B, H, W, 384]
                B, T, D = embeddings.shape
                H = int(math.isqrt(T))
                W = (T + H - 1) // H
                if H * W != T:
                    # Zero-pad to fill grid
                    pad_count = H * W - T
                    pad = torch.zeros(
                        B,
                        pad_count,
                        D,
                        device=embeddings.device,
                        dtype=embeddings.dtype,
                    )
                    embeddings = torch.cat([embeddings, pad], dim=1)
                    if invalid_mask is None:
                        invalid_mask = torch.zeros(
                            B, H * W, dtype=torch.bool, device=embeddings.device
                        )
                        invalid_mask[:, T:] = True
                    if coords.dim() == 2:
                        coords = coords.unsqueeze(0)
                    if coords.dim() == 3:
                        coord_pad = torch.zeros(
                            B, pad_count, 2, device=coords.device, dtype=coords.dtype
                        )
                        coords = torch.cat([coords, coord_pad], dim=1)
                embeddings = embeddings.reshape(B, H, W, D)
                if coords.dim() == 3:
                    coords = coords.reshape(B, H, W, 2)
                if invalid_mask is not None and invalid_mask.dim() == 2:
                    invalid_mask = invalid_mask.reshape(B, H, W)
            else:
                # [H, W, 384] -> [1, H, W, 384]
                embeddings = embeddings.unsqueeze(0)
                if coords.dim() == 3:
                    coords = coords.unsqueeze(0)

        # Now embeddings is [B, H, W, 384], coords is [B, H, W, 2]
        device = next(self.model.slide_encoder.parameters()).device
        embeddings = embeddings.to(device)
        coords = coords.to(device)

        if isinstance(patch_sizes, (int, float)):
            patch_sizes = torch.tensor(
                [patch_sizes], device=device, dtype=torch.float32
            )
            patch_sizes = patch_sizes.expand(embeddings.shape[0])
        else:
            patch_sizes = torch.as_tensor(
                patch_sizes, device=device, dtype=torch.float32
            )

        if invalid_mask is not None:
            invalid_mask = invalid_mask.to(device)

        cls_output, _, _ = self.model.slide_encoder(
            embeddings,
            mask=None,
            invalid_mask=invalid_mask,
            coords_xy=coords,
            patch_sizes=patch_sizes,
        )
        return {"embeddings": cls_output}

    @torch.inference_mode()
    def encode_case(self, slide_embeddings: torch.Tensor) -> torch.Tensor:
        """Aggregate slide embeddings into a case-level embedding.

        Parameters
        ----------
        slide_embeddings : torch.Tensor
            Slide-level CLS embeddings. Shape ``[S, 768]`` where S is the
            number of slides for a patient case.

        Returns
        -------
        torch.Tensor
            Case embedding of shape ``[768]``.
        """
        device = next(self.model.case_transformer.parameters()).device
        slide_embeddings = slide_embeddings.to(device)
        return self.model.case_transformer(slide_embeddings)
