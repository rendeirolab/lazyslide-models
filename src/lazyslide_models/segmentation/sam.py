from typing import ClassVar

import torch

from lazyslide_models._model_registry import register
from lazyslide_models.base import (
    InputConstraint,
    ModelTask,
    SegmentationModel,
    SegmentationOutput,
)


@register(
    key="sam",
    task=ModelTask.segmentation,
    commercial=True,
    license="Apache 2.0",
    description="SAM model for image segmentation",
    github_url="https://github.com/facebookresearch/segment-anything",
    paper_url="https://arxiv.org/abs/2304.02643",
    flops="975.67G",
    input_constraint=InputConstraint(min=1024),
)
class SAM(SegmentationModel):
    SAM_VARIENTS: ClassVar[list[str]] = [
        "facebook/sam-vit-base",
        "facebook/sam-vit-large",
        "facebook/sam-vit-huge",
    ]

    SAM_HQ_VARIENTS: ClassVar[list[str]] = [
        "syscv-community/sam-hq-vit-base",
        "syscv-community/sam-hq-vit-large",
        "syscv-community/sam-hq-vit-huge",
    ]

    def __init__(self, variant="facebook/sam-vit-base", model_path=None, token=None):
        self.variant = variant
        if variant in self.SAM_VARIENTS:
            from transformers import SamModel, SamProcessor
            # from ultralytics import SAM

            self.model = SamModel.from_pretrained(variant, token=token)
            self.processor = SamProcessor.from_pretrained(variant, token=token)
            self._is_hq = False

        elif variant in self.SAM_HQ_VARIENTS:
            from transformers import SamHQModel, SamHQProcessor

            self.model = SamHQModel.from_pretrained(variant, token=token)
            self.processor = SamHQProcessor.from_pretrained(variant, token=token)
            self._is_hq = True
        else:
            raise ValueError(
                f"Unsupported SAM variant: {variant}. "
                f"Choose from {self.SAM_VARIENTS + self.SAM_HQ_VARIENTS}."
            )

    def get_transform(self):
        return None

    @torch.inference_mode()
    def get_image_embedding(self, image) -> torch.Tensor:
        """
        Get the image embedding from the SAM model.

        Returns:
            torch.Tensor: Image embedding tensor of shape (1, C, H, W).

        """
        img_inputs = self.processor(image, return_tensors="pt").to(self.model.device)

        embeddings = self.model.get_image_embeddings(img_inputs["pixel_values"])
        if self._is_hq:
            embeddings = embeddings[0]
        return embeddings.detach().cpu()

    @torch.inference_mode()
    def segment(
        self,
        image,
        image_embedding=None,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        segmentation_maps=None,
        multimask_output=False,
    ):
        """
        Segment the input image using the SAM model.

        Args:
            image (torch.Tensor): Input image tensor of shape (C, H, W).

        """
        inputs = self.processor(
            image,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            segmentation_maps=segmentation_maps,
            return_tensors="pt",
        )
        if image_embedding is not None:
            del inputs["pixel_values"]
            inputs["image_embeddings"] = image_embedding

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float64:
                inputs[k] = v.to(dtype=torch.float32)

        inputs = inputs.to(self.model.device)
        outputs = self.model(**inputs, multimask_output=multimask_output)
        # pred_masks are 256x256 logits in the padded 1024 frame; undo the
        # processor's pad + resize so they land on the input pixels
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
            binarize=False,
        )
        # Per image [num_prompts, num_masks, H, W] → [B, num_prompts*num_masks, H, W]
        prob = torch.stack([m.flatten(0, 1) for m in masks]).sigmoid()
        return SegmentationOutput(
            probability_map=prob,
        )
