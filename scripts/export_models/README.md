# LazySlide export models

This repo keeps track of reproducible codes to build deployment version of models used in LazySlide.

Because some models relied on a specific version of a package, or it cannot be loaded easily from huggingface, 
we will try to build a static version to easily load it for inference.

The export scripts are named as `export_*.py`. They are all self-contained, please run with `uv run --script`.

# Upload weights to HuggingFace

For CTransPath
```bash
weights="CTransPath_exported.pt2"
folder_name="CTransPath"
uv run hf upload RendeiroLab/LazySlide-models-gpl \
  export_artifacts/${weights} \
  ${folder_name}/${weights}
```

For CHIEF
```bash
folder_name="CHIEF"
for weights in \
  "CHIEF_patch_encoder_exported.pt2"\
  "CHIEF_slide_encoder_exported.pt2"
; do
    uv run hf upload RendeiroLab/LazySlide-models-gpl \
      export_artifacts/${weights} \
      ${folder_name}/${weights}
done
```

For PathProfiler
```bash
folder_name="PathProfiler"
for weights in \
  "PathProfiler_tissue_seg_exported.pt2"\
  "PathProfiler_patch_quality_exported.pt2"
; do
    uv run hf upload RendeiroLab/LazySlide-models-gpl \
      export_artifacts/${weights} \
      ${folder_name}/${weights}
done
```

For GrandQC
```bash
folder_name="GrandQC"
for weights in \
  "GrandQC_MPP1_exported.pt2" \
  "GrandQC_MPP2_exported.pt2" \
  "GrandQC_MPP15_exported.pt2" \
  "GrandQC_tissue_seg_exported.pt2" \
; do
    uv run hf upload RendeiroLab/LazySlide-models \
      export_artifacts/${weights} \
      ${folder_name}/${weights}
done
```

For HEST
```bash
weights="HEST_tissue_seg_exported.pt2"
folder_name="HEST"
uv run hf upload RendeiroLab/LazySlide-models \
  export_artifacts/${weights} \
  ${folder_name}/${weights}
```

For MADELEINE
```bash
weights="MADELEINE_exported.pt2"
folder_name="MADELEINE"
uv run hf upload RendeiroLab/LazySlide-models \
  export_artifacts/${weights} \
  ${folder_name}/${weights}
```

For NuLite
```bash
folder_name="NuLite"
for weights in \
  "NuLite_H_exported.pt2" \
  "NuLite_M_exported.pt2" \
  "NuLite_T_exported.pt2" \
; do
    uv run hf upload RendeiroLab/LazySlide-models \
      export_artifacts/${weights} \
      ${folder_name}/${weights}
done
```

For FocusLiteNN
```bash
weights="FocusLiteNN_exported.pt2"
folder_name="FocusLiteNN"
uv run hf upload RendeiroLab/LazySlide-models \
  export_artifacts/${weights} \
  ${folder_name}/${weights}
```