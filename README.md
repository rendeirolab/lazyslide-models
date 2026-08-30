# LazySlide models

<p align="center">
  <i>The model zoo for <a href="https://github.com/rendeirolab/LazySlide">LazySlide</a>, the accessible whole slide image analysis framework</i>
</p>

[![Documentation Status](https://readthedocs.org/projects/lazyslide/badge/?version=stable&style=flat-square)](https://lazyslide.readthedocs.io/en/stable)
![pypi version](https://img.shields.io/pypi/v/lazyslide-models?color=0098FF&logo=python&logoColor=white&style=flat-square)
![conda version](https://img.shields.io/conda/vn/conda-forge/lazyslide-models?style=flat-square&logo=anaconda&logoColor=white&color=%2344A833)
![PyPI - License](https://img.shields.io/pypi/l/lazyslide-models?color=FFD43B&style=flat-square)

`lazyslide-models` contains all the models consumed by LazySlide. LazySlide
itself contains no model code; it is an inference and orchestration layer that
discovers models through the `MODEL_REGISTRY`. Any registered model can be used
by name with zero configuration.

## Model types

The zoo spans 9 task types: vision encoders, multimodal (image-text) models,
segmentation, slide encoders, tile prediction, classical hand-crafted features,
feature prediction, style transfer, and image generation. Every
[timm](https://huggingface.co/timm) model is also available for feature
extraction (e.g. `resnet50`).

The complete per-task model list, with licenses, gated-access badges and
citation keys, is maintained in the
[Model Zoo documentation](https://lazyslide.readthedocs.io/en/stable/avail_models.html).

## Usage

```python
from lazyslide_models import MODEL_REGISTRY, list_models

list_models("segmentation")          # ['cellpose', 'classpose', 'sam', ...]
model = MODEL_REGISTRY["instanseg"]()  # instantiate (weights download on first use)
```

In LazySlide, pass the registered name to any function:

```python
zs.tl.feature_extraction(wsi, model="conch")  # any vision or multimodal key
zs.seg.cells(wsi, model="instanseg")
```

## Exported models

Models that cannot be loaded directly from Hugging Face are exported to
[RendeiroLab/LazySlide-models](https://huggingface.co/RendeiroLab/LazySlide-models)
(permissive and non-commercial licenses) and
[RendeiroLab/LazySlide-models-gpl](https://huggingface.co/RendeiroLab/LazySlide-models-gpl)
(GPL). Reproduction scripts: [`scripts/export_models`](scripts/export_models).

## Contributing a new model

1. Open an [issue](https://github.com/rendeirolab/lazyslide-models/issues) with the
   `[New Model]` label to confirm the model fits the zoo.
2. Follow the
   [integration guide](https://lazyslide.readthedocs.io/en/stable/contributing/new_models.html).
3. Open a pull request against `main`; CI tests the models your PR changes.

## Licenses

Some models are non-commercial or gated, and licenses vary. Check each model's
registry entry (`MODEL_REGISTRY.to_dataframe()`) before use, and cite the
original paper.
