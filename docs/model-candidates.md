# Model candidates

A survey of models that could join the zoo, with the licence and weight-availability
checks already done. Compiled 2026-08-30 against a registry of 82 keys.

Everything here was verified against the HuggingFace API, the GitHub API and the
actual `LICENSE` files — gating, licence, and whether weight files really exist —
rather than taken from the papers. Claims that could not be confirmed say so.

Related: [`rendeirolab/LazySlide#70`](https://github.com/rendeirolab/LazySlide/issues/70)
tracks the older backlog; the [status section](#status-of-the-existing-backlog)
below corrects two entries in it.

## Shortlist

| candidate | why it is top of the list | licence | effort |
|---|---|---|---|
| **KEEP** `Astaxanthin/KEEP` | pathology VLM, Cancer Cell 2026, **18.7k downloads** — the MIT-licensed CLIP the zoo lacks | **MIT** | easy |
| **Feather ×3** `MahmoodLab/abmil.base.*.pc108-24k` | 0.9M-param slide encoder beating TITAN (48.5M) on Patho-Bench, over three encoders already registered | CC-BY-NC-ND | easy |
| **PathoSAM** | MIT nuclei instance segmentation, MIDL 2025 — histopathology-tuned upgrade over the generic `sam` entry | **MIT** | easy |
| **MIPHEI-ViT** | H&E → 16-channel mIF, peer-reviewed, ungated GitHub weights, H-optimus-0 encoder already present | Sanofi NC | needs the runner fix |
| **StainNet-Base** | **every encoder in the zoo is H&E-trained**; this one is IHC/special-stain native | CC-BY-NC-ND | easy |
| **HistoPrism** | pan-cancer H&E→ST, ICLR 2026, same shape as STPath | ⚠ `ND` contradiction | medium |
| **StainFuser** | the only unambiguous licence in virtual staining | **Apache-2.0** | needs a new output kind |
| **VitaminP** | whole-cell + nuclei segmentation, H&E-only at inference | **MIT** | easy |

## H&E → spatial transcriptomics

**The dominant filter is weights, not quality.** Of ~35 named models in this
subfield, only about six publish usable pretrained weights. Top-venue acceptance
turns out to be uncorrelated with releasing a checkpoint: TRIPLEX (CVPR 2024),
BLEEP (NeurIPS 2023), Stem (ICLR 2025) and STFlow (ICML 2025 Spotlight) are all
training-code-only. STFlow is in any case superseded by STPath, already in the zoo.

A common misreading worth heading off: **"HEST-1k leaderboard entries" are not
models.** HEST-Benchmark scores *frozen encoder features + PCA + ridge
regression*, so there is nothing to download beyond an encoder the zoo already
holds. It is an eval harness, not a model source.

| model | output | licence (code / weights) | gated | fits | note |
|---|---|---|---|---|---|
| **HistoPrism** `HuSusu/HistoPrism` | pan-cancer, 38k-gene panel, HEST-1k | CC-BY-NC-4.0 / CC-BY-NC-**ND**-4.0 ⚠ | no | `feature_prediction` (`uni`, `needs_coords`, `whole_slide`) | ICLR 2026, peer-reviewed. Best-fitting candidate |
| **Phoenix** `peng-lab/phoenix` | **single-cell** Xenium expression, flow matching | PolyForm-NC / CC-BY-NC-4.0 | no | see caveat | pan-cancer, 10k+ samples, most actively maintained repo in the field (last commit 2026-08-28, CI + docs + Colab) |
| **SEQUOIA** `gevaertlab/sequoia-*` | slide-level **bulk** RNA-seq | **MIT / MIT** | no | `feature_prediction` (`whole_slide`) | Nat Commun 2024. 100 ungated MIT repos via `PyTorchModelHubMixin.from_pretrained` — near-zero integration risk |
| **MoLF** `HuSusu/MoLF` | pan-cancer, mixture-of-latent-flow, claims human→mouse transfer | NOASSERTION / CC-BY-NC-**ND**-4.0 ⚠ | no | `feature_prediction` (`uni2`) | ICML 2026. Two-stage load (VAE **and** flow) is the main cost |
| **EON** `ViggyVenkat/EON` | per-patch cell-state *program* scores | MIT / MIT | **manual** | `feature_prediction` (`uni2`, 8-neighbour coords) — near-perfect fit | 2,036 downloads, but GitHub 404s and there is **no paper**. Email-the-author item |
| **Paladin** `zhihuanglab/Paladin` | WSI → spatial multi-omics | "other" | manual | `feature_prediction` | no paper yet (2026-08-08) |
| **HNE2Space** `mkkim032599/HNE2Space` | 11 cell-type **proportions** per tile | **Apache-2.0** | no | `tile_prediction` | cheapest win here |
| **CytoFormer** `zhihuanglab/CytoFormer` | cell-type classification from Xenium | CC-BY-NC-4.0 | auto | needs a cell-level path | cell-level |
| **SEAL** `MahmoodLab/SEAL` | ST-aligned *encoder*, not a regressor | none in repo ⚠ / CC-BY-NC-ND | auto | `vision` + `multimodal` | only conch/univ2 uploaded so far |

Caveats worth carrying:

- **Phoenix's protocol fit is unresolved.** Its API is
  `gex_pred, coords_list = pipeline(gene_list, dataloader)` — output rows are
  **cells carrying their own coordinates**, not one row per tile, so it does not
  drop into `tile_prediction` the way a per-tile regressor would. It bundles its
  own `vit_giant_patch14_reg4_dinov2` encoder (so no `features_model_name`
  dependency) and runs a multi-step **ODE sampler**, which is slower and
  stochastic. Needs a closer look before scheduling.
- **SEQUOIA is bulk, not spatial** — the paper recovers spatial signal post-hoc
  with sliding windows. It is also *one model per cancer type × 5 folds*, and
  output shape is `(n_genes,)` per slide rather than `(n_tiles, n_genes)`.
- Prioritisation signal from **HESCAPE** (arXiv 2508.01490): contrastive
  image–gene pretraining *improved* mutation classification but **degraded**
  direct expression prediction versus plain encoders, with batch effects
  dominating. That argues for direct regressors and generative models
  (HistoPrism, MoLF, Phoenix) over retrieval/contrastive approaches (BLEEP,
  mclSTExp) for this task specifically.

**Affects a model already registered:** Path2Space is now published in **Cell
2026**, with a new companion repo `eldadshulman/path2space-companion` under
**Apache-2.0** and weights on Zenodo (`10.5281/zenodo.20174301`, 154-checkpoint
ensemble, ~14k genes). Worth re-checking the existing entry's `license` and
`paper_url` against that release. Separately, `ratschlab/DeepSpot` (**MIT**,
Zenodo weights) is a *different, more permissively licensed* model from the
`deepspotm` currently registered (PolyForm-NC / CC-BY-NC-SA).

## Spatial proteomics

Two different things get called a "spatial proteomics model", and only one is
currently reachable.

### Models that consume multiplexed images

Blocked on multiplex tiling — `wsidata` ships `BioFormatsReader` and
`SpatialDataImage2DReader`, so an OME-TIFF *can* be opened, but the channel
plumbing is not wired up and the tile datasets assume 3-channel RGB.

| model | licence | gated | note |
|---|---|---|---|
| `MahmoodLab/KRONOS` | CC-BY-NC-ND-4.0 | manual | ViT-S/16, 47M patches, 175 markers, 8 platforms |
| `MahmoodLab/KRONOS2` | CC-BY-NC-ND-4.0 | manual | marker-**aware**: takes a single-marker patch stack + marker names, so one model covers arbitrary panels |
| `yandrewl/Eva` | none stated ⚠ | manual | spatial-proteomics FM |
| `MahmoodLab/CARTA` | CC-BY-NC-4.0 | no | TMA core + tissue segmentation for mIF. ⚠ its detector is **YOLOv8n via ultralytics (AGPL-3.0)** — copyleft that would propagate into the library |
| `pennweili/cosie-foundation` | **none** ⚠ | no | ST + proteomics + histology |

The Mahmood lab also ships **CORAL**, a spatial-proteomics toolkit that already
loads KRONOS/KRONOS2 — interop may be a better use of effort than
reimplementation.

### H&E → proteomics / virtual staining

The dominant failure mode of this field is **training code without weights** —
the npj Digital Medicine 2025 benchmark says as much, noting authors "usually
retrain" because nothing reusable ships. The handful that did:

| model | output | paper | licence | weights (verified) | note |
|---|---|---|---|---|---|
| **MIPHEI-ViT** `Sanofi-Public/MIPHEI-ViT` | H&E → **16-ch Orion mIF** dense map | *Comput. Biol. Med.* 206:111564 (2026), peer-reviewed | Sanofi non-commercial | GitHub release `v1.0.0` → `model.safetensors` **26.8 MB, ungated** (the HF mirror `Estabousi/MIPHEI-vit` *is* gated — use GitHub) | best overall pick |
| **DTR / DGR** `birkhoffkiki/DTR` | AF→H&E, H&E→PAS, H&E→3-ch mIHC, **scanner stain transfer** — four models, one repo | *Nat. Commun.* 17:4494 (2026) | CC-BY-NC-**ND** (Zenodo copy says CC-BY-4.0 ⚠) | release tag `weights`, 4 × 145 MB | best breadth per integration |
| **StainFuser** `R-J96/stainFuser` → HF `R-J/StainFuser` | reference-conditioned **stain normalisation** | arXiv 2403.09302 (TIA Lab) | **Apache-2.0 code *and* weights** | ungated `checkpoint.safetensors` | cleanest legal story in the family |
| **HistoPlexer** `ratschlab/HistoPlexer` | H&E → **11-marker IMC** dense map | *Nat. Mach. Intell.* 2025 | CC-BY-NC-**ND** | **`demo/model/pytorch_model.pt`, 80.1 MB, committed in-repo** | melanoma-only (TuPro); `use_fm_features: false`, so no FM dependency |
| **PixCell-VS** `StonyBrook-CVLab/pixcell-virtual-staining` | H&E → IHC (ER/PR/Ki67/HER2), LoRA | arXiv 2506.05127 | ⚠ LoRAs Apache-2.0, but base `PixCell-1024` is CC-BY-NC-**ND** and conditioning needs gated UNI2-h — effective licence is NC-ND | ungated, 5 LoRA + 5 MLP | RGB output |
| **USIGAN** `BairdXiong/USIGAN` | H&E → IHC (ER/HER2/Ki67/PR) | *IEEE TIP* 2026 | ⚠ HF says **MIT**, repo `LICENSE.md` says CC-BY-NC-SA-4.0 | ungated, per-marker `latest_net_G.pth` | breast only |
| **MuPaD-HE2mIF** `xiangjx/MuPaD-HE2mIF` | H&E → 16-ch Orion mIF, single forward pass (flow matching read out at `t=1`) | arXiv 2604.03635 preprint | ⚠ card says `mit`, base `MuPaD-512` is CC-BY-NC-**ND**; MUSK backbone is `gated: manual` | ungated (VAE 335 MB + SiT 5.56 GB) | GitHub 404s; recheck in ~3 months |

**MIPHEI-ViT detail worth knowing before scheduling:** the 26.8 MB is the
*decoder only* — the H-Optimus-0 encoder comes from `bioptimus/H-optimus-0`,
already in the zoo but `gated: auto`, so CI still needs an HF token.
`--tile_size 512`, `--mpp_target 0.5`, trained on ORION-CRC (colorectal). Its
licence text explicitly reserves *"AI training and similar technologies"* — worth
a careful read before shipping.

## Encoders, slide encoders and VLMs

The HEST-Benchmark leaderboard (25 models, updated 2026-04-03) ranks H-Optimus-1
0.4229 > **GenBio-PathFM 0.4197** > H-Optimus-0 0.4150 > UNI2-h 0.4141 >
Virchow 0.4061. **All five are already in the zoo.** There is no urgent *general*
tile-encoder gap. The interesting candidates are a different granularity, a
different stain, slide-level, or a much better licence.

### The standout, and it is a VLM

| model | licence | gated | downloads | note |
|---|---|---|---|---|
| **KEEP** `Astaxanthin/KEEP` | **MIT** | no | **18,678** | ViT-L/16 + BERT, knowledge-graph-enhanced pathology VLM. **Cancer Cell 2026**, peer-reviewed at a top venue. Gives `encode_image` + `encode_text` + zero-shot `score` in one 0.4B model, and plugs the "MIT-licensed pathology CLIP" hole that CONCH (NC) and PLIP leave. Highest download count in this survey by two orders of magnitude |

### Tile encoders worth having for a specific reason

| model | licence | gated | reason |
|---|---|---|---|
| **StainNet-Base / StainNet** `JWonderLand/StainNet-Base` | CC-BY-NC-ND-4.0 | no | DINO on 1.4M **IHC and special-stain** patches. **Every encoder in the zoo is H&E-trained** — nothing else covers non-H&E stains. Plain `timm`, trivial `TimmModel` |
| **LEMON** `aliceblondel/LEMON` | **MIT** | no | single-**nucleus** encoder, 40×40 px at 0.25 mpp. Pairs with InstanSeg/Cellpose/NuLite already in the zoo: segment nuclei, then embed each one. ⚠ the HF repo showed **0 weight files** — confirm before scheduling |
| **Mettle** `slideflow-labs/Mettle` | CC-BY-NC-ND-4.0 | no | H-optimus-0 continued for scanner/stain invariance; self-reported **strict upgrade** over its parent on all five benchmarks it cites (PathoROB 0.922 vs 0.802). **No paper, single-vendor numbers, 114 downloads** — promising but unvetted, and its licence is *worse* than h-optimus-0's Apache-2.0. Add alongside, never instead |
| **Digepath** `xtxx/Digepath` | CC-BY-NC-4.0 | yes | npj Digital Medicine 2026; GI-subspecialty SOTA |
| **EXAONE-Path 2.0 / 2.5** | custom `exaonepath` NC | 2.0 no / 2.5 auto | matched **patch + slide encoder pair** from one vendor. Licence read in full: research-only (`commercial=False`) but it **explicitly permits derivatives and redistribution** with the agreement attached and an `EXAONEPath` name prefix — so vendoring is allowed, unlike `ND` |

### Slide encoders — where the cheap capability is

| model | licence | gated | note |
|---|---|---|---|
| **Feather-24K** `MahmoodLab/abmil.base.{conch_v15,uni_v2,uni}.pc108-24k` | CC-BY-NC-ND-4.0 | auto | **ICML 2025.** A 0.9M-param ABMIL reporting 76.2 avg over 15 Patho-Bench tasks vs TITAN 75.9 (48.5M), THREADS 74.1, GigaPath 72.6, CHIEF 69.8. All three required tile encoders are already in the zoo, and the model is an attention-pooling MLP — best value-to-effort ratio in this survey |
| **CARE** `Zipper-1/CARE` | CC-BY-NC-4.0 | auto | **CVPR 2026 Highlight**, 18.8M, needs only CONCHv1.5. Gotcha: input is **512×512 at 20×**, not the CONCHv1.5 448 default; pins `transformers==4.57.6` |
| **ELF** `luoxd96/ELF` | **GPL-3.0** ⚠ | no | ABMIL ensemble over UNI + CONCHv1.5 + GigaPath + Virchow2 + H-optimus-0 — all five already in the zoo. Copyleft blocks vendoring into this MIT package, and the "GPLv3 + non-commercial" wording is internally contradictory |
| **Prism2-survival** `paige-ai/Prism2-survival` | CC-BY-NC-ND-4.0 | yes | strict *addition* to the existing `prism2`; survival-tuned 2560-d embedding. Hard dependency on `flash-attn>=2.6.3` |
| **COBRA** `KatherLab/COBRA` | GPL-3.0 ⚠ | request | CVPR 2025, FM-agnostic. Needs `mamba-ssm` (CUDA compile, no clean wheels) |

### Generative / chat VLMs

These would use the caption/chat protocols — see
[the I/O protocol spec](specs/2026-08-30-model-io-protocol-design.md).

| model | licence | gated | note |
|---|---|---|---|
| **MedMO 4B/8B** `MBZUAI/MedMO-*` | **Apache-2.0** | no | Qwen3-VL backbone, chat + grounding |
| `google/medgemma-1.5-4b-it` | HAI-DEF terms | auto | 233k downloads; explicitly handles multi-patch WSI. The zoo already has `medsiglip`, its vision tower — this is the generative half |
| **PathGen-CLIP** `jamessyx/PathGen-CLIP`, `-L` | ⚠ base card `cc-by-2.0` but requires accepting conditions; `-L` is CC-BY-NC-4.0 | auto | needs `open_clip_torch` |
| `AtlasAnalyticsLab/PathoSynVLM` | CC-BY-NC-SA-4.0 | no | case-level synoptic reports **from precomputed CONCHv1.5 embeddings** — unusually clean fit, no WSI reader needed |
| **Patho-R1** `WenchuanZhang/Patho-R1-{3B,7B}` | CC-BY-NC-ND-4.0 | yes | AAAI 2026, chain-of-thought |
| `General-Medical-AI/SlideChat_Weight` | ⚠ GitHub Apache-2.0, HF card silent | auto | CVPR 2025, but inference goes through `xtuner` + DeepSpeed, not `transformers` |

## Segmentation, cells and utilities

| model | what | licence | gated | note |
|---|---|---|---|---|
| **PathoSAM** `computational-cell-analytics/patho-sam` | nucleus instance + semantic segmentation, automatic **and** interactive | **MIT** | no | **MIDL 2025**, 74 stars, pushed 2026-06-06. Maps almost exactly onto `SegmentationOutput`. Friction: source install, no PyPI wheel |
| **VitaminP** `idso-fa1-pathology/VitaminP` | whole-cell **and** nuclei segmentation; trained on paired H&E–mIF, **H&E-only at inference** | **MIT** (both HF *and* GitHub) | no | actively maintained (2026-08-19) |
| **KongNet** `TIACentre/KongNet_pretrained_weights` | multi-head nuclei detection + classification; PanNuke, CoNIC, **MIDOG**, MONKEY, PUMA heads | CC-BY-NC-SA-4.0 | no | the **MIDOG head closes the mitosis-detection gap** — that capability is absent from the zoo entirely |
| **Special Stain Classifier** `oskarthaeter/special-stains` | 14 stain classes (H&E-FFPE, H&E-FS, PAS, Reticulin, GMS, Congo Red, …) from a thumbnail *or* 40× patches | CC-BY-NC-ND-4.0 | no | H0-mini backbone already in the zoo. Practical: lets a pipeline refuse to run H&E-only models on a PAS slide |
| **HNE2Cell** `roobee79/HNE2Cell` | cell detection + classification from H&E WSI | **Apache-2.0** | no | permissive |
| `NKI-AI/tissue-bg-all-stains` | tissue-vs-background **across all stains** (H&E *and* IHC) | **Apache-2.0** | no | complements `grandqc-tissue`, which is H&E-only. MONAI U-Net, 7.94M params, 12.0 µm/px. Ships one `.pack` and needs `aifocore`, **not yet on PyPI** — but a stock MONAI UNet is re-hostable |
| `RamonK/DistillPath-*` | distilled UNI2-h / Virchow2 / H-optimus-0 students, timm-loadable | "other" | no | efficiency variants of encoders already present |
| **CellViT++** | nuclei seg + classification on FM encoders | ⚠ original weights Apache-2.0, but the `CellViT++` repo licence **forbids resale** | — | the zoo already has `histoplus`/`nulite` from the same family |

**Registration is the biggest missing capability, and it needs a new protocol.**
`DeeperHistReg` (ACROBAT-2023 winner, PyPI + Docker) and `VALIS` (Nat. Commun.,
MIT, PyPI `valis-wsi`) both want `register(moving, fixed) -> transform`, which no
existing protocol expresses. LazySlide already has
[#199](https://github.com/rendeirolab/LazySlide/issues/199) open for it. Highest
value of any new protocol, and also the highest cost — VALIS drags in
Java/Bio-Formats.

## Licences

Permissive licensing is rare in this field. Of everything surveyed, only **KEEP
(MIT)**, **PathoSAM (MIT)**, **VitaminP (MIT)**, **LEMON (MIT)**, **SEQUOIA
(MIT)**, **MedMO (Apache-2.0)**, **HNE2Space (Apache-2.0)**, **HNE2Cell
(Apache-2.0)**, **StainFuser (Apache-2.0)** and **tissue-bg-all-stains
(Apache-2.0)** would register as `commercial=True`. Everything else is NC, ND,
GPL or unlicensed.

### Metadata is unreliable — check both sources

The same trap that bit Classpose (repo `LICENSE` said CC BY-NC, HF card said
MIT). Every candidate here had its HF card *and* its GitHub `LICENSE` checked;
the two disagreed in five cases:

| model | HF card | GitHub |
|---|---|---|
| HistoPrism | CC-BY-NC-**ND**-4.0 | CC BY-NC 4.0 (no `ND`) |
| MuPaD-HE2mIF | `mit` | base model `MuPaD-512` is CC-BY-NC-**ND**-4.0 |
| USIGAN | `mit` | CC-BY-NC-SA-4.0 |
| ELF | GPL-3.0 | **no LICENSE file at all** |
| MoLF / special-stains | CC-BY-NC-ND-4.0 | `NOASSERTION` |

The `ND` clause is the one that matters: NoDerivatives forbids exactly the
vendor-and-adapt pattern used for Classpose and STPath. Any candidate whose two
sources disagree needs an author email before work starts.

### GPL is a hard blocker

This package is MIT and vendoring is the established pattern for awkward
upstreams (classpose, stpath). Copyleft cannot be vendored into it. That rules
out **ELF**, **COBRA**, and two existing backlog items (**HoverNext**, **GHIST**)
unless they can be a runtime dependency instead.

## Status of the existing backlog

Checked directly rather than inferred. Two entries change materially.

| backlog item | licence | weights — verified | status |
|---|---|---|---|
| **HEX** (Nat Med 2026) | no LICENSE file; README says CC-BY-NC-ND | ⚠ **`hex/checkpoint.pth` is 1.20 MB** — exactly the size of the `1024→256→128→40` MLP head and nothing else. The authors' own comment calls it *"pipeline-only, not the paper-trained checkpoints"*, and `load_state_dict(..., strict=False)` swallows the missing MUSK keys | **not integrable.** Architecture is otherwise a perfect ROSIE-shaped fit, so worth an email to the Li lab rather than a write-off |
| **HistoPlexer** | CC-BY-NC-**ND** ⚠ | ✅ **weights exist after all**: `demo/model/pytorch_model.pt`, **80.1 MB**, committed in the repo tree rather than as a release asset, which is why a release-only check misses it | **upgraded.** No FM dependency, but melanoma-only and unmaintained since 2025-03 |
| **HoverNext** | **GPL-3.0** ⚠ | Zenodo, not HF | copyleft blocks vendoring |
| **GHIST** | **GPL-3.0** ⚠ | none found | copyleft blocks vendoring |
| **iSCALE** (Nat Methods 2025) | `NOASSERTION` ⚠ | none found | also needs **paired ST at inference** (mother H&E + daughter ST), so may not fit `feature_prediction` at all |
| **Hist2Cell** | MIT ✅ | README points at a `model_weights/` dir + OneDrive | best-licensed of the backlog |
| **MISO** | no repo found under that name | — | needs a direct link from the paper |
| **VISTA-PATH** | unstated | GitHub exists, code "under preparation", no HF repo | still unreleased |
| **AtlasPatch** | UNVERIFIED | ✅ `AtlasAnalyticsLab/AtlasPatch` is **live** (token required) | **ready to add** |
| **THREADS** (Nature Cancer 2026) | — | `MahmoodLab/threads` 401s, GitHub 404s, TRIDENT registry says "coming soon" | still unreleased |

## Dead ends

### Best-on-paper, never released

- **Atlas 2** (Aignostics, arXiv 2601.05148) — 2B ViT/8 on 5.5M WSIs reporting
  the best numbers in this survey: HEST **44.8** (vs UNI2-h 42.0, H-optimus-0
  43.0), eva morphology **83.3** (vs Virchow2 80.9), robustness **85.7** (vs
  Virchow2 76.0), best on 22/27 tasks. The paper has **no model availability
  section**, and Aignostics' HF org hosts only RudolfV-2/-B/-S, already in the
  zoo. Atlas 2 ≠ RudolfV-2 (2B ViT/8 on 5.5M WSIs vs 1.1B ViT-g/8 on 300k).
  Appears commercial-only — worth a periodic re-check, nothing more.
- **PLUTO-4S / PLUTO-4G** (PathAI) — proprietary.
- **Virchow2G / -mini** (Paige) — Paige confirmed in HF discussions that only
  Virchow2 was open-sourced; access is via an Azure ML embeddings API.
- **PathChat 2 / PathChat+** (Modella AI) — commercial product.
- **DaX** (Alibaba DAMO), **LitePath, Pathryoshka, MRPT, ASTRA, nnMIL, TICON,
  GRACE, CorePath, BRAVE** — 2026 preprints with strong claims and no weights.
- **STORM** (arXiv 2604.03630, 18 organs, 1.2M spots) — only a results gallery
  exists. Highest-value thing to watch in the ST space.

### Available but not worth it

- `google/path-foundation` — ships as TF-Keras/JAX, not PyTorch. Framework
  mismatch for a 2023-era ViT-S that UNI2/Virchow2 comfortably beat.
- `MedSAM2` — 3D/video oriented, "research and education only", weak WSI fit.
- `pennweili/cosie-foundation`, `yandrewl/Eva`, `AI4PATH/XMAG` — **no licence at
  all** (verified: no `license` key in the card front-matter).
- `kaiko-ai/coralbay` — tagged pathology-adjacent but is a **radiology**
  SwinUNETR.
- `vandijklab/C2S-Scale-*` — single-cell LLMs, no H&E input.
- `recursionpharma/nesso`, `OpenPhenom` — cell-painting microscopy, not WSI.
- **VirtualMultiplexer** — MIT, but weights never published; the "release the
  model weights" issue has sat with 0 replies since Sep 2025.
- **StainNet (2021)** `khtao/StainNet` — weights are in-repo and trivially
  usable, but the repository has **no licence whatsoever**. It is a 7 KB model;
  worth an email. (Name collides with the 2026 `JWonderLand/StainNet-Base`.)
- The **Ozcan lab / Pictor Labs** virtual-staining line — commercialised and
  patented (US 11,783,603, US 12,367,691). Permanently out.

### Gaps with no supply — stop looking

**IHC scoring** (HER2 / PD-L1 / Ki67 — active 2025–26 literature, zero open
weights) and **Gleason grading** (PANDA-derived models are mostly Kaggle
artefacts). The existing `grandqc-*`, `pathprofilerqc` and `focuslitenn` entries
already cover slide QC better than anything found.

## Confidence

Benchmark numbers quoted for Mettle, Feather, CARE, EXAONE Path and Digepath are
**self-reported by their own authors**. The only independent comparisons found
are *Nature Biomedical Engineering 2026* (19 FMs, 6,818 patients — CONCH first,
Virchow2 close second) and *Nature Communications 2026* (32 FMs), and the
latter's finding that ensembles win is the strongest external support for
Feather and ELF. Treat single-vendor tables — Mettle's especially — as claims,
not results.

To re-verify any row:

```bash
curl -s https://huggingface.co/api/models/<repo_id>   # gated, cardData.license, siblings
curl -s https://api.github.com/repos/<owner>/<repo>   # license.spdx_id, pushed_at
```

A missing `license` key in the HF card front-matter is what "no licence" means
throughout.
