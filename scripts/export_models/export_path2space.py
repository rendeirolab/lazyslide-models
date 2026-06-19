#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pooch>=1.8",
#   "tqdm",
#   "torch>=2.7.1",
# ]
# ///
#
# Export the Path2Space MLP ensemble (Shulman et al., Cell 2026).
#
# Path2Space is CTransPath -> 154-MLP ensemble -> per-tile gene expression
# (~14K genes). LazySlide already ships CTransPath separately, so this script
# exports ONLY the MLP ensemble as a single fused nn.Module whose forward()
# averages the 154 member outputs.
#
# Upstream: https://github.com/eldadshulman/path2space-companion
# Pinned commit: dd69f503e0f290ff4c5628ce2f723cbd7c638808
# Weights: https://doi.org/10.5281/zenodo.20174301  (Apache 2.0)

from pathlib import Path

from export_utils import export_model, verify_exported

workdir = Path(__file__).parent
checkpoint_dir = workdir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

export_artifacts = workdir / "export_artifacts"
export_artifacts.mkdir(parents=True, exist_ok=True)

PATH2SPACE_EXPORT_PATH = export_artifacts / "Path2Space_exported.pt2"
PATH2SPACE_GENES_PATH = export_artifacts / "Path2Space_genes.txt"

# %%
import pooch
import torch
import torch.nn as nn

ZENODO_BASE = "https://zenodo.org/records/20174301/files"

# Hashes published at https://zenodo.org/records/20174301 (MD5SUMS.txt).
# Left as None so the script runs even before hashes are pinned; replace with
# 'md5:...' strings to enforce integrity.
mlp_archive = pooch.retrieve(
    url=f"{ZENODO_BASE}/mlp_ensemble.tar.gz",
    known_hash=None,
    fname="mlp_ensemble.tar.gz",
    path=str(checkpoint_dir),
    processor=pooch.Untar(extract_dir="mlp_ensemble_extracted"),
    progressbar=True,
)

genes_file = pooch.retrieve(
    url=f"{ZENODO_BASE}/genes.txt",
    known_hash=None,
    fname="genes.txt",
    path=str(checkpoint_dir),
    progressbar=True,
)

# pooch.Untar returns the list of extracted file paths; locate the ckpts.
ckpt_paths = sorted(Path(p) for p in mlp_archive if Path(p).name == "model_trained.pth")
assert len(ckpt_paths) == 154, f"expected 154 ensemble ckpts, found {len(ckpt_paths)}"

genes = [
    line.strip() for line in Path(genes_file).read_text().splitlines() if line.strip()
]
n_genes = len(genes)
print(f"loaded {n_genes} genes, {len(ckpt_paths)} ckpts")


# %%
# Inlined verbatim from ge_model/path2space/model_mlp.py. State-dict keys
# (layer0.0.*, layer1.0.*) must stay byte-identical for load_state_dict.
class MLP_regression_relu_two(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_hiddens: int,
        n_outputs: int,
        dropout: float,
        bias_init: torch.Tensor | None = None,
    ):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Linear(n_inputs, n_hiddens),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layer1 = nn.Sequential(
            nn.Linear(n_hiddens, n_outputs),
            nn.ReLU(),
        )
        if bias_init is not None:
            with torch.no_grad():
                self.layer1[0].bias = nn.Parameter(bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        return x


class Path2SpaceEnsemble(nn.Module):
    """Fused 154-MLP ensemble.

    Input:  (B, 768)   CTransPath features (raw, no extra normalisation).
    Output: (B, n_genes), non-negative (member outputs pass through ReLU
            before averaging).

    Upstream averages inner-folds (7) then outer-folds (22); since every
    outer group has the same inner size, the nested mean is algebraically
    identical to a single mean over all 154 members.
    """

    def __init__(self, n_genes: int, n_members: int = 154):
        super().__init__()
        self.members = nn.ModuleList(
            [
                MLP_regression_relu_two(
                    n_inputs=768,
                    n_hiddens=768,
                    n_outputs=n_genes,
                    dropout=0.0,  # irrelevant under eval(); cheaper export graph
                    bias_init=None,
                )
                for _ in range(n_members)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = torch.stack([m(x) for m in self.members], dim=0)
        return outs.mean(dim=0)


model = Path2SpaceEnsemble(n_genes=n_genes, n_members=len(ckpt_paths))
for member, ckpt in zip(model.members, ckpt_paths):
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    member.load_state_dict(state, strict=True)
model.eval()

# Match upstream init_random_seed(42) numerics (TF32 off).
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Persist the gene list alongside the artifact for downstream consumers.
PATH2SPACE_GENES_PATH.write_text("\n".join(genes) + "\n")

# %%
# Only batch is dynamic. Feature dim is fixed at 768 by CTransPath.
dynamic_shapes = [{0: torch.export.Dim.AUTO}]
example_input = torch.randn(2, 768)
export_model(
    model,
    example_input,
    PATH2SPACE_EXPORT_PATH,
    dynamic_shapes=dynamic_shapes,
)

# %%
torch.manual_seed(42)
fixed_input = torch.randn(4, 768)
verify_exported(model, PATH2SPACE_EXPORT_PATH, fixed_input, "Path2Space")
