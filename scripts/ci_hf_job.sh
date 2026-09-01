#!/usr/bin/env bash
# Payload for Hugging Face Jobs (issue #20). GitHub Actions and a laptop
# both launch this with:
#   hf jobs run ... IMAGE bash -c "$(cat scripts/ci_hf_job.sh)"
#
# Required env:
#   CI_SHA           git commit to test
# Optional env:
#   CI_BASE_SHA      PR base commit, fetched so --changed-since can diff
#   CI_REPO_URL      default: https://github.com/rendeirolab/lazyslide-models.git
#   CI_PYTEST_ARGS   extra pytest args, e.g. --models plip
#                    or --changed-since <base-sha>
#   CI_DEVICE        cpu (default) or cuda; cuda also probes GPU before pytest
#   CI_XDIST_WORKERS pytest -n (default 4; GPU jobs should pass 1)
set -euo pipefail

REPO_URL="${CI_REPO_URL:-https://github.com/rendeirolab/lazyslide-models.git}"
SHA="${CI_SHA:?CI_SHA is required}"
WORKDIR="${CI_WORKDIR:-/tmp/lazyslide-models}"

if ! command -v git >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq git
fi
# The uv image already has uv. The CUDA pytorch image has neither uv nor curl.
if ! command -v uv >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq curl ca-certificates
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

git clone --filter=blob:none "${REPO_URL}" "${WORKDIR}"
cd "${WORKDIR}"
# Need ancestors of HEAD (and CI_BASE_SHA) so `git diff BASE...HEAD` can
# resolve a merge base. A depth-1 fetch would make --changed-since fall
# back to the whole registry.
git fetch --filter=blob:none origin "${SHA}"
git checkout --force FETCH_HEAD
if [ -n "${CI_BASE_SHA:-}" ]; then
  git fetch --filter=blob:none origin "${CI_BASE_SHA}"
fi

if [ "${CI_DEVICE:-cpu}" = "cuda" ]; then
  export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu124}"
fi

uv sync --dev --group model

if [ "${CI_DEVICE:-cpu}" = "cuda" ]; then
  uv run --no-sync python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.cuda.get_device_name(0))"
  # Slide-encoder tests need flash-attn. Fail honestly if it will not install.
  uv pip install flash-attn --no-build-isolation
fi

# cpu-upgrade advertises many more BLAS threads than it has vCPUs. xdist
# workers each try to spawn a full OpenBLAS pool and then pthread_create fails.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

# Word-split is intentional: CI_PYTEST_ARGS is a space-separated argv fragment.
# Cap workers so a full-registry fallback does not OOM cpu-upgrade (32 GB).
if [ "${CI_DEVICE:-cpu}" = "cuda" ]; then
  case " ${CI_PYTEST_ARGS:-} " in
    *" --device "*) ;;
    *) CI_PYTEST_ARGS="${CI_PYTEST_ARGS:-} --device cuda" ;;
  esac
fi
# shellcheck disable=SC2086
uv run --no-sync pytest tests/ \
  ${CI_PYTEST_ARGS:-} \
  -n "${CI_XDIST_WORKERS:-4}" \
  --dist loadgroup \
  --maxfail=3 \
  -v
