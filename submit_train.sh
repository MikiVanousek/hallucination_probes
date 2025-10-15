#!/usr/bin/env bash
# submit_train.sh — submit a non-interactive job that clones+trains inside the pod

set -euo pipefail

# --- edit if needed ---
JOB_NAME="probe-train-$(date +%s)"
IMAGE="registry.rcp.epfl.ch/multimeditron/basic:latest-felhassa"
PVC_CLAIM="light-scratch"                       # your existing PVC name
REPO_URL="https://github.com/MikiVanousek/hallucination_probes.git"
REPO_BRANCH="main"
REPO_DIR="/workspace/hallucination_probes"
WORKDIR="/workspace"                            # safe base working dir
MODEL_NAME="${MODEL_NAME:-epfl-llm/meditron-7b}"# use OpenMeditron/Meditron3-8B only if you have gated access
SMOKE_TEST="${SMOKE_TEST:-0}"                   # 1 = 200-step dry run
# ----------------------

# Require secrets in your current shell (so we can inject them)
: "${HF_TOKEN:?Set HF_TOKEN in this shell}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in this shell}"
WANDB_ENTITY="${WANDB_ENTITY:-miki-light}"
WANDB_PROJECT="${WANDB_PROJECT:-hallucination-probes}"

CMD='
set -euo pipefail
# write secrets into the pod (no need to pre-exist on PVC)
cat > /workspace/.env.secrets <<EOF
HF_TOKEN='"$HF_TOKEN"'
WANDB_API_KEY='"$WANDB_API_KEY"'
WANDB_ENTITY='"$WANDB_ENTITY"'
WANDB_PROJECT='"$WANDB_PROJECT"'
EOF
chmod 600 /workspace/.env.secrets

# ensure run script exists; if not, create a minimal one
if [ ! -f '"$REPO_DIR"'/run.sh ]; then
  cat > '"$REPO_DIR"'/run.sh <<'"'BASH'"'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# load secrets
set -a; [ -f /workspace/.env.secrets ] && . /workspace/.env.secrets; set +a
# caches
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-300}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# venv + deps
python3 -m venv .venv || true
source .venv/bin/activate
pip install -U pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  pip install transformers datasets accelerate peft wandb huggingface_hub termcolor
fi

# login non-interactively
python - <<PY
import os
from huggingface_hub import login, HfApi
tok=os.environ["HF_TOKEN"]
login(token=tok, add_to_git_credential=True)
print("HF whoami:", HfApi().whoami(token=tok)["name"])
PY
wandb login --relogin "$WANDB_API_KEY" >/dev/null || true

CFG="configs/meditron.yaml"
if [ "${SMOKE_TEST:-0}" = "1" ]; then
  TMP=$(mktemp /tmp/cfg.XXXX).yaml
  cp "$CFG" "$TMP"
  printf "\nmax_steps: 200\nlogging_steps: 5\n" >> "$TMP"
  CFG="$TMP"
fi

# train (and eval)
python -u -m probe.train --config "$CFG"
python -u -m probe.evaluate --config "$CFG" || true
BASH
  chmod +x '"$REPO_DIR"'/run.sh
fi

cd '"$REPO_DIR"'
SMOKE_TEST='"$SMOKE_TEST"' MODEL_NAME='"$MODEL_NAME"' ./run.sh
'

echo "[submit] job: $JOB_NAME"
runai submit --name "$JOB_NAME" \
  -g 1 \
  -i "$IMAGE" \
  --existing-pvc claimname="$PVC_CLAIM",path="$WORKDIR" \
  --git-sync source="$REPO_URL",branch="$REPO_BRANCH",target="$REPO_DIR" \
  --working-dir "$WORKDIR" \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
  -e HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e WANDB__SERVICE_WAIT=300 \
  --command -- bash -lc "$CMD"

echo
echo "[submit] launched: $JOB_NAME"
echo "[submit] tail logs:  runai logs $JOB_NAME -f"
