JOB_NAME="probe-train-$(date +%s)"
IMAGE="registry.rcp.epfl.ch/multimeditron/basic:latest-felhassa"
PVC="light-scratch"
WORKDIR="/workspace"
REPO_URL="https://github.com/MikiVanousek/hallucination_probes.git"
REPO_BRANCH="main"
REPO_DIR="/workspace/hallucination_probes"
MODEL_NAME="${MODEL_NAME:-epfl-llm/meditron-7b}"   # switch to OpenMeditron/Meditron3-8B only if you have access
SMOKE_TEST="${SMOKE_TEST:-0}"

# make sure these are set in your shell:
: "${HF_TOKEN:?missing}"; : "${WANDB_API_KEY:?missing}"
WANDB_ENTITY="${WANDB_ENTITY:-miki-light}"
WANDB_PROJECT="${WANDB_PROJECT:-hallucination-probes}"

CMD='set -euo pipefail
# write secrets
cat > /workspace/.env.secrets <<EOF
HF_TOKEN='"$HF_TOKEN"'
WANDB_API_KEY='"$WANDB_API_KEY"'
WANDB_ENTITY='"$WANDB_ENTITY"'
WANDB_PROJECT='"$WANDB_PROJECT"'
EOF
chmod 600 /workspace/.env.secrets

# get repo
mkdir -p "'"$REPO_DIR"'"
if [ ! -d "'"$REPO_DIR"'/.git" ]; then
  apt-get update && apt-get install -y git ca-certificates >/dev/null 2>&1 || true
  git clone --branch "'"$REPO_BRANCH"'" --depth 1 "'"$REPO_URL"'" "'"$REPO_DIR"'"
fi

# if run.sh is missing, create a minimal one
if [ ! -f "'"$REPO_DIR"'/run.sh" ]; then
  cat > "'"$REPO_DIR"'/run.sh" <<'"'BASH'"'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
set -a; [ -f /workspace/.env.secrets ] && . /workspace/.env.secrets; set +a
export HF_HOME=${HF_HOME:-/workspace/hf}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-300}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
python3 -m venv .venv || true
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt || pip install transformers datasets accelerate peft wandb huggingface_hub termcolor
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
  TMP=$(mktemp /tmp/cfg.XXXX).yaml; cp "$CFG" "$TMP"; printf "\nmax_steps: 200\nlogging_steps: 5\n" >> "$TMP"; CFG="$TMP"
fi
python -u -m probe.train --config "$CFG"
python -u -m probe.evaluate --config "$CFG" || true
BASH
fi

cd "'"$REPO_DIR"'"
chmod +x run.sh || true
# run via bash to bypass noexec volumes
SMOKE_TEST='"$SMOKE_TEST"' MODEL_NAME='"$MODEL_NAME"' bash ./run.sh
'

runai submit --name "$JOB_NAME" \
  -g 1 \
  -i "$IMAGE" \
  --existing-pvc claimname="$PVC",path="$WORKDIR" \
  --working-dir "$WORKDIR" \
  -e HF_HOME=/workspace/hf \
  -e TRANSFORMERS_CACHE=/workspace/hf/transformers \
  -e HF_DATASETS_CACHE=/workspace/hf/datasets \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e WANDB__SERVICE_WAIT=300 \
  --command -- bash -lc "$CMD"

echo "launched: $JOB_NAME"
echo "tail logs: runai logs $JOB_NAME -f"
