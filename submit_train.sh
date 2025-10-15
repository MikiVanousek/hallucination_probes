cd ~/hallucination_probes

cat > submit_train.sh <<'BASH'
#!/usr/bin/env bash
# submit_train.sh — submit a non-interactive Run:AI job that runs ./run.sh in the pod
set -euo pipefail

JOB_NAME="probe-train-$(date +%s)"                           # auto-unique
IMAGE="registry.rcp.epfl.ch/multimeditron/basic:latest-felhassa"
PVC_CLAIM="light-scratch"                                     # your PVC name
WORKDIR="/workspace"                                          # mount point
REPO_URL="https://github.com/MikiVanousek/hallucination_probes.git"
REPO_BRANCH="main"
REPO_DIR="/workspace/hallucination_probes"

# model to use (public by default). Set MODEL_NAME=OpenMeditron/Meditron3-8B if you have gated access.
MODEL_NAME="${MODEL_NAME:-epfl-llm/meditron-7b}"
SMOKE_TEST="${SMOKE_TEST:-0}"                                 # 1 = quick 200-step test

# require secrets in THIS shell so we can write them inside the pod
: "${HF_TOKEN:?Set HF_TOKEN in this shell}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in this shell}"
WANDB_ENTITY="${WANDB_ENTITY:-miki-light}"
WANDB_PROJECT="${WANDB_PROJECT:-hallucination-probes}"

# Command executed inside the pod
CMD='set -euo pipefail
# write secrets for run.sh to read
cat > /workspace/.env.secrets <<EOF
HF_TOKEN='"$HF_TOKEN"'
WANDB_API_KEY='"$WANDB_API_KEY"'
WANDB_ENTITY='"$WANDB_ENTITY"'
WANDB_PROJECT='"$WANDB_PROJECT"'
EOF
chmod 600 /workspace/.env.secrets

# ensure repo exists (git-clone if missing)
mkdir -p "'"$REPO_DIR"'"
if [ ! -d "'"$REPO_DIR"'/.git" ]; then
  apt-get update && apt-get install -y git ca-certificates >/dev/null 2>&1 || true
  git clone --branch "'"$REPO_BRANCH"'" --depth 1 "'"$REPO_URL"'" "'"$REPO_DIR"'"
fi

cd "'"$REPO_DIR"'"
SMOKE_TEST='"$SMOKE_TEST"' MODEL_NAME='"$MODEL_NAME"' ./run.sh
'

echo "[submit] job: $JOB_NAME"
runai submit --name "$JOB_NAME" \
  -g 1 \
  -i "$IMAGE" \
  --existing-pvc claimname="$PVC_CLAIM",path="$WORKDIR" \
  --working-dir "$WORKDIR" \
  -e HF_HOME=/workspace/hf \
  -e TRANSFORMERS_CACHE=/workspace/hf/transformers \
  -e HF_DATASETS_CACHE=/workspace/hf/datasets \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e WANDB__SERVICE_WAIT=300 \
  --command -- bash -lc "$CMD"

echo
echo "[submit] launched: $JOB_NAME"
echo "[submit] tail logs:  runai logs $JOB_NAME -f"
BASH

# fix Windows CRLF if any
sed -i 's/\r$//' submit_train.sh
chmod +x submit_train.sh
