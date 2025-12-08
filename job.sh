#!/usr/bin/env bash
# Usage: ./submit_job.sh <job_name> <num_gpus>
# Example: ./submit_job.sh download-data 0

set -e  # Exit immediately on error

# === Parameters ===
JOB_NAME="$1"
NUM_GPUS="$2"

# === Validation ===
if [ -z "$JOB_NAME" ] || [ -z "$NUM_GPUS" ]; then
  echo "Usage: $0 <job_name> <num_gpus>"
  exit 1
fi

# === User-specific config ===
export GASPAR=lklein
IMAGE="registry.rcp.epfl.ch/multimeditron/basic:latest-$GASPAR"
PVC_NAME="light-scratch"
#MOUNT_PATH="/mloscratch"
MOUNT_PATH="/mnt/light/scratch"
USER_HOME="$MOUNT_PATH/users/$GASPAR"
NODE_POOL="h100"

# ToDo: add back when VSCode config is needed
# -e GITCONFIG_AT="$USER_HOME/.gitconfig" \
# -e GIT_CREDENTIALS_AT="$USER_HOME/.git-credentials" \
# -e VSCODE_CONFIG_AT="$USER_HOME/.vscode-server" \


# === Submit the job ===
runai submit \
  --name "$JOB_NAME" \
  --image "$IMAGE" \
  --pvc "$PVC_NAME:$MOUNT_PATH" \
  --large-shm \
  -e HOME="$USER_HOME" \
  -e NAS_HOME="$USER_HOME" \
  -e HF_TOKEN_AT="$USER_HOME/keys/hf_key.txt" \
  -e HF_API_TOKEN_AT="$USER_HOME/keys/hf_key.txt" \
  -e WANDB_API_KEY_FILE_AT="$USER_HOME/keys/wandb_key.txt" \
  --backoff-limit 0 \
  --run-as-gid 84257 \
  --node-pool "$NODE_POOL" \
  --gpu "$NUM_GPUS" \
  -- sleep infinity

