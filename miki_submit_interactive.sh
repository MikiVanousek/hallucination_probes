# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
  echo "Warning: You have uncommitted changes."
  git status --porcelain
  read -p "Do you want to continue anyway? [y/N]: " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Submission cancelled."
    exit 1
  fi
fi

# Check for unpushed commits
UNPUSHED=$(git log origin/main..HEAD --oneline)
if [ ! -z "$UNPUSHED" ]; then
  echo "Warning: You have unpushed commits:"
  echo "$UNPUSHED"
  read -p "Do you want to continue anyway? [y/N]: " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Submission cancelled."
    exit 1
  fi
fi

JOB_PREFIX="interprobe"
JOB_NUM=1
while runai list | grep -q "${JOB_PREFIX}-${JOB_NUM}"; do
  ((JOB_NUM++))
done

JOB_NAME="${JOB_PREFIX}-${JOB_NUM}"
echo "Submitting job: $JOB_NAME"

runai submit \
  --name $JOB_NAME \
  --image registry.rcp.epfl.ch/multimeditron/basic:latest-$GASPAR\
  --pvc light-scratch:/mloscratch \
  --large-shm \
  -e NAS_HOME=/mloscratch/users/$GASPAR \
  -e HF_API_KEY_FILE_AT=/mloscratch/users/$GASPAR/keys/hf_key.txt \
  -e WANDB_API_KEY_FILE_AT=/mloscratch/users/$GASPAR/keys/wandb_key.txt \
  -e GITCONFIG_AT=/mloscratch/users/$GASPAR/.gitconfig \
  -e GIT_CREDENTIALS_AT=/mloscratch/users/$GASPAR/.git-credentials \
  -e VSCODE_CONFIG_AT=/mloscratch/users/$GASPAR/.vscode-server \
  -e HOVEN='hoven' \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e HF_TOKEN=$HF_TOKEN \
  -e HF_WRITE_TOKEN=$HF_TOKEN \
  --backoff-limit 0 \
  --run-as-gid 84257 \
  --node-pool h100 \
  --gpu 1 \
  -- sleep infinity
