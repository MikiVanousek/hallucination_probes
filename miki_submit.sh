if [ -z "$1" ]; then
  echo "Error: No argument provided"
  echo "Usage: $0 <config path>"
  exit 1
fi

PROBE_NUM=1
while runai list | grep -q "probe${PROBE_NUM}"; do
  ((PROBE_NUM++))
done

JOB_NAME="probe${PROBE_NUM}"
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
  --backoff-limit 3 \
  --run-as-gid 84257 \
  --node-pool h100 \
  --gpu 1 \
  -- bash -c "cd /mloscratch/users/vanousek/hallucination_probes && git pull origin main && ./run.sh $1"
