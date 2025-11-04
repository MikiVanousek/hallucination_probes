runai submit \
  --name probe \
  --image registry.rcp.epfl.ch/multimeditron/basic:latest-$GASPAR\
  BASH
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
    TMP=$(mktemp /tmp/cfg.XXXX).yaml; cp "$CFG" "$TMP"; printf "nmax_steps: 200nlogging_steps: 5n" >> "$TMP"; CFG="$TMP"
  fi
  python -u -m probe.train --config "$CFG"
  python -u -m probe.evaluate --config "$CFG" || true
  BASH
  fi

  cd "/workspace/hallucination_probes"
  chmod +x run.sh || true
  # run via bash to bypass noexec volumes
  SMOKE_TEST=0 MODEL_NAME=epfl-llm/meditron-7b bash ./run.sh
  : -c: line 20: syntax error near unexpected token `newline'ight-scratch:/mloscratch \
  --large-shm \
  -e NAS_HOME=/mloscratch/users/$GASPAR \
  -e HF_API_KEY_FILE_AT=/mloscratch/users/$GASPAR/keys/hf_key.txt \
  -e WANDB_API_KEY_FILE_AT=/mloscratch/users/$GASPAR/keys/wandb_key.txt \
  -e GITCONFIG_AT=/mloscratch/users/$GASPAR/.gitconfig \
  -e GIT_CREDENTIALS_AT=/mloscratch/users/$GASPAR/.git-credentials \
  -e VSCODE_CONFIG_AT=/mloscratch/users/$GASPAR/.vscode-server \
  --backoff-limit 3 \
  --run-as-gid 84257 \
  --node-pool h100 \
  --gpu 1 \
  -- sleep infinity
