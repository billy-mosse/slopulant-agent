#!/usr/bin/env bash
# CLM stack: Qwen3-8B bf16 encoder on llama.cpp CUDA (:8090, last-token pooling) + clm-serve heads on CPU (:8700).
set -euo pipefail
STACK="${STACK:-$HOME/repos/RINGUSB1/clm-stack}"
BIN="$HOME/llama.cpp-src/llama.cpp-0.5.0/build/bin"
LOGS="$HOME/models"
export HF_HUB_OFFLINE=1 CLM_DEVICE=cpu CLM_CKPT="$STACK/CLM-v0.1-8B/CLM_v0.1-8B.pt"

curl -sf localhost:8090/health >/dev/null || {
  setsid nohup "$BIN/llama-server" -m "$HOME/models/qwen3-8b-bf16.gguf" --alias qwen3-8b \
    --embeddings --pooling last -ngl 999 -c 2048 -np 1 -b 2048 -ub 2048 \
    --host 127.0.0.1 --port 8090 > "$LOGS/encoder.log" 2>&1 < /dev/null &
  until curl -sf localhost:8090/health >/dev/null; do sleep 2; done
}
curl -sf -o /dev/null localhost:8700/ || {
  setsid nohup "$HOME/clm-venv/bin/clm-serve" --host 127.0.0.1 --ckpt "$CLM_CKPT" \
    --emb-url http://127.0.0.1:8090/v1/embeddings > "$LOGS/clm-serve.log" 2>&1 < /dev/null &
  until curl -sf -o /dev/null localhost:8700/; do sleep 1; done
}
echo "encoder: http://127.0.0.1:8090/v1/embeddings   heads + playground: http://127.0.0.1:8700/"
