#!/usr/bin/env bash
# Dedicated Ollama for this demo on port 11435 (keeps it off the shared instance on 11434).
# Reuses the model files already downloaded for the default Ollama.
set -euo pipefail
if curl -sf -m 2 localhost:11435/api/tags >/dev/null; then echo "Already running on :11435"; exit 0; fi
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}" OLLAMA_KEEP_ALIVE=24h \
  nohup ollama serve > "$(dirname "$0")/ollama.log" 2>&1 &
echo "Started Ollama on :11435 (log: ollama.log)"
