#!/usr/bin/env bash
# Demo setup on the GPU pod: two FLUX servers side by side (~52 GB of GPU memory).
#   port 8000: LoRA step 12,000 (roll-focused data) -> scoop and tip
#   port 8002: LoRA step 10,000                     -> to the mouth and back to rest
# Loading takes ~2.5 min each; logs in /workspace/server.log and /workspace/server2.log.
set -euo pipefail
source /root/venv/bin/activate
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
cd /workspace/korora
C=/workspace/outputs/korora_lora/checkpoints
pkill -f "python -m [f]eeder.flux_server" || true
sleep 3
nohup python -m feeder.flux_server --use-peft --policy-path $C/012000/pretrained_model --port 8000 > /workspace/server.log 2>&1 &
nohup python -m feeder.flux_server --use-peft --policy-path $C/010000/pretrained_model --port 8002 > /workspace/server2.log 2>&1 &
until grep -q "FLUX loaded" /workspace/server.log && grep -q "FLUX loaded" /workspace/server2.log; do sleep 5; done
echo "both FLUX servers ready"
