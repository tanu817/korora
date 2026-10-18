set -euxo pipefail
export HF_HOME=/workspace/hf UV_CACHE_DIR=/root/uvx-cache HF_XET_HIGH_PERFORMANCE=1
HF="/root/venv/bin/hf"
for i in 1 2 3 4 5; do $HF download black-forest-labs/flux-3-action-so101 --revision c9e13b2aca6a0a472b3ea03fd90cff32d3e85849 --local-dir /workspace/models/flux-3-action-so101 && break; sleep 15; done
for i in 1 2 3 4 5; do $HF download black-forest-labs/flux-3-action-base --revision 62878e2925e59b7a89ec14463ce89932624c490d --include "video_vae.safetensors" --include "text_encoder/*" && break; sleep 15; done
echo DOWNLOAD_DONE
