set -euxo pipefail
export HF_HOME=/workspace/hf UV_CACHE_DIR=/root/uv-cache UV_LINK_MODE=copy
cd /workspace
[ -d lerobot ] || git clone https://github.com/huggingface/lerobot.git
cd lerobot && git fetch -q && git checkout -q 146d8474 && git log -1 --format="%h %cd"
uv venv /root/venv --python 3.12
source /root/venv/bin/activate
uv pip install "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[training,flux3,peft,diffusion]"
uv pip install "natten==0.21.6+torch2110cu128" --find-links https://whl.natten.org/
cd /workspace && ([ -d korora ] && git -C korora pull -q || git clone https://github.com/tanu817/korora.git)
echo SETUP_DONE
rm -rf /root/uv-cache
