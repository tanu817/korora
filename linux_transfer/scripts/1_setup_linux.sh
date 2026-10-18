#!/usr/bin/env bash
# Sets up LeRobot + LeLab on Linux (Ubuntu/Debian-style). Safe to re-run. Needs sudo for apt + serial-port permissions.
# NOT yet tested on this laptop. If a step fails, paste the error to Claude.
set -euo pipefail

echo "== 1/6 system packages (needs sudo) =="
sudo apt-get update
sudo apt-get install -y git curl ffmpeg v4l-utils build-essential

echo "== 2/6 serial-port + camera permissions =="
sudo usermod -aG dialout,video "$USER"
echo "   (added you to groups dialout+video; this only takes effect after you LOG OUT AND IN, or reboot)"

echo "== 3/6 uv (Python tool installer) =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv --version

echo "== 4/6 LeRobot source =="
mkdir -p "$HOME/dev"
[ -d "$HOME/dev/lerobot" ] || git clone https://github.com/huggingface/lerobot.git "$HOME/dev/lerobot"

echo "== 5/6 Python 3.12 env + LeRobot (feetech motors + dataset extras) =="
uv venv "$HOME/dev/lerobot-env" --python 3.12
# shellcheck disable=SC1091
source "$HOME/dev/lerobot-env/bin/activate"
uv pip install -e "$HOME/dev/lerobot[feetech,dataset]"
python -c "import lerobot, torch; print('lerobot ok | torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

echo "== 6/6 LeLab (graphical app) =="
uv tool install git+https://github.com/huggingface/leLab.git || uv tool upgrade lelab

echo
echo "DONE. Next: log out/in (for serial permissions), then 'source ~/dev/lerobot-env/bin/activate' and 'hf auth login'."
