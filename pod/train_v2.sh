set -euxo pipefail
source /root/venv/bin/activate
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
cd /workspace
lerobot-train --config_path=/workspace/outputs/korora_lora/checkpoints/last/pretrained_model/train_config.json --resume=true
echo TRAIN_DONE
