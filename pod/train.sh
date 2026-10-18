set -euxo pipefail
source /root/venv/bin/activate
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
cd /workspace
P=/workspace/models/flux-3-action-so101-normfix
lerobot-train --config_path=$P/lora.json \
  --policy.path=$P \
  --dataset.repo_id=ArnavG07/korora_spoon_feed --dataset.root=/workspace/hf/lerobot/ArnavG07/korora_spoon_feed --dataset.eval_split=0.0 --eval_steps=0 \
  --output_dir=/workspace/outputs/korora_lora --job_name=korora-lora
echo TRAIN_DONE
