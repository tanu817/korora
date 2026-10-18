source /root/venv/bin/activate
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/korora
cd /workspace/korora
python scripts/attention_maps.py --policy /workspace/outputs/korora_lora/checkpoints/010000/pretrained_model --peft --name finetuned --out /workspace/attention
python scripts/attention_maps.py --policy /workspace/models/flux-3-action-so101-normfix --name zeroshot --out /workspace/attention
