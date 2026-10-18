source /root/venv/bin/activate
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
cd /workspace/korora
python -m feeder.flux_server --use-peft --policy-path "$1"
