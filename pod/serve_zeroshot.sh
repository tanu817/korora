source /root/venv/bin/activate
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
cd /workspace/korora
python -m feeder.flux_server --policy-path /workspace/models/flux-3-action-so101-normfix
