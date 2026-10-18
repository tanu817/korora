source /root/venv/bin/activate
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/korora
cd /workspace/korora
C=/workspace/outputs/korora_lora/checkpoints
python scripts/eval_displacement.py --out /workspace/displacement.json \
  zero-shot=/workspace/models/flux-3-action-so101-normfix \
  step-1000=$C/001000/pretrained_model:peft \
  step-4000=$C/004000/pretrained_model:peft \
  step-7000=$C/007000/pretrained_model:peft \
  step-10000=$C/010000/pretrained_model:peft \
  step-10000-EMA=$C/010000/pretrained_model_ema:peft
