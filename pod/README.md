# GPU pod scripts (RunPod, RTX PRO 6000 96 GB, Ubuntu 24.04)

The arm runs on the laptop; FLUX runs here. Order used on Sun 18 Oct 2026:

| Script | What it does |
|---|---|
| `setup.sh` | LeRobot at commit `146d8474` (same as the laptop), PyTorch 2.11 + CUDA 12.8, NATTEN, clones this repo. Keep the uv cache on local disk: unpacking wheels on the network volume fails. |
| `download.sh` | `flux-3-action-so101@c9e13b2` and the pinned encoders from `flux-3-action-base@62878e2` (gated: accept both licences on Hugging Face first). Then run `scripts/fix_flux_normalization.py` to make the `-normfix` copy. |
| `train.sh` | LoRA fine-tune, 10,000 steps, `korora_spoon_feed` (136 demos). Offline mode and an explicit `--dataset.root` because the pod's network drops out; eval disabled (`--eval_steps=0 --dataset.eval_split=0.0`). |
| `train_v2.sh` | Resumes from step 10,000 on `korora_spoon_feed_v2` (+29 new scoop/tip demos, upweighted 3x) to step 12,000; the checkpoint's `train_config.json` is edited to point at v2. |
| `serve_zeroshot.sh` / `serve_lora.sh <checkpoint>` | Starts `feeder/flux_server.py` on localhost:8000. Reach it from the laptop with `ssh -N -L 8001:localhost:8000 root@<pod> -p <port>`. |
| `eval.sh` | `scripts/eval_displacement.py` over zero-shot and checkpoints 1k/4k/7k/10k/10k-EMA. |
| `attn.sh` | `scripts/attention_maps.py` for the fine-tuned and zero-shot models. |

All paths assume the network volume at `/workspace` and this repo at `/workspace/korora`.
