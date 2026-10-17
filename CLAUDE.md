# CLAUDE.md — Korora: Robot Spoon-Feeder (ERH Physical AI Hackathon)

## Who we are
- Team of beginners at the ERH Physical AI Hackathon (Zurich), Black Forest Labs track "Imagine, Then Act".
- We have NO prior experience with robot learning, LeRobot, or world models.
- **Explain things simply, in short chunks. Explain what a command does before running it.**

## The track rules (from the brief)
- Make an SO-101 robot arm do a **long, multi-step task** using the **FLUX 3 Action** world model (open 7B World Action Model).
- **FLUX must be in the loop**: the arm must be driven live by FLUX, not by replayed recordings.
- The task must run **autonomously**.
- **Longer chains score higher.** We may use our own data (recorded with the leader arm).
- We must **justify our time split** in the pitch.

### Judging weights
| Criterion | Weight |
|---|---|
| Demo works on real hardware | 25% |
| Technical difficulty | 20% |
| Use of track tech (FLUX) | 20% |
| Idea and originality | 15% |
| Real-world use | 10% |
| Demo video | 10% |

**Reliability beats ambition.** A working shorter demo beats a broken long one.

### Evaluation (from the organizer track PDF): BOTH are graded
- **Zero-shot:** run the released `flux-3-action-so101` checkpoint on our skills WITHOUT fine-tuning. Record which checkpoint (revision) was used, what worked, and where it failed.
- **Post-training:** fine-tune on our demos, then test under the SAME conditions. Show what improved, what still fails, and whether a human had to intervene.
- Keep a log of: successful attempts, total attempts, completed task steps, recoveries, interventions. → `results/log.csv`

## Compute situation (as of Sat 12:45)
- **No track compute.** AWS access is promised "later" → use it for LoRA training (needs a large NVIDIA GPU, ideally 48–80 GB, Ubuntu).
- **Teammate's Lenovo Legion** (Windows + WSL2 Ubuntu): for zero-shot runs/rollout only if its GPU has enough VRAM (~24 GB+). Being set up now.
- **Tanu's laptop** (Windows on ARM, no NVIDIA GPU): code, brain/sim, git. Cannot run FLUX.
- Recorded datasets must be **pushed to the Hugging Face Hub** so AWS can pull them for training.
- Calibration files must be **copied** to whichever machine runs the arm (same robot id `so101`, leader id `leader`).

## Hardware
- SO-101 follower arm + SO-101 leader arm (for teleoperation / recording demos). **Not yet calibrated.**
- The SO-101 has one camera mounted (assumed wrist). **We still need a second, fixed USB webcam for the scene view.**
- **TWO cameras required by the checkpoint:**
  - fixed **scene** camera → key `observation.images.scene` (left half of model image)
  - **wrist** camera on the gripper → key `observation.images.wrist` (right half)
  - Each is resized to 256×256 and joined into one 512×256 image. A swapped camera will NOT raise an error — visually verify scene=left, wrist=right.
- Scene camera taped in place, desk lamp for constant light.
- Laptop may need a USB-A-to-USB-C adapter with USB 3.0 data.
- Software stack: **LeRobot + `Flux3Policy`**, recording/calibration via **LeLab** (GUI). See the "FLUX 3 Action reference" section below — do not guess APIs.

## The task: Robot Spoon-Feeder
An SO-101 arm feeds a person who can't use their arms (e.g. paralysis): it scoops food with a spoon
and brings it to the mouth, bite after bite. **At the hackathon we use a mannequin head / marked
target as the "mouth" — never a real person's face.**

### Task chain (draft — instruction text lives in `feeder/config.py`)
1. Pick up the spoon
2. Scoop food from the bowl
3. Bring the spoon to the mouth (keep it level, don't spill)
4. Move the spoon back to the bowl
5. Repeat 2–4 for several bites (longer chain = higher score)
6. Put the spoon down

### Scene
- Food that scoops reliably with little mess (puffed rice, lentils, cereal).
- Spoon with a thick handle the gripper can hold.
- Tape-mark bowl, spoon rest and mouth target so every run starts the same.
- Scene camera must see bowl, spoon and target at once.

### Keep the instruction vocabulary SMALL
- BFL's SO-101 example needed **~200 demos for a handful of related pick-and-place tasks**.
- Every distinct text instruction FLUX must learn costs demos. ~5 instructions is the plan.
- Instruction text must be **identical** between recording, training and rollout (same wording, same casing).
- Scooping is likely the hardest skill → expect it to need the most demos.

### Architecture: "harness" + "hands"
Organizers' recommended **"Option B: Harness + FLUX"**: FLUX does atomic skills, a simple Python harness sequences, checks and recovers.
- **Harness (our code, `feeder/`):** runs the chain, sends one instruction per subtask, checks success after each (human check first, camera later), retries a failed step.
- **Hands (FLUX 3 Action, `feeder/flux_hands.py`):** scene + wrist images, joint state and text instruction → chunk of arm actions. Fine-tuned (LoRA) on our teleop demos.
- FLUX does NOT know when a subtask is "done" and does NOT notice a failed grasp until it re-plans — **the harness must check success after every subtask**.

### Repo structure
```
feeder/
  config.py       # instruction text + per-step durations
  interfaces.py   # Hands / Eyes protocols
  flux_hands.py   # runs FLUX (lerobot-rollout) per subtask
results/log.csv   # zero-shot vs post-training results (graded)
```
The old shell-game code is kept in git under the tag `shell-game-archive`.

## Development rules
- **Build a simulation mode first** so the brain can be developed and tested without the robot.
- **Small steps.** One working piece at a time; test it before moving on.
- **Never run code that moves the real arm without telling us first** and explaining what motion to expect. Default to slow speeds.
- Respect motor limits: SO-101 servos overheat with long continuous use — suggest breaks.
- Commit to git often with clear messages. Don't commit datasets, model weights, or videos (use .gitignore).
- Keep dependencies minimal. Pin versions once things work.
- If something fails twice the same way, stop and explain the problem instead of trying random fixes.
- **Feature freeze: Sunday ~10:15.** After that, only bug fixes — no new features or refactors.

## Timeline (Sat 11:15 → Sun ~14:15)
| Phase | Time | Goal |
|---|---|---|
| 0 Orientation | Sat 11:15–12:45 | Install, download weights, calibrate in LeLab, teleop works, both cameras OK |
| 1 Physical setup + ZERO-SHOT | 12:45–14:15 | Scene, bowl, spoon, lighting; run base checkpoint on each subskill, log results (graded!) |
| 2 Walking skeleton | 14:15–16:15 | ~10 demos of one skill → short LoRA fine-tune → run on arm; time the training |
| 3 Data collection | 16:15–21:15 | 30–50 demos per skill type |
| 4 Training + brain | 21:15–01:15 | Main training run; brain + vision + sim done |
| Sleep | 01:15–05:15 | Overnight training run |
| 5 Test & fix | 05:15–08:15 | Per-skill success rates; extra demos for failures |
| 6 Full chain | 08:15–10:15 | End-to-end runs; FREEZE at 10:15 |
| 7 Video/slides/repo | 10:15–12:15 | MP4 demo video, deck, README |
| 8 Submit & rehearse | 12:15–14:15 | Submit by 13:00 |

### Fallback levels
- **Bronze:** one FLUX-driven scoop + bring to mouth.
- **Silver:** full chain, one bite (pick spoon → scoop → mouth → put down).
- **Gold:** several bites in a row + retry when a scoop fails.

## Submission requirements
- GitHub repository (this repo), slide deck, demo video (**MP4**), team name.
- README must explain: what it does, how to run it, how FLUX is used, and our time split.

## FLUX 3 Action reference (read before writing FLUX/LeRobot code)

### Links
- SO-101 + LeRobot guide (PRIMARY): https://docs.bfl.ai/flux_3/flux3_action_so101
- Model card: https://huggingface.co/black-forest-labs/flux-3-action-so101
- Collection (base, so101, droid models): https://huggingface.co/collections/black-forest-labs/flux-3-action
- Blog / how it works: https://huggingface.co/blog/black-forest-labs/flux-3-action
- Overview: https://docs.bfl.ai/flux_3/flux3_action_overview
- Inference: https://docs.bfl.ai/flux_3/flux3_action_inference · Fine-tuning: https://docs.bfl.ai/flux_3/flux3_action_finetuning
- Code: https://github.com/black-forest-labs/flux-action · LeRobot docs: https://huggingface.co/docs/lerobot/flux3
- Sim demo Space: multimodalart/flux-3-action-so101-sim

### What the model is
- 7B diffusion transformer "world action model". Frozen video VAE encodes camera frames; frozen Qwen3-VL-4B encodes the text instruction.
- Predicts **future video frames and actions together** (denoised jointly) — that's the "imagine". At control time, frame decoding is skipped; the current API returns actions.
- **Inputs:** scene + wrist camera (512×256 composite), robot state (6 joints), text instruction.
- **Outputs (SO-101):** 42 actions predicted, first **32 executed (~1.07 s at 30 Hz)**, then observe again and replan.
- Sampling: 4 Euler steps, shift 6.93, guidance 3, seed 42, BF16.
- **No safety limits:** the model does not bound velocity, force or workspace. Keep a hand on the power/stop. Our code must enforce limits.

### Data contract (must be identical in recording, training and rollout)
- LeRobot **v3** dataset at **30 Hz**.
- 6 joints in this order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
- Units: **degrees** for the 5 arm joints, **percent** for gripper (`--robot.use_degrees=true`).
- Same robot id + calibration as training. Use the saved processors/normalization from the checkpoint; do NOT normalize twice.
- Camera keys `scene` and `wrist`. If recorded as `top`/`gripper_cam`, use:
  `--rename_map='{"observation.images.top":"observation.images.scene","observation.images.gripper_cam":"observation.images.wrist"}'` (same at rollout).
- Every episode includes the task instruction text.

### Install (Linux, Python 3.12, NVIDIA GPU with CUDA 12.8 driver) — likely on the track compute
```sh
git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv venv --python 3.12
source .venv/bin/activate
uv pip install 'torch==2.11.0' 'torchvision==0.26.0' --index-url https://download.pytorch.org/whl/cu128
uv pip install -e '.[training,flux3,peft,diffusion]'
uv pip install 'natten==0.21.6+torch2110cu128' --find-links https://whl.natten.org/
ffmpeg -version
```

### Download weights
```sh
hf auth login
hf download black-forest-labs/flux-3-action-so101 \
  --revision c9e13b2aca6a0a472b3ea03fd90cff32d3e85849 \
  --local-dir models/flux-3-action-so101
hf download black-forest-labs/flux-3-action-base \
  --revision 62878e2925e59b7a89ec14463ce89932624c490d \
  --include 'video_vae.safetensors' --include 'text_encoder/*' \
  --local-dir models/flux-3-action-base
policy_dir="$(pwd)/models/flux-3-action-so101"
encoders_dir="$(pwd)/models/flux-3-action-base"
```
Sanity check: load `models/flux-3-action-so101/config.json` and print `camera_keys, camera_layout, fps, chunk_size, n_action_steps`; assert fps == 30.

### Zero-shot / rollout (THIS MOVES THE ARM — confirm both camera views first)
```sh
lerobot-rollout \
  --strategy.type=base --inference.type=sync \
  --policy.path=<checkpoint path> [--policy.use_peft=true for our LoRA] --policy.device=cuda \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101 \
  --robot.use_degrees=true \
  --robot.cameras='{"scene":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":1,"width":640,"height":480,"fps":30}}' \
  --fps=30 --task="<instruction>" --duration=30
```
- Zero-shot: `--policy.path=$policy_dir` without `use_peft`.
- Only synchronous rollout is supported. The base strategy does not record episodes.
- For the harness, prefer calling the policy from Python (`Flux3Policy.from_pretrained(...)` + `make_pre_post_processors(...)`) so we can switch `--task` per subtask without restarting.

### LoRA fine-tune (one GPU)
```sh
python -m lerobot.scripts.lerobot_train \
  --config_path=examples/flux3/lora.json \
  --policy.path="$policy_dir" --policy.device=cuda \
  --policy.video_vae_id="$encoders_dir/video_vae.safetensors" \
  --policy.text_encoder_id="$encoders_dir/text_encoder" \
  --dataset.repo_id=YOUR_ORG/YOUR_SO101_DATASET \
  --output_dir=outputs/so101_lora
```
- Preset: LoRA rank/alpha 32, adapter LR 1e-4, head LR 5e-4, BF16, grad checkpointing, EMA 0.999, batch 2 × grad-accum 4, **10,000 microsteps (= 2,500 optimizer updates)**.
- Resume: `lerobot-train --config_path=outputs/so101_lora/checkpoints/last/pretrained_model/train_config.json --resume=true`
- Compare `pretrained_model` vs `pretrained_model_ema` on identical tests — EMA is not automatically better.
- Reference: ~200 teleop demos learned "put the blue box into the container".

### Known pitfalls
- Swapped cameras fail silently.
- Changed calibration / robot id between recording and rollout breaks the policy.
- After a failed grasp, the arm keeps executing the old plan until the next replan (~1 s).
- Training time on our compute is unknown — measure it with the walking-skeleton run.

## Open questions for mentors
- [ ] How do we access track compute, and how long do 10k LoRA steps take there?
- [ ] Is the SO-101 already calibrated in LeLab, and which robot id / port?
- [ ] Recommended instruction wording / how many distinct instructions per LoRA?
- [ ] Can we decode FLUX's imagined future frames for our demo video (flux-action repo)?
- [ ] Can one LoRA learn all our subskills, or should we train per-skill?
