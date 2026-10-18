# Korora: a robot arm that feeds people who can't use their arms

**ERH Physical AI Hackathon, Zurich · Black Forest Labs track "Imagine, Then Act"**

An SO-101 arm with a spoon fixed to its gripper scoops food from a bowl, brings it to the mouth, tips
it in and returns, bite after bite. Every movement is generated live by **FLUX 3 Action**, BFL's 7B
world action model, fine-tuned on our own teleoperation demos. A small Python harness runs the chain,
and our bridge connects the arm in Zurich to the GPU in the cloud.

> At the hackathon the "mouth" is a paper cup the food is tipped into, never a real face.

## Why it matters

People with paralysis, ALS, MS, stroke or severe arthritis often depend on another person for every
single bite. Meals get scheduled around a caregiver's time, they're slow, and they cost dignity.
Dedicated feeding robots exist, but they're expensive and pre-programmed for fixed bowls and spoons.
Korora shows another route: a low-cost open-source arm that learns the job from a few dozen
demonstrations and is driven by a general-purpose world model. A caregiver could teach it a new bowl,
a new spoon or a new seating position just by demonstrating it.

## What it does

| Step | Instruction FLUX receives |
|---|---|
| 1 | `Scoop food from the bowl` |
| 2 | `Bring the spoon to the mouth` |
| 3 | `Tip the food into the mouth` |
| 4 | `Move the spoon back to the rest position` |
| 5 | repeat for several bites |

The instruction text lives in `feeder/config.py` and is identical in recording, training and rollout.

## How it works

```
 LAPTOP (Zurich): arm + 2 cameras                         CLOUD GPU (RunPod, Romania)
 ┌──────────────────────────────────────────┐            ┌──────────────────────────────┐
 │ HARNESS  feeder/chain.py                  │            │ HANDS  FLUX 3 Action (7B)    │
 │  sends one instruction per step, logs it, │            │  + our LoRA                  │
 │  optional human check + retry, voice      │            │                              │
 │                                           │  ~1/s      │  scene + wrist images        │
 │ BRIDGE  feeder/flux_remote.py  ──────────────────────▶ │  8-tick joint + command      │
 │  reads joints + cameras at 30 Hz,         │  history,  │  history, instruction        │
 │  keeps FLUX's 8-tick history,             │  2 images  │      │                       │
 │  safety limits, executes commands  ◀────────────────── │  imagines the next frames    │
 │                                           │  20 joint  │  and denoises the next 42    │
 │ SO-101 follower                           │  commands  │  joint commands with them    │
 └──────────────────────────────────────────┘            └──────────────────────────────┘
                                                 feeder/flux_server.py
```

- **Hands = FLUX 3 Action.** Input: the scene camera and wrist camera side by side, the robot's joint
  history and the text instruction. It predicts future video frames and joint commands together
  ("imagine, then act"). We execute 20 commands (0.67 s), then it plans again from a fresh picture.
- **Harness = our code.** FLUX doesn't know when a step is done, so the harness gives each step a
  fixed amount of motion time (tuned from the demo lengths), runs the steps in order and logs every
  step to `results/log.csv`. With `--check`, a person judges each step and can trigger a retry.
- **Bridge.** LeRobot's remote inference doesn't support FLUX yet, so we wrote our own. The laptop
  keeps FLUX's 8-tick memory and sends it in one request per chunk. The GPU answers in ~0.75 s, and
  the round trip from Zurich is ~0.9 s.
- **Safety.** FLUX has no limits of its own. The bridge caps each joint at 4° per tick between
  commands, keeps an emergency clamp of 30° from the measured position, holds the taped gripper still,
  moves slowly to a known start pose before every run, and switches the motors off on any connection
  error.

## Results

All runs, zero-shot and fine-tuned, are logged in `results/log.csv`: every live test, what happened
and why. Per-tick trajectories of the bridge runs are in `results/runs/`.

**Zero-shot** (`flux-3-action-so101@c9e13b2` + normalization fix): **0 of 4 skills**, no human
intervention. Asked to scoop, it headed for the food bowl but got stuck on the rim. On the other
instructions it pushed or knocked over the cup.

**Fine-tuned** (LoRA, 136 demos, 10,000 steps):
- *Bring the spoon to the mouth* reached the demonstrated mouth pose in 7 s, within 6° on every joint.
  The demos take 8.2 s.
- *Scoop* turns into the bowl exactly along the demonstrated path (base −5.7° → −31.3°; demos −5.4° →
  −32.4°), but the wrist roll that scoops stays too small. We recorded 29 new demos with a clear roll
  and continued training to step 12,000.
- *Back to rest* moves in the right direction. *Tip* doesn't roll far enough, for the same reason as
  the scoop.

**Trajectory error** (open loop, next 20 steps, degrees; `results/displacement.json`):

| Model | Mean ADE over 4 skills |
|---|---|
| Stand still (baseline) | 2.36 |
| Zero-shot | 3.14 (worse than not moving) |
| Step 1,000 | 2.25 |
| Step 4,000 | 2.06 |
| Step 7,000 | **1.85** |
| Step 10,000 | 1.90 |
| Step 10,000 EMA | 1.96 |

**What FLUX looks at** (`scripts/attention_maps.py`, `results/attention/`): the planned-action tokens
put ~42 % of their attention on the instruction, ~32 % on the frames FLUX imagines for the next
second, and only ~9 % on the current camera images (5–6 % scene, 3–4 % wrist). Inside the image they
follow the spoon tip.

## Problems we hit and how we solved them

Full write-up with evidence for BFL: **[docs/flux_findings.md](docs/flux_findings.md)**.

| Problem | Effect on the arm | Fix |
|---|---|---|
| Released SO-101 normalization in a different joint convention | every joint reading far out of range | `scripts/fix_flux_normalization.py` (organizers' quantiles) |
| Fixed sampling seed + demos that start idle | the arm "waits" forever | fresh seed per request |
| Command history seeded with follower values (gripper 100 vs 0) | 3× worse predictions, shoulder lifts | seed it like a demo start |
| Integrated joint deltas when the arm can't follow | spoon pushed into the table (−130° command) | per-joint lead limit from the demos |
| EMA weights wash out fast motions | weak scoop | raw weights |
| Pink colour cast midway through recording | hovering over the bowl | per-skill colour matching |
| Different start pose | misses the bowl | automatic reset to the demo start pose |
| The recipe's training-time eval | 19 min stall per check | eval disabled |
| Pod network drop-outs | install and training crashes | offline mode, explicit dataset path |
| Cheap wrist camera freezes after ~1.5 h | recording and rollout timeouts | unplug to cool, retries in the bridge |

## Run it

### 1. GPU (Linux, NVIDIA, CUDA 12.8)
See [`pod/README.md`](pod/README.md). In short:
```sh
bash pod/setup.sh && bash pod/download.sh
python scripts/fix_flux_normalization.py models/flux-3-action-so101 models/flux-3-action-so101-normfix
python -m feeder.flux_server --policy-path <checkpoint>/pretrained_model --use-peft   # our LoRA
```

### 2. Robot laptop (LeRobot, SO-101, cameras: scene = 0, wrist = 2)
```sh
ssh -N -L 8001:localhost:8000 root@<pod ip> -p <pod port>                 # tunnel to the GPU
python -m feeder.chain --sim --bites 2                                     # simulation, nothing moves
python -m feeder.flux_remote --task "Scoop food from the bowl" --no-motion # one skill, arm stays still
python -m feeder.flux_remote --task "Scoop food from the bowl"             # one skill, ARM MOVES
python -m feeder.chain --bites 3                                           # full chain, ARM MOVES
python -m feeder.chain --check --steps to_mouth,feed,back_to_rest          # a person judges, retries
```
Every real run first moves slowly to the demo start pose (`--no-reset` skips it). Keep a hand on the
follower's power switch.

### 3. Voice control
`live_transcription.py`: say "feed me" to start the chain and "stop" to stop. Set
`FLUX_SERVER=http://localhost:8001` and `FEEDER_DRY_RUN=false` to drive the arm through the bridge.
Test without microphone or robot: `python live_transcription.py --test`.

### Recording and training
Demos were recorded with LeLab (leader-arm teleoperation), merged with `lerobot-edit-dataset` and
trained with the `lora.json` that ships with the checkpoint. Laptop setup and recording scripts are in
`linux_transfer/`. Datasets (private): `ArnavG07/korora_spoon_feed` (136 demos) and
`ArnavG07/korora_spoon_feed_v2` (+29 roll-focused demos).

## Repository

| Path | What it is |
|---|---|
| `feeder/chain.py` | The harness: bite chain, logging, check and retry, simulation |
| `feeder/flux_server.py` | Bridge, GPU side: FLUX loaded once, LoRA merged, one request per chunk |
| `feeder/flux_remote.py` | Bridge, robot side: arm, cameras, FLUX history, safety, start pose, trajectories |
| `feeder/config.py` | Instructions, step durations, start pose, colour targets, command limits |
| `feeder/flux_hands.py` | Alternative hands with `lerobot-rollout` (needs a local GPU) |
| `live_transcription.py` | Voice control ("feed me" / "stop") |
| `scripts/fix_flux_normalization.py` | SO-101 normalization fix applied to a checkpoint copy |
| `scripts/eval_displacement.py` | ADE/FDE of checkpoints on the demos |
| `scripts/attention_maps.py` | What the action tokens attend to |
| `scripts/seed_start_test.py`, `scripts/server_smoke_test.py` | Seed experiment, server check |
| `pod/` | Scripts used on the GPU pod |
| `docs/flux_findings.md` | Field report on FLUX 3 Action for BFL |
| `results/` | `log.csv` (all runs), `runs/` (trajectories), `displacement.json`, `attention/`, `training_loss.json` |
| `linux_transfer/` | Laptop setup, calibration files, hardware check, recording script |

## Time split (Sat 11:15 → Sun 14:15)

| When | What |
|---|---|
| Sat 11:15 – ~24:00 | Orientation, installing LeRobot and LeLab, moving the arm setup to Linux, calibration, scene and spoon mount |
| Sun 00:00 – 01:10 | Finding arms and cameras, LeLab recording setup, camera fixes |
| Sun 01:10 – 03:30 | Recording 136 demos (4 skills) |
| Sun 03:30 – 05:00 | Dataset merge and upload, cloud GPU setup, normalization fix, bridge |
| Sun 05:00 – 08:30 | LoRA training (2 h 47 min), with zero-shot tests (graded) and checkpoint tests in parallel |
| Sun 08:30 – 11:30 | Live testing and fixing the problems above |
| Sun 11:30 – 12:30 | Evaluation: displacement error, attention maps, findings for BFL |
| Sun 12:30 – 13:45 | 29 roll-focused demos, training continued to step 12,000 |
| Sun 13:45 – 14:15 | Final tests, demo video, slides |

## Team

Team Korora: _names_. Built with LeRobot, LeLab, FLUX 3 Action and RunPod.
