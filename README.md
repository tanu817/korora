# Korora — Robot Spoon-Feeder 🥄

**ERH Physical AI Hackathon · Black Forest Labs track "Imagine, Then Act"**

An SO-101 robot arm feeds a person who can't use their arms: with a spoon fixed to its gripper it
scoops food, brings it to the mouth, tips it in and repeats, bite after bite. Every physical move is performed live by
**FLUX 3 Action**; a small Python harness sequences the steps, checks them and retries.

> At the hackathon the "mouth" is a cup the food is tipped into.

## Task chain

1. Scoop food from the bowl
2. Bring the spoon to the mouth (cup)
3. Tip the food into the mouth
4. Move the spoon back to the rest position
5. Repeat 1–4 for several bites

Instruction text: `feeder/config.py`.

## How FLUX is used

```
 HARNESS (feeder/)                            HANDS (FLUX 3 Action)
 ─────────────────                            ──────────────────────
 next step in the chain ──"Scoop food from     scene + wrist camera
 check it worked           the bowl"──────▶    + joint state + text
 retry if it failed    ◀──── result ───────    → arm actions (replans ~1 s)
```

## Run with FLUX on the real arm ⚠️ moves the arm

FLUX runs on a rented GPU (RunPod); the arm and cameras are on the laptop. A small bridge connects them:
`feeder/flux_server.py` keeps FLUX loaded on the GPU, `feeder/flux_remote.py` drives the arm and sends
FLUX's 8-tick history in one request per ~1 s action chunk.

```sh
# on the GPU pod (zero-shot model, or add --use-peft with our LoRA checkpoint)
python -m feeder.flux_server --policy-path models/flux-3-action-so101-normfix
# on the laptop: tunnel, then the chain
ssh -N -L 8001:localhost:8000 root@<pod ip> -p <pod port>
python -m feeder.chain --sim --bites 2                                    # simulation, nothing moves
python -m feeder.chain --server http://localhost:8001 --bites 3           # autonomous, arm moves
python -m feeder.chain --server http://localhost:8001 --check             # a person judges / retries
python -m feeder.flux_remote --task "Scoop food from the bowl" --no-motion  # one skill, arm stays still
# every real run first moves slowly to the demo start pose (config.START_POSE); --no-reset skips it
```

Safety in the bridge: 4°/tick rate limit between commands, 30° emergency limit, gripper held
(the spoon is taped), start delay to let go of the arm. Every step is appended to `results/log.csv`.

## Results

Zero-shot vs post-training: `results/log.csv`.

## Voice-controlled feeding

`live_transcription.py` uses the externally managed Faster-Whisper model and
connects recognized commands to the same `feeder/` hands interface used by the
rest of the project. Saying `feed me` starts the configured feeding sequence;
`stop` changes back to standby and terminates the active FLUX rollout.

Install the voice-control dependencies with:

```powershell
python -m pip install -r .\requirements.txt
```

Run the safe print-only integration test before using a microphone:

```powershell
python .\live_transcription.py --test
```

This tests the voice-command matching and prints every FLUX rollout command
for the configured number of bites. It does not open the microphone, download
the Whisper model, or move the robot.

The default Whisper configuration uses the CPU, so NVIDIA CUDA Python packages
are not required. If `Config.DEVICE` is changed to `cuda`, install the matching
NVIDIA packages for the selected CUDA version before running the script.

The sequence is safe to try without hardware by default: `FEEDER_DRY_RUN`
defaults to `true`, so each FLUX command is printed instead of executed. On
the robot computer, after checking calibration, cameras and the policy path:

```powershell
$env:FEEDER_DRY_RUN = "false"
$env:FLUX_POLICY_PATH = "models/flux-3-action-so101"
$env:FEEDER_BITES = "3"
python .\live_transcription.py
```

Set `WHISPER_MODEL` to another Faster-Whisper model identifier such as
`medium` to change the externally downloaded speech model. The model is
cached by Faster-Whisper, not stored in this repository.

## Files

| File | What it does |
|---|---|
| `feeder/config.py` | **Edit here**: instruction text + time per step |
| `feeder/interfaces.py` | Hands / Eyes interfaces |
| `feeder/chain.py` | **The harness**: runs the bite chain, logs every step, optional human check + retry |
| `feeder/flux_server.py` | Bridge, GPU side: FLUX loaded once, one request per action chunk |
| `feeder/flux_remote.py` | Bridge, robot side: arm + cameras + FLUX history, safety limits |
| `feeder/flux_hands.py` | Alternative hands: `lerobot-rollout` per subtask (needs a local GPU) |
| `scripts/fix_flux_normalization.py` | Organizers' SO-101 joint normalization fix, applied to a checkpoint copy |
| `results/log.csv` | Zero-shot vs post-training results |

The earlier shell-game version is saved under the git tag `shell-game-archive`.

## Time split

_TODO: fill in at the end (required by the track)._

## Team

_TODO_
