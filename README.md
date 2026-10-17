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
4. Move the spoon back to the bowl
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

_TODO: harness entry point._ `feeder/flux_hands.py` runs `lerobot-rollout` once per subtask
(see `CLAUDE.md` for the exact commands).

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
| `feeder/flux_hands.py` | Calls FLUX 3 Action (`lerobot-rollout`) per subtask |
| `results/log.csv` | Zero-shot vs post-training results |

The earlier shell-game version is saved under the git tag `shell-game-archive`.

## Time split

_TODO: fill in at the end (required by the track)._

## Team

_TODO_
