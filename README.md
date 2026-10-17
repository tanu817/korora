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
