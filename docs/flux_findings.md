# FLUX 3 Action on a real SO-101: what we learned (Korora, spoon-feeding)

Team Korora, ERH Physical AI Hackathon, Black Forest Labs track. This is a field report on running
`black-forest-labs/flux-3-action-so101` (revision `c9e13b2`) zero-shot and fine-tuned on a real,
contact-rich, multi-step task. Every point below comes with the evidence we measured; raw logs are in
`results/log.csv`, `results/runs/` (per-tick trajectories) and `results/displacement.json`.

## Setup

| | |
|---|---|
| Task | Spoon fixed to the gripper: scoop food, bring it to a "mouth" cup, tip it in, return to rest |
| Robot | SO-101 follower + leader, scene camera (top-down) + wrist camera, LeRobot `use_degrees=true` |
| Data | 136 teleop demos, 4 instructions (49 scoop, 48 to mouth, 19 tip, 20 back to rest), 30 Hz, ~18 min |
| Fine-tune | `lora.json` from the checkpoint: LoRA r32, batch 2 x accum 4, 10,000 steps, all episodes (no eval split) |
| Compute | 1x RTX PRO 6000 Blackwell 96 GB (RunPod EU). Training 1.0 step/s, 2 h 47 min, 30 GB VRAM |
| Inference | Our own bridge (LeRobot remote inference doesn't support FLUX.3 yet): 0.75 s per chunk on the GPU, ~0.9 s round trip from Zurich |

## Headline results

- **Zero-shot: 0/4 skills.** The base checkpoint moved toward the right object for scooping, but never
  performed a skill; it got stuck on the bowl rim, pushed the mouth cup and knocked it over.
- **After fine-tuning, the model reproduces the demos open-loop** (scoop: mean displacement error 1.3 deg
  over 20 steps vs 2.0 deg for "stand still"), and live it now drives the spoon into the food bowl
  along the demonstrated path. It does not yet complete the scoop (the wrist roll stalls in the food).
- To get there we had to work around **eight model-level issues** below. Each one silently produced
  plausible-looking but wrong behaviour; none raised an error.

## Issues found, with evidence

Each entry: what we saw → evidence → which part of the model → impact → our workaround → suggestion.

### 1. Released SO-101 normalization is in a different joint convention
- **Saw:** the organizers flagged it; we verified it on the checkpoint files.
- **Evidence:** saved `state.q01/q99` for shoulder_lift = +84.8 / +184.2 deg; our arm at rest reads -99.9 deg
  in LeRobot degrees. The values match a flipped shoulder with +-90 deg offsets (the sim's old
  `lerobot_to_model` mapping).
- **Model part:** checkpoint-owned processor state (history normalizer quantiles).
- **Impact:** every joint reading is normalized far outside its range; zero-shot is meaningless without the fix.
- **Workaround:** `scripts/fix_flux_normalization.py` writes the corrected quantiles into a copy of the
  checkpoint (used for zero-shot, training and rollout).
- **Suggestion:** ship processors in raw LeRobot degrees, and warn at load time when the live state lies
  far outside q01..q99.

### 2. A fixed sampling seed deadlocks the policy at the start of a skill
- **Saw:** with everything else fixed, the arm "stuttered in place" for the whole run.
- **Evidence:** our scoop demos start with a median 1.7 s of standing still (0-5.4 s) before the leader
  moves. From a still start, `inference_seed` (fixed) returns the same "keep waiting" plan every chunk.
  With a fresh seed per request the planned motion from the same input varies from 0.9 to 5.4 deg, and
  live the arm starts on its own after ~3 s.
- **Model part:** sampler (`inference_seed` used for every call) x history conditioning.
- **Impact:** the robot never starts a skill whose demos begin idle.
- **Workaround:** random seed per request in our server.
- **Suggestion:** default to fresh noise per call (or seed + call counter); document trimming idle starts.

### 3. Past-command conditioning is brittle to how the history is seeded
- **Saw:** at a still start the arm lifted its shoulder, then fell back.
- **Evidence:** recorded "commands" are leader positions: gripper 0 % (taped follower reads 100 %),
  shoulder ~2 deg below the follower. Open-loop on scoop demos, seeding the command history with follower
  values (as a naive deployment does) tripled the error (1.3 -> 3.8 deg) and made the shoulder move
  (error 1.0 -> 2.5 deg). Seeding with leader-like values fixed the start (0.2-0.5 deg).
- **Model part:** history preprocessor / `condition_on_past_actions` (commands, not states).
- **Impact:** a deployment that initializes history from the robot state is out of distribution from step 1.
- **Workaround:** seed with state + mean leader-follower offset and the skill's typical gripper command.
- **Suggestion:** document what "command" means at rollout start and provide a helper to initialize it;
  augment training with command-history noise.

### 4. Integrated joint deltas drift when the arm can't follow
- **Saw:** the spoon pushed down into the table and twisted on it.
- **Evidence:** per-tick log: the integrated shoulder command ran to -130 deg while the arm sat at its stop
  (-100 deg); LeRobot's safety clamp fired 617 times in 45 s. In the demos the leader is never more than
  ~5-8 deg ahead of the follower (99.9th percentile), 18 deg for the wrist roll.
- **Model part:** action representation (joint deltas integrated from the last command, not the measured state).
- **Impact:** in contact or at a joint limit, the command history leaves the training distribution and
  the error compounds.
- **Workaround:** keep the remembered command within the demos' per-joint leader-follower range of the
  measured joints.
- **Suggestion:** anchor deltas on the measured state at replanning, or train with this clamp.

### 5. EMA weights wash out fast motions
- **Evidence:** open-loop wrist roll on scoop demos: EMA predicts 7-15 deg where the demo rolls 25-48 deg;
  the raw weights predict 21-31 deg. Mean displacement error 1.8 deg (EMA) vs 1.3 deg (raw).
- **Model part:** training recipe (EMA 0.999 used as the default eval/export weights).
- **Suggestion:** for short fine-tunes on small data, default to the raw weights or a lower decay.

### 6. Gross motion is learned quickly, contact-rich precision is not
- **Evidence:** action MSE falls from 0.33 to 0.08 within ~1,000 steps and stays flat to step 10,000.
  Live, the base rotation to the bowl matches the demos (-5.7 -> -31.3 deg vs -5.4 -> -32.4), but the
  wrist roll that scoops stops at 21 deg (demos up to 66) once the spoon is in the food.
- **Model part:** action head / data scale for contact phases.
- **Impact:** the skill that matters (scooping) is the one that fails.

### 7. Sensitive to visual changes that don't matter for the task
- **Evidence:** the scene camera's color cast changed midway through recording (scoop demos yellowish,
  later demos pink). Live, a pink scene during "scoop" made the arm hover over the bowl. We added
  per-skill color matching. The released `lora.json` has `augment: false`.
- **Model part:** visual conditioning; recipe without photometric augmentation.
- **Suggestion:** enable color jitter by default in the LoRA recipe.

### 8. Strong dependence on the exact start pose
- **Evidence:** starting 8-14 deg away from the demo start pose (pan, wrist), the policy executed the
  demonstrated motion and missed the bowl. Moving to the mean demo start pose first fixed the approach.
- **Model part:** with ~8 % of attention on the current camera image (below), the policy behaves largely
  like a proprioceptive trajectory replay.

## What the model looks at (attention)

`scripts/attention_maps.py` hooks the 28 joint transformer blocks and measures where the predicted-action
tokens put their attention (averaged over heads, action tokens, blocks and denoising steps).

| Moment (fine-tuned, step 10000) | Text | Imagined future frames | Current camera (scene / wrist) | Planned actions | State + past commands |
|---|---|---|---|---|---|
| scoop start | 43% | 29% | 8% (5% / 3%) | 12% | 7% |
| scoop turn to bowl | 42% | 34% | 9% (6% / 4%) | 11% | 4% |
| scoop in food | 41% | 34% | 9% (6% / 4%) | 11% | 4% |
| to mouth | 41% | 32% | 9% (6% / 3%) | 12% | 7% |
| tip | 43% | 31% | 8% (5% / 3%) | 12% | 6% |
| back to rest | 40% | 32% | 9% (5% / 4%) | 12% | 7% |

- The planned actions put only **8-9 %** of their attention on the current camera frames, about **a third on
  the frames FLUX imagines** for the next second ("imagine, then act"), and ~41 % on the instruction tokens.
- Within the camera frames, attention peaks on the **spoon tip** when it is in the bowl, at the mouth cup
  and while tipping; the bowl and cup themselves get little. Bright spots on the top border of the scene
  image appear in every frame regardless of content and are most likely attention sinks.
- Heat maps: `results/attention/finetuned_*.png` (scene left, wrist right; brighter = more attention).

## Displacement error

`scripts/eval_displacement.py`: FLUX gets the real 8-tick history at 5 moments in 6 demos per skill and
predicts 20 steps; ADE = mean absolute joint error over the 20 steps, FDE = error at step 20 (degrees).
These are training episodes (fit, not generalization).

DISPLACEMENT_TABLE

## Tooling notes

- LeRobot's `remote_inference` lists FLUX.3 as unsupported; synchronous rollout with a remote GPU needed
  our own bridge (`feeder/flux_server.py`, `feeder/flux_remote.py`): one request per chunk with the full
  8-tick history.
- The recipe's `eval_steps: 500` with `eval_split: 0.2` stalled training for 19 minutes per evaluation on
  our GPU (would have turned 2.8 h into ~9 h); we disabled it.
- Model load ~130 s; merging the LoRA into the base weights did not change inference speed.
- The model has no safety limits; our bridge enforces a 4 deg/tick rate limit, a 30 deg emergency clamp
  and holds the (taped) gripper.
