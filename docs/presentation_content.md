# Korora — presentation content (for building the slide deck)

Brief for whoever builds the deck: one slide per `##` section below, in this order. Numbers are measured
(sources given); don't round them further or invent new ones. Figures to use are listed per slide with
their file paths in the repo `tanu817/korora`. Audience: Black Forest Labs judges (FLUX 3 Action track,
"Imagine, Then Act") — they care most about how FLUX was used and what we learned about the model.
Judging weights: demo on real hardware 25 %, technical difficulty 20 %, use of FLUX 20 %, idea 15 %,
real-world use 10 %, video 10 %.

---

## 1. Title
- **Korora — a robot arm that feeds people who can't use their arms**
- SO-101 arm + FLUX 3 Action, fine-tuned on our own demos, driven live by FLUX
- Team Korora · ERH Physical AI Hackathon, Zurich · Black Forest Labs track
- Visual: photo/still of the arm with the spoon over the bowl (from the demo video)

## 2. The problem
- People with paralysis or limited arm function depend on a caregiver for every single bite.
- Feeding is repetitive, long and personal: a good fit for a robot that does the same steps reliably
  while the person keeps control of the pace.
- Our task: scoop food → bring it to the mouth → tip it in → return → repeat, bite after bite.
- At the hackathon the "mouth" is a paper cup; never a real face.

## 3. How it works: harness + hands
- **Hands = FLUX 3 Action** (7B world action model): sees the scene camera + wrist camera, the joint
  angles and a text instruction; predicts the next second of arm motion *together with* the next video
  frames ("imagine, then act").
- **Harness = our Python code** (`feeder/chain.py`): sends one instruction per step, runs the bite chain,
  logs every step, optional human check + retry. Also voice control ("feed me" / "stop").
- **Bridge**: the GPU is in the cloud (RunPod, Romania), the arm is on a laptop in Zurich. LeRobot's
  remote inference doesn't support FLUX yet, so we built our own: one request per action chunk with
  FLUX's full 8-tick history; ~0.75 s on the GPU, ~0.9 s round trip.
- The 4 instructions FLUX learned: "Scoop food from the bowl", "Bring the spoon to the mouth",
  "Tip the food into the mouth", "Move the spoon back to the rest position".
- Visual: diagram laptop (arm + 2 cameras + harness) ⇄ bridge ⇄ RunPod GPU (FLUX). Diagram of the
  4-step chain looping.

## 4. Our data
- **136 teleoperation demos** recorded with the leader arm: 49 scoop, 48 to mouth, 19 tip, 20 back to rest.
- 30 Hz, two cameras (scene top-down + wrist), ~18 minutes in total, dataset `ArnavG07/korora_spoon_feed`.
- Visual: the side-by-side scene | wrist frame (that is exactly what FLUX sees).

## 5. Training
- LoRA fine-tune of `flux-3-action-so101` with BFL's own recipe (rank 32, 10,000 steps, batch 2 × 4).
- 1× RTX PRO 6000 (96 GB), 1 step per second, **2 h 47 min**, 30 GB of GPU memory.
- **Action loss fell from 0.33 to 0.08 within the first ~1,000 steps, then stayed flat**; video loss kept
  falling slowly (0.13 → 0.10).
- Visual: the training-curve chart (dashboard https://claude.ai/artifact/Nqs3TULKcEtTUb8EiPLPJQ; data in
  the training log).

## 6. Zero-shot vs fine-tuned (graded comparison)
- **Zero-shot** (released checkpoint `c9e13b2` + normalization fix): **0/4 skills**, no human intervention.
  It headed for the bowl when asked to scoop but got stuck on the rim; on the other skills it pushed or
  knocked over the mouth cup. (Details: `results/log.csv`, phase `zero-shot`.)
- **Fine-tuned**: open-loop it reproduces the demos (scoop: mean displacement error 1.3° over the next
  20 steps vs 2.0° for "stand still"; see slide 9). Live it now drives the spoon **into the food bowl
  along the demonstrated path** (base rotation −5.7° → −31.3°, demos −5.4° → −32.4°). The scooping
  wrist roll still stalls in the food (21° vs up to 66° in the demos).
- Visual: two-column table zero-shot | fine-tuned per skill; a still from each.
- (Update this slide with the final live results before presenting.)

## 7. What it took: 8 problems between "trained" and "works on the arm"
Each one produced plausible-looking but wrong behaviour without any error message:
1. **Normalization in the wrong joint convention** in the released checkpoint (shoulder expected at +85…+184°,
   our arm rests at −100°) → patched quantiles (fix from the organizers).
2. **Fixed sampling seed → the arm "waits forever"**: our demos start with ~1.7 s of standing still; with a
   fixed seed FLUX returns the same "keep waiting" plan every time → fresh seed per request.
3. **History seeding**: FLUX remembers its past *commands* (= leader arm positions). Seeding them with the
   follower's values tripled the prediction error (1.3° → 3.8°) → seed like a demo start.
4. **Delta drift**: FLUX predicts joint changes; when the arm hits the table the summed command ran to −130°
   (arm at −100°) and pushed into the table → keep the command within the demos' leader–follower range.
5. **EMA weights wash out fast motions** (wrist roll 7–15° instead of 25–48°) → use the raw weights.
6. **Lighting change** mid-recording (pink colour cast) confused the scoop → per-skill colour matching.
7. **Start pose**: 8–14° off the demo start and FLUX misses the bowl → automatic reset to the demo start pose.
8. **Training-time eval** in the recipe stalled training 19 min per check (2.8 h → ~9 h) → disabled.
- Visual: a timeline of live runs (06:41 → 11:20) with the fix after each, from `results/log.csv`.

## 8. What FLUX looks at (attention maps)
- We hooked FLUX's 28 joint transformer blocks and measured where the **planned-action tokens** attend:

| | Text instruction | Imagined future frames | Current camera (scene / wrist) | Own planned actions | Joint state + past commands |
|---|---|---|---|---|---|
| Average over 6 moments | ~42 % | ~32 % | **~9 % (5–6 % / 3–4 %)** | ~12 % | ~6 % |

- FLUX plans mainly from **what it imagines next** and the instruction, and only ~9 % from the current
  camera picture → explains why it reacted so little to visual changes (light, start pose).
- Inside the camera image it **tracks the spoon tip**; the bowl and cup barely light up.
- Visual: `results/attention/finetuned_scoop_in_food.png`, `..._to_mouth.png`, `..._tip.png`
  (scene left, wrist right, brighter = more attention). Mention that the bright spots on the top border
  are most likely attention sinks.
- (Zero-shot comparison maps: `results/attention/zeroshot_*.png`, if present.)

## 9. Displacement error (how close the predicted trajectory is to the demo)
- Open-loop: FLUX gets the real 8-tick history at 5 moments in 6 demos per skill and predicts 20 steps;
  ADE = mean joint error over the 20 steps, FDE = error at step 20 (degrees). Training episodes.
| Model | Scoop ADE / FDE | To mouth ADE / FDE | Tip ADE / FDE | Back to rest ADE / FDE | Mean ADE |
|---|---|---|---|---|---|
| stand still | 2.00 / 3.28 | 1.57 / 3.18 | 2.64 / 4.81 | 3.23 / 5.85 | **2.36** |
| zero-shot | 3.55 / 7.10 | 2.48 / 6.00 | 3.38 / 7.23 | 3.16 / 6.84 | **3.14** |
| step-1000 | 2.41 / 4.44 | 1.72 / 3.14 | 2.55 / 5.21 | 2.32 / 4.61 | **2.25** |
| step-4000 | 1.93 / 3.35 | 1.79 / 3.28 | 2.07 / 4.17 | 2.44 / 4.84 | **2.06** |
| step-7000 | 1.89 / 3.26 | 1.36 / 2.37 | 1.69 / 3.22 | 2.46 / 4.97 | **1.85** |
| step-10000 | 1.50 / 2.62 | 1.56 / 2.81 | 1.68 / 3.35 | 2.86 / 5.48 | **1.90** |
| step-10000-EMA | 1.92 / 3.36 | 1.51 / 2.87 | 2.07 / 4.19 | 2.34 / 4.59 | **1.96** |

Wrist-roll ADE on scoop (degrees): stand still 6.34, zero-shot 5.06, step-1000 4.29, step-4000 4.43, step-7000 4.38, step-10000 3.42, step-10000-EMA 4.53.

- Messages:
  - **Zero-shot is worse than standing still** (mean ADE 3.14° vs 2.36°): the base model actively moves the wrong way on our task.
  - **Fine-tuning keeps improving the trajectory until ~7,000 steps** (2.25° → 2.06° → 1.85°), although the
    training loss looked flat after 1,000 steps: the loss curve undersells the training.
  - Still only ~20 % better than "stand still" overall; the **wrist roll** stays the largest error (scoop 3.4–4.5°).
  - "Back to rest" barely beats standing still and gets worse with more training (2.32° → 2.86°).
  - Use a grouped bar chart: mean ADE per model, with the "stand still" line as reference.

## 10. Our suggestions for BFL
- Ship SO-101 processors in raw LeRobot degrees and warn when live joints fall outside the normalization range.
- Fresh sampling noise per call by default (fixed seed + idle demo starts = deadlock).
- Document and help initialize the past-command history at rollout start (leader vs follower values).
- Anchor or clamp integrated deltas when the arm can't follow (contact, joint limits).
- Prefer raw over EMA weights for short fine-tunes on small data, or a lower decay.
- Enable colour augmentation in the LoRA recipe (`augment: false` today); cheaper default eval in `lora.json`.
- Support FLUX in LeRobot's remote inference (most teams don't have a GPU next to the arm).
- Full write-up with evidence: `docs/flux_findings.md`.

## 11. Time split (required by the track)
- Saturday: setup, calibration, scene, recording tools (team to fill in exact hours).
- Sun 01:10–03:30: recording 136 demos (4 skills).
- Sun 04:40–05:40: cloud GPU setup, model download, normalization fix, bridge.
- Sun 05:40–08:30: training (2 h 47 min), in parallel zero-shot tests (05:27–05:50) and checkpoint tests.
- Sun 08:45–12:00: live testing and debugging the 8 problems above; evaluation and attention analysis.
- Rest: demo, video, slides.

## 12. Demo
- Video: autonomous run of the chain with FLUX driving the arm (to be recorded).
- Be honest on stage: what works live (approach to the bowl along the demonstrated path, all 4 instructions
  driving the arm) and what doesn't yet (completing the scoop).

## 13. Lessons / what we'd do next
- Trim idle time at the start of demos; record more scoop demos with the spoon going deep into the food.
- Keep lighting and object placement fixed from the first demo on; tape everything.
- Retrain with colour augmentation and raw weights; then a full multi-bite chain with retries.
