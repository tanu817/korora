# CONTEXT (Arnav) — Spoon-feeding robot, ERH Physical AI Hackathon, Zurich — continuing on the LEGION (Linux)
Updated Sat 17 Oct 2026 ~23:00. Submission Sun 18 Oct 14:00 (GitHub repo + slides + demo video). Judging 14:30.

## Instructions for the assistant
- User = Arnav (HF user ArnavG07), robotics beginner. One step at a time, short. Do work yourself; hand over only what needs hands
  (plugging cables, moving arms, interactive prompts, keys). Run interactive commands in a separate terminal window.
- Don't invent FLUX 3 Action / flux-action APIs. Never paste tokens in chat. Say what a command changes before running it.
- Safety: hand near the follower's power plug. Hold the follower when torque turns off. NEVER use camera "M200J3".
- Do NOT recalibrate. Calibration is done (see below). If a tool asks to calibrate, STOP and ask.
- Do not start paid GPU/cloud training without the team's compute owner (shared EUR 133 budget, reimbursed with invoice).

## Project (whole team agreed: spoon-feeding; the old cup-shuffle plan is dropped)
Spoon fixed in the follower gripper (gripper never moves; keep the LEADER gripper trigger still). Scoops crisps from a bowl, brings them to a
cup with a black flap ("mouth"). FLUX 3 Action does arm motion; our harness chooses the instruction. FLUX must stay in the control loop.
FOUR steps = four recording sessions, each with its own exact task text, all in ONE dataset (`ArnavG07/spoon_feed`):
1. `Scoop food from the bowl`   2. `Bring the spoon to the mouth`   3. `Tip the food into the mouth`   4. `Move the spoon back to the rest position`
(step 4 may later be changed to end at rest so the chain loops). Start poses: step1 from rest; steps 2-4 start where the previous step ended.
Settings: 20 s episodes, 15 s reset, first record 3 test episodes per step and CHECK them, then ~30 per step. Steps 2-4 use --resume=true with the
full dataset name. Nothing real has been recorded yet.

## Hardware / calibration (stable across laptops)
- Follower board serial 5B14114369, id `follower`. Leader board serial 5B8E116976, id `leader`. Linux names: /dev/serial/by-id/*5B14114369* and *5B8E116976*.
- Calibration files (already in this bundle, same ids): ~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower.json and
  .../teleoperators/so_leader/leader.json. Install with scripts/2_install_calibration.sh. LeLab and lerobot-* both read that folder.
  In LeLab pick the existing `follower`/`leader` config files; never press Calibrate. Gripper range is tiny by design (gripper is fixed).
- Cameras: 640x480@30. wrist = "USB Camera" (blurry close-up), scene = "Global Shutter Camera" (top-down: cup, spoon, bowl).
  Camera names MUST be `scene` and `wrist` everywhere (FLUX's expected names unverified: check the flux-action docs).
  Indices differ per machine: look at the images from lerobot-find-cameras. Only one program can hold a camera at a time.
- Recording command: scripts/4_record_linux.sh (warmup_s: 15 per camera, ports auto-detected).

## Lessons from the Mac (why we switched)
- On the Mac, everything shared one flaky USB hub: devices dropped in and out, the scene camera started slowly (up to ~9 s), and after the
  first open it returned zero frames on reopen; LeLab then failed with "failed to set fps=30 (actual_fps=10.0)". Photo Booth/browsers holding a camera also
  broke it. If a camera acts up: close every other app/tab using cameras, unplug/replug that camera, avoid probing it repeatedly.
- LeLab's own record code did not set camera warmup (default 1 s, too short for the scene camera). On the Mac we patched
  site-packages/lelab/record.py (warmup_s + connect retries). If the same timeout appears, use the CLI script (it sets warmup_s: 15).
- The torchcodec / objc warnings on record are harmless. LeLab record screen: "Resume" is only for adding to an existing dataset (steps 2-4).
- Intermittent "motor 4 (wrist_flex) timed out" happened once on Windows: reseat its cable, power-cycle the follower brick.
- Practice recording on the Mac worked via the CLI (2 episodes, 30 Hz, no dropped ticks). LeLab appends a timestamp to a NEW dataset name.

## Next steps on the Legion
1. `bash scripts/1_setup_linux.sh`, then log out/in (serial permissions). 2. `source ~/dev/lerobot-env/bin/activate && hf auth login` (token typed in terminal only).
3. `bash scripts/2_install_calibration.sh`. 4. Plug in arms (USB data + each own power brick) and cameras; `bash scripts/3_find_hardware_linux.sh`; LOOK at camera images.
5. Teleop test, then step 1 (3 episodes), check data, continue. 6. Get FLUX docs (camera names, GPU needs) from the team's compute owner / track lead (Nikhil).
7. This Legion has an RTX 3070 Mobile (8 GB VRAM): it CANNOT run FLUX. Zero-shot, LoRA training and rollout all run on AWS. Legion = teleop + recording only.
