# linux_transfer — everything the Legion (Linux) needs
Copy this whole folder (or linux_transfer.zip) to the Linux side (USB stick, cloud drive, or `scp`). Then open a terminal in this folder and run Claude Code
there: it reads CLAUDE.md automatically.
1. `bash scripts/1_setup_linux.sh`   (apt + uv + LeRobot + LeLab; asks for sudo; then LOG OUT AND IN)
2. `source ~/dev/lerobot-env/bin/activate && hf auth login`
3. `bash scripts/2_install_calibration.sh`   (puts the calibration where LeRobot AND LeLab look; backs up anything existing)
4. `bash scripts/3_find_hardware_linux.sh`
5. LeLab: `lelab` (opens http://localhost:8000); robot with leader+follower ports, existing `leader`/`follower` config files, cameras named `scene` and `wrist`.
6. Or CLI: `SCENE_IDX=.. WRIST_IDX=.. TASK="Scoop food from the bowl" bash scripts/4_record_linux.sh`
Scripts are untested on this laptop. If one fails, paste the error to Claude.
