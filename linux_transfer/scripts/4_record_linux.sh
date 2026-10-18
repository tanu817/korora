#!/usr/bin/env bash
# Command-line recording. Set the camera indices (look at the images first!):
#   SCENE_IDX=2 WRIST_IDX=0 TASK="Scoop food from the bowl" bash scripts/4_record_linux.sh
# Arm ports are found automatically from their serial numbers. Defaults: 3 episodes, 20 s each, 15 s reset, local only.
set -euo pipefail
# shellcheck disable=SC1091
source "$HOME/dev/lerobot-env/bin/activate"
FPORT="$(ls /dev/serial/by-id/*5B14114369* | head -1)"
LPORT="$(ls /dev/serial/by-id/*5B8E116976* | head -1)"
: "${SCENE_IDX:?set SCENE_IDX (top-down camera index)}"
: "${WRIST_IDX:?set WRIST_IDX (wrist camera index)}"
DATASET="${DATASET:-ArnavG07/spoon_feed}"
TASK="${TASK:?set TASK (the exact step text)}"
EPISODES="${EPISODES:-3}"; EP_S="${EP_S:-20}"; RESET_S="${RESET_S:-15}"
PUSH="${PUSH:-false}"; RESUME="${RESUME:-false}"
echo "follower=$FPORT leader=$LPORT scene=$SCENE_IDX wrist=$WRIST_IDX task=\"$TASK\" resume=$RESUME"
lerobot-record \
  --robot.type=so101_follower --robot.port="$FPORT" --robot.id=follower \
  --robot.cameras="{ scene: {type: opencv, index_or_path: $SCENE_IDX, width: 640, height: 480, fps: 30, warmup_s: 15}, wrist: {type: opencv, index_or_path: $WRIST_IDX, width: 640, height: 480, fps: 30, warmup_s: 15}}" \
  --teleop.type=so101_leader --teleop.port="$LPORT" --teleop.id=leader \
  --dataset.repo_id="$DATASET" --dataset.single_task="$TASK" \
  --dataset.num_episodes="$EPISODES" --dataset.episode_time_s="$EP_S" --dataset.reset_time_s="$RESET_S" \
  --dataset.push_to_hub="$PUSH" --resume="$RESUME" --display_data=false
# Keys: Right arrow = end episode early, Left = redo, Esc = stop. Keep the leader's gripper trigger still.
