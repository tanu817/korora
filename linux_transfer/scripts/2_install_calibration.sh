#!/usr/bin/env bash
# Copies the arm calibration made on the Windows laptop (ids: follower, leader) into LeRobot's calibration folder.
# Backs up anything already there. Run from the mac_transfer folder:  bash scripts/2_install_calibration.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HOME/.cache/huggingface/lerobot/calibration"
mkdir -p "$DEST/robots/so_follower" "$DEST/teleoperators/so_leader"

for f in robots/so_follower/follower.json teleoperators/so_leader/leader.json; do
  if [ -f "$DEST/$f" ]; then
    cp "$DEST/$f" "$DEST/$f.bak.$(date +%s)"
    echo "backed up existing $f"
  fi
  cp "$HERE/calibration/$f" "$DEST/$f"
  echo "installed $f"
done
echo "Done. NOTE: not verified that LeLab reads this same folder. If LeLab asks to calibrate, tell Claude first."
echo "To recalibrate instead: at the first prompt type 'c' + Enter (plain Enter reuses the old file)."
