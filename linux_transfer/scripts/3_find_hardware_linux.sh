#!/usr/bin/env bash
# Read-only: lists the arms (by stable serial number) and cameras. Moves nothing.
# Follower board serial = 5B14114369, leader board serial = 5B8E116976.
set -uo pipefail
echo "== Arms (stable names include the serial number) =="
ls -l /dev/serial/by-id/ 2>/dev/null || echo "none found (arms plugged in with DATA cables + powered? are you in the dialout group?)"
echo
echo "   follower should be the one containing 5B14114369"
echo "   leader   should be the one containing 5B8E116976"
echo
echo "== Cameras (v4l2) =="
v4l2-ctl --list-devices 2>/dev/null || echo "v4l2-ctl missing: sudo apt install v4l-utils"
echo
echo "Linux makes 2 /dev/videoN nodes per camera; use the FIRST (even-numbered) one of each. Then LOOK at images with:"
echo "   source ~/dev/lerobot-env/bin/activate && lerobot-find-cameras opencv"
echo "Wrist cam = USB Camera (blurry close-up). Scene cam = Global Shutter Camera (top-down). Ignore the laptop's built-in webcam."
