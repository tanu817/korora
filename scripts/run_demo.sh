#!/usr/bin/env bash
# The demo exactly as it ran best (Sun 18 Oct, 14:10 and 14:26): one bite, all four steps, autonomous.
#   scoop + tip         -> FLUX LoRA step 12,000 (pod port 8000, tunnel 8001)
#   to mouth + back     -> FLUX LoRA step 10,000 (pod port 8002, tunnel 8003)
#   20 commands per FLUX plan, slow reset to the demo start pose first.
# Needs: pod/serve_demo.sh running on the pod, follower + both cameras plugged in.
# !!! THE ARM MOVES. Keep a hand on the follower's power switch and catch the arm at the end. !!!
#   POD="root@213.173.111.33 -p 32719" bash scripts/run_demo.sh
set -euo pipefail
POD=${POD:?set POD, e.g. POD="root@<pod ip> -p <pod ssh port>"}
cd "$(dirname "$0")/.."
for pair in 8001:8000 8003:8002; do
  local_port=${pair%%:*}
  curl -s --max-time 3 "localhost:$local_port" >/dev/null || \
    ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -L "$local_port:localhost:${pair##*:}" $POD
done
for p in 8001 8003; do echo -n "server on $p: "; curl -s --max-time 5 "localhost:$p" | grep -o 'checkpoints/[0-9]*'; done
python -m feeder.chain --bites "${BITES:-1}" --exec-steps 20 \
  --server http://localhost:8001 \
  --server-map to_mouth=http://localhost:8003,back_to_rest=http://localhost:8003 \
  --checkpoint "step-12000 scoop+tip / step-10000 to_mouth+back (demo)" --yes
