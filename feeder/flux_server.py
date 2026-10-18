"""FLUX bridge, GPU side: keeps FLUX in memory and answers one request per action chunk.

Runs on the RunPod pod. The laptop (feeder/flux_remote.py) keeps FLUX's 8-tick
history itself and sends it in one request, so the distance to the pod costs one
round trip per ~1 s chunk instead of one per 30 Hz tick.

    python -m feeder.flux_server --policy-path /workspace/models/flux-3-action-so101-normfix
    python -m feeder.flux_server --policy-path <lora checkpoint>/pretrained_model --use-peft

Listens on localhost only; the laptop reaches it through `ssh -N -L 8000:localhost:8000 ...`.
"""
from __future__ import annotations

import argparse
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.rollout.context import _load_pretrained_policy

CAMERAS = ("scene", "wrist")


def decode_jpeg(data: str) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class Flux:
    def __init__(self, policy_path: str, use_peft: bool):
        cfg = PreTrainedConfig.from_pretrained(policy_path)
        cfg.pretrained_path = policy_path
        cfg.use_peft = use_peft
        cfg.device = "cuda"
        self.cfg = cfg
        policy = _load_pretrained_policy(cfg)
        if use_peft:
            # Fold the LoRA into the base weights once: same outputs, no adapter overhead per request.
            policy = policy.merge_and_unload()
        self.policy = policy.to("cuda").eval()
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=policy_path)
        self.lock = threading.Lock()

    def frames(self, first: str, last: str) -> torch.Tensor:
        """(1, n_obs, 3, H, W) in [0, 1]. FLUX encodes only the first and last history frame
        (history_snapshots=2), so the frames in between are copies and never looked at."""
        a, b = (torch.from_numpy(decode_jpeg(x)).permute(2, 0, 1).float() / 255 for x in (first, last))
        n = self.cfg.n_obs_steps
        return torch.stack([a] * (n - 1) + [b])[None]

    def act(self, req: dict) -> list[list[float]]:
        """Same steps as LeRobot's synchronous engine, but with the history passed in explicitly."""
        obs = {
            "observation.state": torch.tensor(req["state"], dtype=torch.float32)[None],
            "observation.command_history": torch.tensor(req["command"], dtype=torch.float32)[None],
            "task": req["task"],
            "robot_type": "so_follower",
        }
        for cam in CAMERAS:
            obs[f"observation.images.{cam}"] = self.frames(*req[cam])
        amp = torch.autocast("cuda") if self.cfg.use_amp else torch.autocast("cuda", enabled=False)
        with self.lock, torch.inference_mode(), amp:
            self.policy.reset()
            self.pre.reset()
            self.post.reset()
            batch = self.pre(obs)
            chunk = self.policy.predict_action_chunk(batch)[:, : self.cfg.n_action_steps]
            commands = self.post(chunk)
        return commands[0].float().cpu().tolist()


def make_handler(flux: Flux, info: dict):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._reply(200, info)

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                t0 = time.perf_counter()
                actions = flux.act(req)
                self._reply(200, {"actions": actions, "infer_s": time.perf_counter() - t0})
            except Exception as e:  # report to the laptop instead of dropping the connection
                self._reply(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--use-peft", action="store_true", help="policy-path is a LoRA checkpoint")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    t0 = time.perf_counter()
    flux = Flux(args.policy_path, args.use_peft)
    info = {
        "policy_path": args.policy_path,
        "use_peft": args.use_peft,
        "n_obs_steps": flux.cfg.n_obs_steps,
        "n_action_steps": flux.cfg.n_action_steps,
        "fps": flux.cfg.fps,
    }
    print(f"FLUX loaded in {time.perf_counter() - t0:.0f} s: {info}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(flux, info)).serve_forever()


if __name__ == "__main__":
    main()
