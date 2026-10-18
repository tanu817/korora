"""FLUX bridge, robot side: drives the SO-101 with FLUX running on the RunPod pod.

Keeps FLUX's 8-tick history (state, camera frames, previous command) on this
laptop, mirroring LeRobot's synchronous engine, and sends it in one request per
action chunk. The 32 returned joint commands are executed at 30 Hz.

    # 1) tunnel to the pod, in its own terminal:
    #      ssh -N -L 8000:localhost:8000 root@<pod ip> -p <pod port> -i ~/.ssh/id_ed25519
    # 2) safe test, the arm does NOT move:
    #      python -m feeder.flux_remote --task "Scoop food from the bowl" --no-motion
    # 3) real run:
    #      python -m feeder.flux_remote --task "Scoop food from the bowl"

!!! Without --no-motion THIS MOVES THE REAL ARM. Keep a hand on the power switch. !!!
"""
from __future__ import annotations

import argparse
import base64
import json
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from . import config

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FPS = 30
DEFAULT_DURATION = 15  # seconds, for instructions not listed in config.DURATIONS


@dataclass
class RemoteSettings:
    server: str = "http://localhost:8000"
    port: str = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14114369-if00"  # follower board
    robot_id: str = "follower"           # must match the calibration file used for recording
    scene_cam: str | int = 0
    wrist_cam: str | int = 2
    max_step_deg: float = 4.0            # max change between consecutive commands, per joint per tick
    # LeRobot's max_relative_target measures from the *present* position. Under gravity the shoulder sags
    # a few degrees below its goal, so a tight value there stalls the arm; keep it as a wide emergency brake.
    emergency_limit_deg: float = 30.0
    hold_gripper: bool = True            # spoon is taped: keep the gripper where it is, don't squeeze
    no_motion: bool = False              # read cameras/joints and query FLUX, but never send commands
    confirm_each: bool = True            # ask before every subtask
    jpeg_quality: int = 90


def encode_jpeg(rgb: np.ndarray, quality: int) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode()


class RemoteFluxHands:
    """Hands implementation (see interfaces.py) backed by feeder/flux_server.py."""

    def __init__(self, s: RemoteSettings):
        self.s = s
        self.info = self._get_info()
        self.n_obs = self.info["n_obs_steps"]
        self.robot = self._connect_robot()
        self.history: deque = deque(maxlen=self.n_obs)  # (state, scene, wrist, previous command)
        self.last_command: np.ndarray | None = None
        self.last_sent: np.ndarray | None = None  # rate limit is applied between consecutive sent goals
        self.gripper_hold: float | None = None
        self._stop = threading.Event()

    def _get_info(self) -> dict:
        with urllib.request.urlopen(self.s.server, timeout=10) as r:
            info = json.load(r)
        print(f"FLUX server: {info}")
        return info

    def _connect_robot(self):
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        cams = {
            name: OpenCVCameraConfig(index_or_path=idx, width=640, height=480, fps=FPS, warmup_s=3)
            for name, idx in (("scene", self.s.scene_cam), ("wrist", self.s.wrist_cam))
        }
        robot = SO101Follower(
            SO101FollowerConfig(
                port=self.s.port,
                id=self.s.robot_id,
                cameras=cams,
                use_degrees=True,
                max_relative_target=self.s.emergency_limit_deg,
            )
        )
        robot.connect(calibrate=False)
        return robot

    def _observe(self) -> None:
        """One tick of history, like the synchronous preprocessor: the first tick fills all slots."""
        obs = self.robot.get_observation()
        state = np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
        if self.last_command is None:
            self.last_command = state.copy()
            self.last_sent = state.copy()
            self.gripper_hold = float(state[-1])
        entry = (state, obs["scene"], obs["wrist"], self.last_command.copy())
        for _ in range(self.n_obs if not self.history else 1):
            self.history.append(entry)

    def _request_chunk(self, task: str) -> np.ndarray:
        first, last = self.history[0], self.history[-1]
        q = self.s.jpeg_quality
        req = {
            "task": task,
            "state": [h[0].tolist() for h in self.history],
            "command": [h[3].tolist() for h in self.history],
            "scene": [encode_jpeg(first[1], q), encode_jpeg(last[1], q)],
            "wrist": [encode_jpeg(first[2], q), encode_jpeg(last[2], q)],
        }
        t0 = time.perf_counter()
        http = urllib.request.Request(
            self.s.server + "/act", data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(http, timeout=30) as r:
                reply = json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"FLUX server error: {json.load(e).get('error')}") from e
        print(f"   chunk: round trip {time.perf_counter() - t0:.2f} s (GPU {reply['infer_s']:.2f} s)")
        return np.array(reply["actions"], dtype=np.float32)

    def run(self, task: str, duration: float) -> None:
        self._stop.clear()
        queue: deque = deque()  # a new instruction never inherits commands planned for the old one
        end = time.perf_counter() + duration
        while time.perf_counter() < end and not self._stop.is_set():
            tick = time.perf_counter()
            self._observe()
            if not queue:
                queue.extend(self._request_chunk(task))
            command = queue.popleft()
            self.last_command = command
            if not self.s.no_motion:
                step = self.s.max_step_deg
                sent = np.clip(command, self.last_sent - step, self.last_sent + step)
                if self.s.hold_gripper:
                    sent[-1] = self.gripper_hold
                self.robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, sent, strict=True)})
                self.last_sent = sent
            time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - tick)))

    def _duration(self, instruction: str) -> float:
        for key, text in config.INSTRUCTIONS.items():
            if text == instruction:
                return config.DURATIONS.get(key, DEFAULT_DURATION)
        return DEFAULT_DURATION

    def do(self, instruction: str) -> None:
        if self.s.confirm_each and not self.s.no_motion:
            input(f"   ARM WILL MOVE: {instruction!r}. Hand on the power switch? [Enter / Ctrl+C] ")
        self.run(instruction, self._duration(instruction))

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.robot.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="exact instruction text, e.g. 'Scoop food from the bowl'")
    parser.add_argument("--duration", type=float, help="seconds (default: config.DURATIONS)")
    parser.add_argument("--no-motion", action="store_true", help="never send commands to the arm")
    parser.add_argument("--server", default=RemoteSettings.server)
    parser.add_argument("--yes", action="store_true", help="skip the keyboard confirmation (confirmed elsewhere)")
    parser.add_argument("--start-delay", type=float, default=3.0,
                        help="seconds the arm holds still after the motors switch on, so hands can let go")
    args = parser.parse_args()

    if args.task not in config.INSTRUCTIONS.values():
        print(f"WARNING: {args.task!r} is not one of the trained instructions in feeder/config.py")
    hands = RemoteFluxHands(RemoteSettings(server=args.server, no_motion=args.no_motion))
    try:
        if not args.no_motion and not args.yes:
            input(f"ARM WILL MOVE: {args.task!r}. Hand on the power switch? [Enter / Ctrl+C] ")
        print(f"Motors on, holding still for {args.start_delay:.0f} s. Let go of the arm now.", flush=True)
        time.sleep(args.start_delay)
        hands.run(args.task, args.duration or hands._duration(args.task))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        hands.close()


if __name__ == "__main__":
    main()
