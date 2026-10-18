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
import csv
import json
import math
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
    # FLUX predicts command deltas that the bridge integrates. When the arm can't follow (e.g. shoulder
    # already at its stop) the integrated command drifts far outside the demos. Keep the remembered
    # command within config.COMMAND_LEAD_LIMIT of the measured joints.
    limit_command_lead: bool = True
    trajectory_dir: str = "results/runs"  # per-tick CSV of every run, for diagnosis (gitignored)
    # LeRobot's max_relative_target measures from the *present* position. Under gravity the shoulder sags
    # a few degrees below its goal, so a tight value there stalls the arm; keep it as a wide emergency brake.
    emergency_limit_deg: float = 30.0
    hold_gripper: bool = True            # spoon is taped: keep the gripper where it is, don't squeeze
    no_motion: bool = False              # read cameras/joints and query FLUX, but never send commands
    confirm_each: bool = True            # ask before every subtask
    jpeg_quality: int = 90
    exec_steps: int = 10                 # execute this many of each predicted chunk, then replan
    # Per-skill FLUX server (instruction key -> URL), e.g. a checkpoint that is better at one skill.
    server_map: dict | None = None
    match_scene_colors: bool = True      # rescale scene colors to the running skill's demos (config.SCENE_MEAN_RGB)


def parse_server_map(text: str | None) -> dict:
    """'to_mouth=http://localhost:8003,back_to_rest=http://localhost:8003' -> {key: url}."""
    pairs = [item.split("=", 1) for item in (text or "").split(",") if item]
    bad = [k for k, _ in pairs if k not in config.INSTRUCTIONS]
    if bad:
        raise SystemExit(f"--server-map keys must be from {list(config.INSTRUCTIONS)}, got {bad}")
    return dict(pairs)


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
        self.start_gripper_command: float | None = None  # leader gripper value at the start of the skill's demos
        self.scene_target: np.ndarray | None = None  # demo mean RGB of the running skill
        self.scene_gain: np.ndarray | None = None    # per-channel gain, fixed at the start of each step
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
        # Feetech buses occasionally return a corrupted packet while connecting. Retry, and never
        # leave the motors stiff behind a failed connect.
        # The desktop's PipeWire also grabs cameras for a few seconds now and then: wait and retry.
        for attempt in (1, 2, 3):
            try:
                robot.connect(calibrate=False)
                return robot
            except (ConnectionError, TimeoutError) as e:  # camera timeouts raise TimeoutError
                print(f"   connecting to the arm failed (attempt {attempt}): {e}")
                time.sleep(5)
                for cleanup in (
                    robot.bus.disable_torque,
                    lambda: robot.bus.disconnect(disable_torque=False),
                    *(cam.disconnect for cam in robot.cameras.values()),
                ):
                    try:
                        cleanup()
                    except Exception:
                        pass
        raise ConnectionError("Could not connect to the arm; motors switched off")

    def _observe(self) -> None:
        """One tick of history, like the synchronous preprocessor: the first tick fills all slots."""
        obs = self.robot.get_observation()
        state = np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
        if self.last_command is None:
            # seed the command history like the start of a recorded demo (commands = leader positions)
            self.last_command = state.copy()
            self.last_command[:5] += np.array(config.COMMAND_OFFSET, dtype=np.float32)
            if self.start_gripper_command is not None:
                self.last_command[5] = self.start_gripper_command
            self.last_sent = state.copy()
            if self.gripper_hold is None:
                self.gripper_hold = float(state[-1])
        scene = obs["scene"]
        if self.scene_target is not None:
            if self.scene_gain is None:
                self.scene_gain = np.clip(self.scene_target / scene.reshape(-1, 3).mean(0), 0.5, 2.0)
                print(f"   scene color gain (R, G, B): {np.round(self.scene_gain, 2)}")
            scene = np.clip(scene * self.scene_gain, 0, 255).astype(np.uint8)
        entry = (state, scene, obs["wrist"], self.last_command.copy())
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
        key = next((k for k, text in config.INSTRUCTIONS.items() if text == task), None)
        server = (self.s.server_map or {}).get(key, self.s.server)
        http = urllib.request.Request(
            server + "/act", data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(http, timeout=30) as r:
                reply = json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"FLUX server error: {json.load(e).get('error')}") from e
        print(f"   chunk: round trip {time.perf_counter() - t0:.2f} s (GPU {reply['infer_s']:.2f} s)")
        return np.array(reply["actions"], dtype=np.float32)

    def move_to_start(self, speed_deg: float = 1.0) -> None:
        """Slowly move to config.START_POSE (not FLUX: the reset a person does before each demo),
        then forget the history so FLUX starts fresh, as at the start of a recorded demo."""
        state = self.robot.get_observation()
        current = np.array([state[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
        target = current.copy()
        target[:5] = config.START_POSE
        steps = max(1, int(np.ceil(np.abs(target - current).max() / speed_deg)))
        print(f"   moving to start pose over {steps / FPS:.1f} s: {np.round(target[:5], 1)}")
        for k in range(1, steps + 1):
            goal = current + (target - current) * k / steps
            self.robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, goal, strict=True)})
            time.sleep(1 / FPS)
        time.sleep(1.0)  # settle
        self.history.clear()
        self.last_command = None

    def run(self, task: str, duration: float) -> None:
        """Execute `duration` seconds of motion (30 Hz ticks). Waiting for FLUX between chunks
        doesn't count, so step lengths match the demo lengths whatever the GPU speed."""
        self._stop.clear()
        key = next((k for k, text in config.INSTRUCTIONS.items() if text == task), None)
        target = config.SCENE_MEAN_RGB.get(key) if self.s.match_scene_colors else None
        self.scene_target = None if target is None else np.array(target, dtype=np.float32)
        self.scene_gain = None
        # every recorded demo started with a fresh history, so every step does too
        self.start_gripper_command = config.START_GRIPPER_COMMAND.get(key)
        self.history.clear()
        self.last_command = None
        queue: deque = deque()  # a new instruction never inherits commands planned for the old one
        log_dir = Path(self.s.trajectory_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{key or 'custom'}.csv"
        log_file = log_path.open("w", newline="")
        log = csv.writer(log_file)
        log.writerow(["t", "new_chunk"] + [f"state_{j}" for j in JOINTS] + [f"cmd_{j}" for j in JOINTS]
                     + [f"sent_{j}" for j in JOINTS])
        t_start = time.perf_counter()
        goal = config.GOAL_POSE.get(key)
        goal = None if goal is None else np.array(goal, dtype=np.float32)
        closest = np.inf
        ticks_left = round(duration * FPS)
        # safety cap if the GPU is very slow: the motion itself plus up to 10 s per FLUX request
        deadline = time.perf_counter() + duration + 10 * math.ceil(ticks_left / self.s.exec_steps)
        while ticks_left > 0 and time.perf_counter() < deadline and not self._stop.is_set():
            ticks_left -= 1
            tick = time.perf_counter()
            self._observe()
            new_chunk = not queue
            if new_chunk:
                queue.extend(self._request_chunk(task)[: self.s.exec_steps])
            command = queue.popleft()
            state = self.history[-1][0]
            if self.s.limit_command_lead:
                lead = np.array(config.COMMAND_LEAD_LIMIT, dtype=np.float32)
                command[:5] = np.clip(command[:5], state[:5] - lead, state[:5] + lead)
            self.last_command = command
            if not self.s.no_motion:
                step = self.s.max_step_deg
                sent = np.clip(command, self.last_sent - step, self.last_sent + step)
                if self.s.hold_gripper:
                    sent[-1] = self.gripper_hold
                self.robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, sent, strict=True)})
                self.last_sent = sent
            log.writerow([round(tick - t_start, 3), int(new_chunk)] + [round(float(v), 2) for v in state]
                         + [round(float(v), 2) for v in command] + [round(float(v), 2) for v in self.last_sent])
            if goal is not None:
                dist = float(np.abs(state[:4] - goal).max())
                closest = min(closest, dist)
                if dist <= config.GOAL_REACHED_DEG or (
                    closest <= config.GOAL_NEAR_DEG and dist >= closest + config.GOAL_PASSED_DEG
                ):
                    print(f"   goal pose reached (closest {closest:.1f} deg) after {(tick - t_start):.0f} s")
                    break
            time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - tick)))
        log_file.close()
        print(f"   trajectory saved: {log_path}")

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
    parser.add_argument("--exec-steps", type=int, default=RemoteSettings.exec_steps,
                        help="actions executed per FLUX request before replanning (model predicts 32)")
    parser.add_argument("--start-delay", type=float, default=3.0,
                        help="seconds the arm holds still after the motors switch on, so hands can let go")
    parser.add_argument("--no-color-match", action="store_true", help="send the scene camera colors unchanged")
    parser.add_argument("--no-reset", action="store_true",
                        help="don't move to config.START_POSE first (default: always reset)")
    args = parser.parse_args()

    if args.task not in config.INSTRUCTIONS.values():
        print(f"WARNING: {args.task!r} is not one of the trained instructions in feeder/config.py")
    hands = RemoteFluxHands(
        RemoteSettings(
            server=args.server,
            no_motion=args.no_motion,
            exec_steps=args.exec_steps,
            match_scene_colors=not args.no_color_match,
        )
    )
    try:
        if not args.no_motion and not args.yes:
            input(f"ARM WILL MOVE: {args.task!r}. Hand on the power switch? [Enter / Ctrl+C] ")
        print(f"Motors on, holding still for {args.start_delay:.0f} s. Let go of the arm now.", flush=True)
        time.sleep(args.start_delay)
        if not args.no_reset and not args.no_motion:
            hands.move_to_start()
        hands.run(args.task, args.duration or hands._duration(args.task))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        hands.close()


if __name__ == "__main__":
    main()
