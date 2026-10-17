"""FLUX 3 Action as the "hands".

v0 (this file): runs the documented `lerobot-rollout` command once per subtask,
with the subtask as --task. Simple and uses only the official CLI.
Downside: the 7B model is re-loaded for every subtask (slow). Once this works,
v1 should load Flux3Policy once in Python and switch the instruction per subtask.

!!! THIS MOVES THE REAL ARM. Keep a hand on the power switch. !!!
Must run on the machine with the NVIDIA GPU AND the arm + both cameras attached.
"""
from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass

from . import config

DEFAULT_DURATION = 15  # seconds, for instructions not listed in config.DURATIONS


@dataclass
class FluxSettings:
    policy_path: str                     # base checkpoint dir (zero-shot) or our LoRA checkpoint
    use_peft: bool = False               # True for our LoRA checkpoint
    port: str = "/dev/ttyACM0"
    robot_id: str = "so101"
    scene_cam: str | int = 0             # camera index or /dev/videoN
    wrist_cam: str | int = 1
    confirm_each: bool = True            # ask before every subtask (recommended at first)
    dry_run: bool = False                # only print the command


class FluxHands:
    def __init__(self, s: FluxSettings):
        self.s = s

    def _duration(self, instruction: str) -> int:
        for key, text in config.INSTRUCTIONS.items():
            if text == instruction:
                return config.DURATIONS.get(key, DEFAULT_DURATION)
        return DEFAULT_DURATION

    def command(self, instruction: str) -> list:
        s = self.s
        cams = {
            "scene": {"type": "opencv", "index_or_path": s.scene_cam, "width": 640, "height": 480, "fps": 30},
            "wrist": {"type": "opencv", "index_or_path": s.wrist_cam, "width": 640, "height": 480, "fps": 30},
        }
        cmd = [
            "lerobot-rollout",
            "--strategy.type=base", "--inference.type=sync",
            f"--policy.path={s.policy_path}", "--policy.device=cuda",
            "--robot.type=so101_follower", f"--robot.port={s.port}", f"--robot.id={s.robot_id}",
            "--robot.use_degrees=true",
            f"--robot.cameras={json.dumps(cams)}",
            "--fps=30", f"--task={instruction}", f"--duration={self._duration(instruction)}",
        ]
        if s.use_peft:
            cmd.insert(4, "--policy.use_peft=true")
        return cmd

    def do(self, instruction: str) -> None:
        cmd = self.command(instruction)
        print("   $ " + " ".join(shlex.quote(c) for c in cmd))
        if self.s.dry_run:
            return
        if self.s.confirm_each:
            input("   ARM WILL MOVE. Hand on the power switch? [Enter to go, Ctrl+C to stop] ")
        subprocess.run(cmd, check=True)
