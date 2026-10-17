"""Human stand-ins: rehearse the full game BEFORE FLUX works.

HumanHands: prints each instruction; a teammate does it (by hand or with the leader arm) and presses Enter.
HumanEyes:  asks you what is on the board (until the camera code works).

This is also how you test that the instructions make sense and the board is reachable.
"""
from __future__ import annotations

from . import config


class HumanHands:
    def do(self, instruction: str) -> None:
        input(f"   >>> DO THIS: {instruction}   [press Enter when done] ")


class HumanEyes:
    def occupied_slots(self) -> frozenset:
        while True:
            raw = input("   Which slots have a cup? (e.g. A1 A2 B3): ").upper().replace(",", " ").split()
            if all(s in config.SLOTS for s in raw):
                return frozenset(raw)
            print(f"   Use only these names: {' '.join(config.SLOTS)}")

    def ball_visible_at(self, slot: str) -> bool:
        return input(f"   Is the ball visible at {slot}? (y/n): ").strip().lower().startswith("y")

    def ball_in_bowl(self) -> bool:
        return input("   Is the ball in the bowl? (y/n): ").strip().lower().startswith("y")
