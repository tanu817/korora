"""The two things the brain talks to.

Hands = something that performs an instruction (FLUX on the real arm, a human, or the simulator).
Eyes  = something that tells us what is on the board (camera + OpenCV, a human, or the simulator).

Swap implementations without touching the game logic.
"""
from __future__ import annotations

from typing import Protocol


class Hands(Protocol):
    def do(self, instruction: str) -> None:
        """Perform one subtask, e.g. 'slide the cup from A1 to B1'. Blocks until finished."""


class Eyes(Protocol):
    def occupied_slots(self) -> frozenset:
        """Slots that currently have a cup on them, e.g. frozenset({'A1','A2','B3'})."""

    def ball_visible_at(self, slot: str) -> bool:
        """After a reveal: is the ball lying at this slot?"""

    def ball_in_bowl(self) -> bool:
        """Is the ball in the bowl?"""
