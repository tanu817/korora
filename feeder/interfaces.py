"""The things the harness talks to.

Hands = something that performs one instruction (FLUX on the real arm, or a human while rehearsing).
Eyes  = something that checks whether a step worked (camera + OpenCV later, or a human for now).

Swap implementations without touching the task logic.
"""
from __future__ import annotations

from typing import Protocol


class Hands(Protocol):
    def do(self, instruction: str) -> None:
        """Perform one subtask, e.g. 'Scoop food from the bowl'. Blocks until finished."""


class Eyes(Protocol):
    # TODO: decide which checks the spoon-feeding harness needs, e.g.:
    #   spoon_in_gripper() -> bool
    #   spoon_has_food()   -> bool
    #   spoon_at_mouth()   -> bool
    ...
