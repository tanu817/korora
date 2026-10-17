"""A fake world so we can test the brain without a robot.

SimHands can randomly fail (do nothing) or do the wrong legal move, so the
retry / resync / recovery logic gets exercised.
"""
from __future__ import annotations

import random

from . import config
from .board import Move, trained_moves


class SimWorld:
    """The TRUTH (the brain never looks at this directly)."""

    def __init__(self, ball_slot: str):
        self.cups = list(config.START_LAYOUT)        # cups[i] = slot, or None once lifted away
        self.ball_cup = self.cups.index(ball_slot)   # which cup the ball is under
        self.lifted_from: dict = {}                  # cup index -> slot it was lifted from
        self.ball_in_bowl = False

    def on_board(self) -> list:
        return [s for s in self.cups if s is not None]

    def apply(self, move: Move) -> None:
        if move.kind == "slide":
            src, dst = move.slots
            self.cups[self.cups.index(src)] = dst
        else:
            src, mid, dst = move.slots
            i_mid, i_src = self.cups.index(mid), self.cups.index(src)
            self.cups[i_mid] = dst
            self.cups[i_src] = mid


class SimHands:
    def __init__(self, world: SimWorld, rng: random.Random,
                 fail_rate: float = 0.0, wrong_rate: float = 0.0, verbose: bool = False):
        self.world, self.rng = world, rng
        self.fail_rate, self.wrong_rate, self.verbose = fail_rate, wrong_rate, verbose
        self._moves = {m.instruction: m for m in trained_moves()}
        self._reveals = {config.REVEAL_TEXT.format(s): s for s in config.SLOTS}

    def do(self, instruction: str) -> None:
        w = self.world
        if self.rng.random() < self.fail_rate:
            if self.verbose:
                print("   (sim: the arm missed - nothing moved)")
            return

        if instruction in self._moves:
            move = self._moves[instruction]
            if self.rng.random() < self.wrong_rate:
                others = [m for m in trained_moves() if m != move and m.is_legal(set(w.on_board()))]
                if others:
                    move = self.rng.choice(others)
                    if self.verbose:
                        print(f"   (sim: the arm did the WRONG move: {move})")
            if move.is_legal(set(w.on_board())):
                w.apply(move)

        elif instruction in self._reveals:
            slot = self._reveals[instruction]
            if slot in w.cups:
                i = w.cups.index(slot)
                w.cups[i] = None
                w.lifted_from[i] = slot

        elif instruction == config.PICK_TEXT:
            if w.ball_cup in w.lifted_from:      # ball is uncovered
                w.ball_in_bowl = True

        else:
            raise ValueError(f"sim does not know instruction: {instruction!r}")


class SimEyes:
    def __init__(self, world: SimWorld):
        self.world = world

    def occupied_slots(self) -> frozenset:
        return frozenset(self.world.on_board())

    def ball_visible_at(self, slot: str) -> bool:
        w = self.world
        return (not w.ball_in_bowl) and w.lifted_from.get(w.ball_cup) == slot

    def ball_in_bowl(self) -> bool:
        return self.world.ball_in_bowl
