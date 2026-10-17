"""The game loop (the "harness"): shuffle -> find the ball -> recover if wrong -> ball into bowl.

After EVERY subtask it looks at the board and checks the move really happened.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import config
from .board import BoardState, Move, reveal_instruction, trained_moves
from .interfaces import Eyes, Hands
from .shuffle import choose_next_move


@dataclass
class GameResult:
    success: bool = False
    shuffle_moves: int = 0          # moves that actually happened
    steps_completed: int = 0        # every successful subtask (moves + reveals + pick)
    retries: int = 0                # a move did nothing and was repeated
    resyncs: int = 0                # board changed in an unexpected way
    reveals: int = 0                # cups lifted to find the ball
    found_first_try: bool = False
    interventions: int = 0          # times a human had to help
    events: list = field(default_factory=list)

    @property
    def recoveries(self) -> int:
        return self.retries + self.resyncs + max(0, self.reveals - 1)


class NeedHuman(Exception):
    """The harness is stuck and a human must fix the board."""


def play_game(hands: Hands, eyes: Eyes, ball_slot: str, rng: random.Random,
              verbose: bool = True, on_stuck=None) -> GameResult:
    """Play one full game. `ball_slot` = where the human hid the ball (row A).

    `on_stuck(state, observed)` is called when retries run out; it should let a
    human fix the board and return. If None, the game stops.
    """
    res = GameResult()
    say = print if verbose else (lambda *a, **k: None)

    def log(msg: str) -> None:
        res.events.append(msg)
        say(msg)

    state = BoardState(ball_cup=list(config.START_LAYOUT).index(ball_slot))
    observed = eyes.occupied_slots()
    if observed != state.occupied:
        raise NeedHuman(f"board should start with cups at {sorted(state.occupied)}, camera sees {sorted(observed)}")

    min_moves = rng.randint(*config.MIN_MOVES_RANGE)
    log(f"Ball hidden at {ball_slot}. Shuffling at least {min_moves} moves.")
    say(state.ascii())

    # ---------------- 1. SHUFFLE ----------------
    last = None
    while True:
        move = choose_next_move(state.occupied, res.shuffle_moves, min_moves, rng, last)
        if move is None:
            break
        expected = move.result(state.occupied)
        before = state.occupied
        for attempt in range(config.MAX_RETRIES + 1):
            log(f"Move {res.shuffle_moves + 1}: {move}   -> FLUX: \"{move.instruction}\"")
            hands.do(move.instruction)
            observed = eyes.occupied_slots()
            if observed == expected:
                state.apply(move)
                res.shuffle_moves += 1
                res.steps_completed += 1
                break
            if observed == before:
                res.retries += 1
                log("   nothing changed -> retry")
                continue
            # something else happened: can one trained move explain it?
            explained = [m for m in trained_moves()
                         if m.is_legal(set(before)) and m.result(before) == observed]
            if explained:
                state.apply(explained[0])
                log(f"   arm did a different move ({explained[0]}) -> tracked it")
            else:
                state.resync(observed)
                log(f"   unexpected board {sorted(observed)} -> resync, ball position now a GUESS")
            res.resyncs += 1
            res.shuffle_moves += 1
            res.steps_completed += 1
            break
        else:
            res.interventions += 1
            log("   still nothing after retries -> NEED HUMAN")
            if on_stuck is None:
                return res
            on_stuck(state, eyes.occupied_slots())
            observed = eyes.occupied_slots()
            if observed != state.occupied:
                state.resync(observed)
        say(state.ascii(show_ball=state.certain))
        last = move

    # ---------------- 2. FIND THE BALL ----------------
    guess = state.ball_slot
    order = [guess] + sorted(s for s in state.occupied if s != guess)
    log(f"Shuffle done ({res.shuffle_moves} moves). I think the ball is at {guess}"
        + ("" if state.certain else " (guess - tracking was lost)") + ".")
    found = None
    for slot in order:
        for attempt in range(config.MAX_RETRIES + 1):
            log(f"Reveal {slot}   -> FLUX: \"{reveal_instruction(slot)}\"")
            hands.do(reveal_instruction(slot))
            if slot not in eyes.occupied_slots():
                break
            res.retries += 1
            log("   cup was not lifted -> retry")
        else:
            res.interventions += 1
            log("   cup still not lifted -> NEED HUMAN")
            if on_stuck is None:
                return res
            on_stuck(state, eyes.occupied_slots())
        res.reveals += 1
        res.steps_completed += 1
        if eyes.ball_visible_at(slot):
            found = slot
            break
        log("   no ball here -> RECOVERY: try the next cup")
    if found is None:
        log("Ball not found under any cup!")
        return res
    res.found_first_try = found == guess
    log(f"Found the ball at {found}" + (" on the first try!" if res.found_first_try else " after recovery."))

    # ---------------- 3. BALL INTO BOWL ----------------
    for attempt in range(config.MAX_RETRIES + 1):
        log(f"Pick   -> FLUX: \"{config.PICK_TEXT}\"")
        hands.do(config.PICK_TEXT)
        if eyes.ball_in_bowl():
            res.steps_completed += 1
            res.success = True
            log("Ball is in the bowl. DONE!")
            return res
        res.retries += 1
        log("   ball not in bowl -> retry")
    res.interventions += 1
    log("Could not get the ball into the bowl -> NEED HUMAN")
    return res


LOG_COLUMNS = ["timestamp", "phase", "checkpoint", "instruction", "attempt", "success",
               "steps_completed", "steps_total", "recoveries", "interventions", "notes"]


def append_log(path: str | Path, phase: str, checkpoint: str, attempt: int, res: GameResult,
               notes: str = "") -> None:
    """Add one game to results/log.csv (used for the zero-shot vs post-training comparison)."""
    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOG_COLUMNS)
        steps_total = res.shuffle_moves + res.reveals + 1
        w.writerow([datetime.now().isoformat(timespec="seconds"), phase, checkpoint, "full game",
                    attempt, int(res.success), res.steps_completed, steps_total,
                    res.recoveries, res.interventions, notes])
