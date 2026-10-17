"""Moves and the board state (where each cup is, which cup has the ball)."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config


@dataclass(frozen=True)
class Move:
    kind: str           # "slide" or "push"
    slots: tuple        # (from, to) or (from, middle, to)

    def is_legal(self, occupied: set) -> bool:
        if self.kind == "slide":
            src, dst = self.slots
            return src in occupied and dst not in occupied
        src, mid, dst = self.slots
        return src in occupied and mid in occupied and dst not in occupied

    def result(self, occupied: set) -> frozenset:
        """Which slots are occupied after this move."""
        occ = set(occupied)
        occ.discard(self.slots[0])
        occ.add(self.slots[-1])
        if self.kind == "push":
            occ.add(self.slots[1])  # middle stays occupied (by the pushing cup)
        return frozenset(occ)

    @property
    def instruction(self) -> str:
        template = config.SLIDE_TEXT if self.kind == "slide" else config.PUSH_TEXT
        return template.format(*self.slots)

    def __str__(self) -> str:
        return f"{self.kind} {' -> '.join(self.slots)}"


def trained_moves() -> list:
    return [Move(kind, tuple(slots)) for kind, slots in config.TRAINED_MOVES]


def reveal_instruction(slot: str) -> str:
    return config.REVEAL_TEXT.format(slot)


def all_instructions() -> list:
    """Every distinct instruction FLUX must learn (= what you record demos for)."""
    out = [m.instruction for m in trained_moves()]
    out += [reveal_instruction(s) for s in sorted(config.END_LAYOUT)]
    out.append(config.PICK_TEXT)
    return out


@dataclass
class BoardState:
    """What the brain believes. cups[i] = slot of cup i. ball_cup = index of the ball's cup."""
    cups: list = field(default_factory=lambda: list(config.START_LAYOUT))
    ball_cup: int = 0
    certain: bool = True   # False after an unexplained change: ball position is a guess

    @property
    def occupied(self) -> frozenset:
        return frozenset(self.cups)

    @property
    def ball_slot(self) -> str:
        return self.cups[self.ball_cup]

    def apply(self, move: Move) -> None:
        if not move.is_legal(set(self.cups)):
            raise ValueError(f"illegal move {move} for cups at {self.cups}")
        if move.kind == "slide":
            src, dst = move.slots
            self.cups[self.cups.index(src)] = dst
        else:
            src, mid, dst = move.slots
            i_mid, i_src = self.cups.index(mid), self.cups.index(src)
            self.cups[i_mid] = dst
            self.cups[i_src] = mid

    def resync(self, observed: frozenset) -> None:
        """Observed cups don't match what we expected and no single move explains it.
        Keep cups that are still where we think; move the rest to the new slots."""
        new_slots = sorted(observed - set(self.cups))
        for i, slot in enumerate(self.cups):
            if slot not in observed and new_slots:
                self.cups[i] = new_slots.pop(0)
        self.certain = False

    def ascii(self, show_ball: bool = True) -> str:
        rows = []
        for r in "AB":
            cells = []
            for c in "123":
                s = r + c
                if s in self.cups:
                    has_ball = show_ball and self.ball_slot == s
                    cells.append("[o]" if has_ball else "[ ]")
                else:
                    cells.append(" . ")
            rows.append(f"{r}  " + " ".join(cells))
        return "    1   2   3\n" + "\n".join(rows)
