"""Choosing the next shuffle move."""
from __future__ import annotations

import random
from collections import deque

from . import config
from .board import Move, trained_moves


def shortest_path_to_end(occupied: frozenset, moves: list | None = None) -> list | None:
    """Fewest trained moves to bring the cups back to END_LAYOUT (BFS)."""
    moves = moves or trained_moves()
    start = frozenset(occupied)
    if start == config.END_LAYOUT:
        return []
    queue, seen = deque([(start, [])]), {start}
    while queue:
        occ, path = queue.popleft()
        for m in moves:
            if m.is_legal(set(occ)):
                nxt = m.result(occ)
                if nxt == config.END_LAYOUT:
                    return path + [m]
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, path + [m]))
    return None


def choose_next_move(occupied: frozenset, moves_done: int, min_moves: int,
                     rng: random.Random, last: Move | None = None) -> Move | None:
    """Return the next move, or None when the shuffle is finished.

    - Before `min_moves`: a random legal trained move.
    - After `min_moves`: stop as soon as cups are in END_LAYOUT, otherwise keep moving randomly.
    - After MAX_MOVES: take the shortest path home.
    """
    if moves_done >= min_moves and occupied == config.END_LAYOUT:
        return None
    if moves_done >= config.MAX_MOVES:
        path = shortest_path_to_end(occupied)
        if path is None:
            raise RuntimeError(f"cannot get back to {sorted(config.END_LAYOUT)} from {sorted(occupied)}")
        return path[0] if path else None
    legal = [m for m in trained_moves() if m.is_legal(set(occupied))]
    if not legal:
        raise RuntimeError(f"no legal trained move from {sorted(occupied)}")
    return rng.choice(legal)
