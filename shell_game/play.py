"""Run the shell game.

Examples (run from the repo folder):
  python -m shell_game.play                          # simulation, 1 game
  python -m shell_game.play --games 200 --quiet --fail-rate 0.15 --wrong-rate 0.05
  python -m shell_game.play --hands human --eyes human   # rehearse with a teammate / leader arm
  python -m shell_game.play --list-instructions      # what to record demos for
  python -m shell_game.play --hands flux --eyes human --policy models/flux-3-action-so101 \
        --log results/log.csv --phase zero-shot --checkpoint "so101@c9e13b2"
"""
from __future__ import annotations

import argparse
import random
import sys

from . import config
from .board import all_instructions
from .runner import NeedHuman, append_log, play_game


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Robot shell game")
    p.add_argument("--hands", choices=["sim", "human", "flux"], default="sim")
    p.add_argument("--eyes", choices=["sim", "human"], default="sim")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--ball", choices=list(config.START_LAYOUT), default=None,
                   help="where the ball is hidden (asked / random if not given)")
    p.add_argument("--fail-rate", type=float, default=0.0, help="sim: chance a move does nothing")
    p.add_argument("--wrong-rate", type=float, default=0.0, help="sim: chance the arm does a wrong move")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--list-instructions", action="store_true")
    # FLUX
    p.add_argument("--policy", help="checkpoint path for --hands flux")
    p.add_argument("--peft", action="store_true", help="policy is our LoRA checkpoint")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--scene-cam", default="0")
    p.add_argument("--wrist-cam", default="1")
    p.add_argument("--dry-run", action="store_true", help="flux: print commands, don't move the arm")
    # logging
    p.add_argument("--log", help="append each game to this CSV (e.g. results/log.csv)")
    p.add_argument("--phase", default="sim", help="zero-shot / post-training / sim")
    p.add_argument("--checkpoint", default="", help="which model checkpoint was used")
    a = p.parse_args(argv)

    if a.list_instructions:
        ins = all_instructions()
        print(f"{len(ins)} instructions to record demos for (copy them EXACTLY):")
        for i, t in enumerate(ins, 1):
            print(f"  {i:2}. {t}")
        return 0

    rng = random.Random(a.seed)
    wins = first_try = 0
    total_moves = total_recov = total_interv = 0

    for g in range(1, a.games + 1):
        ball = a.ball or (rng.choice(config.START_LAYOUT) if a.eyes == "sim"
                          else input(f"Where is the ball hidden? {config.START_LAYOUT}: ").strip().upper())
        hands, eyes = _make(a, rng, ball)
        if not a.quiet:
            print(f"\n=========== GAME {g} ===========")
        try:
            res = play_game(hands, eyes, ball, rng, verbose=not a.quiet,
                            on_stuck=_ask_human if a.eyes == "human" else None)
        except NeedHuman as e:
            print(f"STOPPED: {e}")
            return 1
        wins += res.success
        first_try += res.found_first_try
        total_moves += res.shuffle_moves
        total_recov += res.recoveries
        total_interv += res.interventions
        if a.log:
            append_log(a.log, a.phase, a.checkpoint, g, res)

    n = a.games
    print(f"\n{n} game(s): success {wins}/{n}, ball found on first try {first_try}/{n}, "
          f"avg shuffle moves {total_moves / n:.1f}, recoveries {total_recov}, human interventions {total_interv}")
    return 0


def _make(a, rng, ball):
    if a.hands == "sim" or a.eyes == "sim":
        from .sim import SimEyes, SimHands, SimWorld
        world = SimWorld(ball)
    if a.hands == "sim":
        hands = SimHands(world, rng, a.fail_rate, a.wrong_rate, verbose=not a.quiet)
    elif a.hands == "human":
        from .human import HumanHands
        hands = HumanHands()
    else:
        if not a.policy:
            sys.exit("--hands flux needs --policy <checkpoint path>")
        from .flux_hands import FluxHands, FluxSettings
        cam = lambda v: int(v) if v.isdigit() else v
        hands = FluxHands(FluxSettings(a.policy, a.peft, a.port, scene_cam=cam(a.scene_cam),
                                       wrist_cam=cam(a.wrist_cam), dry_run=a.dry_run))
    if a.eyes == "sim":
        eyes = SimEyes(world)
    else:
        from .human import HumanEyes
        eyes = HumanEyes()
    return hands, eyes


def _ask_human(state, observed):
    input("   HUMAN: fix the board (put the cups where they should be), then press Enter ")


if __name__ == "__main__":
    raise SystemExit(main())
