"""The harness: runs the feeding chain bite after bite through FLUX (the hands).

    # simulation, no arm and no GPU needed:
    python -m feeder.chain --sim --bites 2
    # real arm, autonomous (needs the tunnel to feeder/flux_server.py, see flux_remote.py):
    python -m feeder.chain --server http://localhost:8001 --checkpoint lora-step-10000
    # real arm, a person judges every step (Enter = worked, r = retry, q = quit):
    python -m feeder.chain --server http://localhost:8001 --checkpoint lora-step-10000 --check

Every step and a summary of the whole chain are appended to results/log.csv.

!!! Without --sim THIS MOVES THE REAL ARM. Keep a hand on the power switch. !!!
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from . import config
from .interfaces import Hands

RESULTS = Path(__file__).resolve().parent.parent / "results"
LOG = RESULTS / "log.csv"  # graded; --sim writes to sim_log.csv instead


class SimHands:
    """Stands in for FLUX so the chain logic can be tested without the arm or the GPU."""

    def do(self, instruction: str) -> None:
        print(f"   [sim] FLUX would now do: {instruction!r}")
        time.sleep(0.2)

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def log_row(phase: str, checkpoint: str, instruction: str, attempt: int, success: str,
            steps_completed: int, steps_total: int, recoveries: int, interventions: int, notes: str) -> None:
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        if new:
            csv.writer(f).writerow(["timestamp", "phase", "checkpoint", "instruction", "attempt", "success",
                                    "steps_completed", "steps_total", "recoveries", "interventions", "notes"])
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), phase, checkpoint, instruction, attempt, success,
            steps_completed, steps_total, recoveries, interventions, notes,
        ])


def ask(instruction: str) -> str:
    """Human check after a step. Returns 'ok', 'retry' or 'quit'."""
    answer = input(f"   Did '{instruction}' work? [Enter = yes, r = retry, q = quit] ").strip().lower()
    return {"": "ok", "r": "retry", "q": "quit"}.get(answer, "ok")


def run_chain(hands: Hands, bites: int, check: bool, max_retries: int, phase: str, checkpoint: str) -> None:
    steps_total = bites * len(config.BITE)
    completed = recoveries = 0
    stopped = False
    for bite in range(1, bites + 1):
        print(f"\n=== Bite {bite}/{bites} ===")
        for key in config.BITE:
            instruction = config.INSTRUCTIONS[key]
            for attempt in range(1, max_retries + 2):
                print(f"-> {instruction} (attempt {attempt})")
                hands.do(instruction)
                verdict = ask(instruction) if check else "unchecked"
                success = {"ok": "1", "retry": "0", "quit": "0"}.get(verdict, "")
                log_row(phase, checkpoint, instruction, attempt, success, 0, 1, 0, 0,
                        f"chain bite {bite}/{bites}" + ("" if check else ", autonomous (no success check)"))
                if verdict == "quit":
                    stopped = True
                    break
                if verdict != "retry":
                    completed += 1
                    break
                recoveries += 1
            else:
                print(f"   giving up on '{instruction}' after {max_retries} retries")
                stopped = True
            if stopped:
                break
        if stopped:
            break
    log_row(phase, checkpoint, f"FULL CHAIN ({bites} bites)", 1, "1" if completed == steps_total else "0",
            completed, steps_total, recoveries, 0, "check mode (human judged steps)" if check else "autonomous")
    print(f"\nChain finished: {completed}/{steps_total} steps, {recoveries} retries. Logged to {LOG}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bites", type=int, default=config.BITES)
    parser.add_argument("--check", action="store_true", help="a person judges each step and can ask for a retry")
    parser.add_argument("--max-retries", type=int, default=1, help="retries per step in --check mode")
    parser.add_argument("--phase", default="post-training", help="zero-shot or post-training (for the log)")
    parser.add_argument("--checkpoint", default="", help="which model ran, for the log")
    parser.add_argument("--server", default="http://localhost:8001")
    parser.add_argument("--sim", action="store_true", help="no arm, no GPU: print the steps only")
    parser.add_argument("--yes", action="store_true", help="skip the keyboard confirmation (confirmed elsewhere)")
    parser.add_argument("--start-delay", type=float, default=3.0, help="seconds to let go of the arm")
    parser.add_argument("--exec-steps", type=int, default=10,
                        help="actions executed per FLUX request before replanning (model predicts 32)")
    args = parser.parse_args()

    global LOG
    if args.sim:
        LOG = RESULTS / "sim_log.csv"
        hands = SimHands()
        checkpoint, phase = args.checkpoint or "sim", "sim"
    else:
        from .flux_remote import RemoteFluxHands, RemoteSettings

        hands = RemoteFluxHands(
            RemoteSettings(server=args.server, confirm_each=False, exec_steps=args.exec_steps)
        )
        checkpoint, phase = args.checkpoint or hands.info["policy_path"], args.phase
    try:
        if not args.sim:
            if not args.yes:
                input(f"ARM WILL MOVE for {args.bites} bites. Hand on the power switch? [Enter / Ctrl+C] ")
            print(f"Motors on, holding still for {args.start_delay:.0f} s. Let go of the arm now.", flush=True)
            time.sleep(args.start_delay)
        run_chain(hands, args.bites, args.check, args.max_retries, phase, checkpoint)
    except KeyboardInterrupt:
        hands.stop()
        print("\nStopped.")
    finally:
        hands.close()


if __name__ == "__main__":
    main()
