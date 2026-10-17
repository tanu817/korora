# Korora — Robot Shell Game 🥤🔴

**ERH Physical AI Hackathon · Black Forest Labs track "Imagine, Then Act"**

An SO-101 robot arm plays the shell game on its own: a human hides a ball under one of 3 cups,
the robot **shuffles the cups itself** (sliding cups and pushing one cup with another),
then **finds the ball**, **recovers** if it guessed wrong, and drops the ball into a bowl.
Every physical move is performed live by **FLUX 3 Action**; a small Python "brain" sequences,
checks and recovers.

## How it works

```
 BRAIN (shell_game/)                          HANDS (FLUX 3 Action)
 ─────────────────────                        ──────────────────────
 pick a random legal move ──"slide the cup     scene + wrist camera
 track which cup has ball    from A1 to B1"──▶  + joint state + text
 look at board, check move ◀── board changes ── → arm actions (replans ~1 s)
 retry / resync / recover
```

This is the "Harness + FLUX" design recommended by the organizers: FLUX does atomic skills,
the harness handles the long-horizon chain.

## The board and the moves

```
    1   2   3
A  [ ] [ ] [ ]   ← cups start AND end here
B   .   .   .
```

Only **6 trained moves** (5 slides + 1 push). We chose them with an exhaustive search over all
small move sets on the 2×3 board: this set mixes the ball best (the best possible guess is ~41%,
perfect would be 33%), never gets stuck, and always brings the cups back to row A, so "reveal"
only has to be learned for 3 slots. A game averages **~13 shuffle moves → ~17 FLUX subtasks**.

Simpler-looking sets (e.g. row pushes + column slides) barely mix: the ball stays in place ~84%
of the time.

## Quick start (no robot, any laptop, no installs)

```
python -m shell_game.play                       # watch one simulated game
python -m shell_game.play --list-instructions   # the 10 instructions to record demos for
python -m shell_game.play --games 1000 --quiet --fail-rate 0.15 --wrong-rate 0.05
python -m unittest discover tests               # run the tests
```

## Rehearse with humans (before FLUX works)

```
python -m shell_game.play --hands human --eyes human
```
The program tells a teammate each move; they do it with the leader arm (or by hand) and type what
the board looks like. Great for testing the board and practising the demo.

## Run with FLUX on the real arm ⚠️ moves the arm

On the machine with the NVIDIA GPU, the arm and both cameras:
```
# print the commands only
python -m shell_game.play --hands flux --eyes human --policy models/flux-3-action-so101 --dry-run
# zero-shot (released checkpoint), logged for grading
python -m shell_game.play --hands flux --eyes human --policy models/flux-3-action-so101 \
    --log results/log.csv --phase zero-shot --checkpoint "flux-3-action-so101@c9e13b2"
# our fine-tuned LoRA
python -m shell_game.play --hands flux --eyes human --peft --policy outputs/so101_lora/checkpoints/last/pretrained_model \
    --log results/log.csv --phase post-training --checkpoint "lora-<step>"
```

## Files

| File | What it does |
|---|---|
| `shell_game/config.py` | **Edit here**: board, trained moves, exact instruction text |
| `shell_game/board.py` | Moves + what the brain believes (cup positions, ball) |
| `shell_game/shuffle.py` | Picks the next random legal move; shortest way home |
| `shell_game/runner.py` | The game loop: shuffle → find → recover → bowl, + CSV logging |
| `shell_game/sim.py` | Fake arm/camera that can miss or do wrong moves |
| `shell_game/human.py` | Human arm/eyes for rehearsing |
| `shell_game/flux_hands.py` | Calls FLUX 3 Action (`lerobot-rollout`) per subtask |
| `results/log.csv` | Zero-shot vs post-training results |

## Time split

_TODO: fill in at the end (required by the track)._

## Team

_TODO_
