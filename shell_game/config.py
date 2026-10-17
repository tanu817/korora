"""
Game configuration. EDIT THIS FILE to change the board, the trained moves
or the exact instruction wording.

IMPORTANT: the instruction text must be EXACTLY the same when you
(1) record demos, (2) train, and (3) run the robot. Copy-paste from here.
"""

# Board: 2 rows x 3 columns
#         col1  col2  col3
# row A:  A1    A2    A3     <- cups start here and end here
# row B:  B1    B2    B3
SLOTS = ["A1", "A2", "A3", "B1", "B2", "B3"]

# Where the 3 cups stand when the game starts, and where they must
# be when the shuffle is finished (only these slots need a "reveal" skill).
START_LAYOUT = ("A1", "A2", "A3")
END_LAYOUT = frozenset(START_LAYOUT)

# The ONLY moves the robot is trained on.
# Chosen by an exhaustive search over all small move sets (see README):
# this set mixes the ball best (best guess ~41%, perfect would be 33%),
# never gets stuck, and always returns all cups to row A.
#   ("slide", (from, to))           move one cup into an empty slot
#   ("push",  (from, middle, to))   cup at `from` pushes the cup at `middle`
#                                   into the empty slot `to`
TRAINED_MOVES = [
    ("slide", ("A1", "B1")),
    ("slide", ("B1", "A1")),
    ("slide", ("B1", "B2")),
    ("slide", ("B2", "B3")),
    ("slide", ("B3", "A3")),
    ("push", ("A3", "A2", "A1")),
]

# Instruction text sent to FLUX for each skill.
SLIDE_TEXT = "slide the cup from {0} to {1}"
PUSH_TEXT = "push the cup at {0} into the cup at {1} so it moves to {2}"
REVEAL_TEXT = "lift the cup at {0} and put it aside"
PICK_TEXT = "pick up the ball and put it in the bowl"

# Shuffle length: the robot makes at least this many moves (random in range),
# then keeps going until all cups are back in END_LAYOUT.
MIN_MOVES_RANGE = (5, 10)
# Safety cap: after this many moves, take the shortest path back to END_LAYOUT.
MAX_MOVES = 20

# How often to retry a move that did nothing before asking a human for help.
MAX_RETRIES = 2
