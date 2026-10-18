"""Edit here: the spoon-feeding task chain.

Instruction text must be IDENTICAL in recording, training and rollout
(same wording, same casing). FINAL: these exact strings are used in LeLab recordings.
"""

# The spoon is fixed to the gripper; the "mouth" is a cup the food is tipped into.
INSTRUCTIONS = {
    "scoop": "Scoop food from the bowl",
    "to_mouth": "Bring the spoon to the mouth",
    "feed": "Tip the food into the mouth",
    "back_to_rest": "Move the spoon back to the rest position",
}

# The chain for one bite. The harness repeats it for several bites.
BITE = ["scoop", "to_mouth", "feed", "back_to_rest"]
BITES = 3

# Seconds of arm MOTION per subtask (pauses while FLUX plans don't count).
# ~90th percentile of the recorded demo lengths. Tune after testing.
DURATIONS = {
    "scoop": 11,
    "to_mouth": 10,
    "feed": 9,
    "back_to_rest": 6,
}
