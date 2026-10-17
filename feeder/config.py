"""Edit here: the spoon-feeding task chain.

Instruction text must be IDENTICAL in recording, training and rollout
(same wording, same casing). FINAL: these exact strings are used in LeLab recordings.
"""

# The spoon is fixed to the gripper; the "mouth" is a cup the food is tipped into.
INSTRUCTIONS = {
    "scoop": "Scoop food from the bowl",
    "to_mouth": "Bring the spoon to the mouth",
    "feed": "Tip the food into the mouth",
    "back_to_bowl": "Move the spoon back to the bowl",
}

# The chain for one bite. The harness repeats it for several bites.
BITE = ["scoop", "to_mouth", "feed", "back_to_bowl"]
BITES = 3

# Seconds FLUX gets for each subtask in one rollout. Tune after testing.
DURATIONS = {
    "scoop": 15,
    "to_mouth": 12,
    "feed": 8,
    "back_to_bowl": 12,
}
