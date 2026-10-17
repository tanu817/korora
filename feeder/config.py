"""Edit here: the spoon-feeding task chain.

Instruction text must be IDENTICAL in recording, training and rollout
(same wording, same casing). Draft wording; finalise before recording demos.
"""

INSTRUCTIONS = {
    "pick_spoon": "pick up the spoon",
    "scoop": "scoop food from the bowl",
    "to_mouth": "bring the spoon to the mouth",
    "back_to_bowl": "move the spoon back to the bowl",
    "place_spoon": "put the spoon down",
}

# Seconds FLUX gets for each subtask in one rollout. Tune after testing.
DURATIONS = {
    "pick_spoon": 15,
    "scoop": 15,
    "to_mouth": 12,
    "back_to_bowl": 12,
    "place_spoon": 12,
}
