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

# Joint angles (deg) every scoop demo starts from: mean over the 49 recorded scoop demos.
# shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (gripper is held).
START_POSE = (-5.5, -99.9, 96.4, 52.8, 1.3)

# FLUX's command history must look like the recordings, where "commands" were the LEADER arm's
# positions. At the start of each step the bridge seeds it with: follower state + this mean
# leader-follower offset (deg, from the 136 demos) and the skill's typical leader gripper value.
COMMAND_OFFSET = (0.1, -2.0, -0.4, -0.35, -0.1)
START_GRIPPER_COMMAND = {"scoop": 0.0, "to_mouth": 100.0, "feed": 0.0, "back_to_rest": 100.0}

# Max distance (deg) FLUX's remembered command may lead the measured joints, per joint (pan, lift,
# elbow, wrist_flex, wrist_roll): ~99.9th percentile of |leader - follower| in the demos. Stops the
# integrated command from drifting when the arm can't follow, while leaving the scoop's wrist roll
# room to push through the food.
COMMAND_LEAD_LIMIT = (6.0, 8.0, 8.0, 7.0, 20.0)

# Goal pose (pan, lift, elbow, wrist_flex; median end of the demos) at which the harness ends a step.
# FLUX doesn't know when a step is done: live, to_mouth reached the mouth after 6-8 s and then either
# overshot or, in a slow run, ran out of time. The step stops once within GOAL_REACHED_DEG, or, after
# coming within GOAL_NEAR_DEG, as soon as the arm moves GOAL_PASSED_DEG away from its closest point.
GOAL_POSE = {"to_mouth": (-11.9, -47.0, 37.7, 42.9)}
GOAL_REACHED_DEG = 8.0
GOAL_NEAR_DEG = 30.0
GOAL_PASSED_DEG = 3.0

# The scoop is done once the wrist roll has passed this angle (new demos: 65-103 deg, live 14-85 deg
# depending on when FLUX starts the roll); the step then runs SCOOP_FINISH_S longer to lift the spoon.
SCOOP_ROLL_DONE_DEG = 60.0
SCOOP_FINISH_S = 1.5

# Mean scene-camera RGB of each skill's training demos. The scene camera turned pink midway through
# recording (scoop/to_mouth demos are yellowish, feed/back_to_rest pink), so the bridge rescales the live
# scene colors to match the skill being run. Measured from ArnavG07/korora_spoon_feed.
SCENE_MEAN_RGB = {
    "scoop": (172, 150, 144),  # 15 new daylight demos (13:00, upweighted 3x in v2); old ones were (190, 156, 104)
    "to_mouth": (185, 151, 95),
    "feed": (159, 136, 131),  # 14 new daylight demos (13:00, upweighted 3x in v2); old ones were (215, 110, 115)
    "back_to_rest": (215, 113, 118),
}

# Seconds of arm MOTION per subtask (pauses while FLUX plans don't count).
# ~90th percentile of the recorded demo lengths. Tune after testing.
DURATIONS = {
    "scoop": 25,  # upper limit; the step ends after the scooping roll (SCOOP_ROLL_DONE_DEG)
    "to_mouth": 14,  # upper limit; the step ends at the demo goal pose (GOAL_POSE), usually after 6-8 s
    "feed": 9,
    "back_to_rest": 10,  # live: only ~2/3 of the way to rest after 6 s
}
