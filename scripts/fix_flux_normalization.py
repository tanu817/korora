"""Write the corrected SO-101 joint normalization into a copy of the FLUX checkpoint.

The released flux-3-action-so101 processors ship quantiles in a different joint
convention (shoulder_lift flipped, +-90 deg offsets) than raw LeRobot SO-101 degrees
(--robot.use_degrees=true). The organizers' fix replaces them:
https://huggingface.co/spaces/multimodalart/flux-3-action-so101-sim/commit/09edebb006d793cb215ce8c0fccfc9eacd71dfc6

Training (--policy.path) and rollout both load normalization from the checkpoint's
processor state files, so pointing both at the patched copy applies the fix everywhere.
Big weight files are symlinked, not copied. The original checkpoint is not modified.
Edits the state files directly, so it runs without a GPU.

Usage:
    python scripts/fix_flux_normalization.py models/flux-3-action-so101 models/flux-3-action-so101-normfix
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
# State is absolute degrees (gripper %); action is per-step command deltas (gripper absolute).
NORMALIZATION = {
    "state": {
        "q01": [-37.969, -99.316, -45.78, 22.987, -68.712, 0.45],
        "q99": [35.288, 43.076, 90.308, 95.704, 17.753, 40.324],
    },
    "action": {
        "q01": [-1.652, -3.235, -3.147, -2.198, -1.708, 0.0],
        "q99": [1.701, 3.487, 3.307, 2.072, 1.699, 40.733],
    },
}

HISTORY_STEPS = {
    "flux3_observation_history_normalizer",
    "flux3_action_target_normalizer",
    "flux3_action_history_unnormalizer",
}
PIPELINES = ("policy_preprocessor.json", "policy_postprocessor.json")
SYMLINK_ABOVE_BYTES = 50 * 1024 * 1024


def copy_checkpoint(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=False)
    for path in src.rglob("*"):
        target = dst / path.relative_to(src)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.stat().st_size > SYMLINK_ABOVE_BYTES:
            target.symlink_to(path.resolve())
        else:
            shutil.copy2(path, target)


def patch_state_file(path: Path) -> None:
    old = load_file(path)
    new = {}
    for key, value in old.items():
        stream, q = key.split(".")  # e.g. "state.q01"
        new[key] = torch.tensor(NORMALIZATION[stream][q], dtype=value.dtype)
        if new[key].shape != value.shape:
            raise SystemExit(f"{path.name}: {key} has shape {tuple(value.shape)}, expected 6 joints")
        print(f"  {key:11s} old {[round(v, 3) for v in value.tolist()]}")
        print(f"  {'':11s} new {[round(v, 3) for v in new[key].tolist()]}")
    save_file(new, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="downloaded flux-3-action-so101 checkpoint")
    parser.add_argument("dst", type=Path, help="new folder for the patched copy (must not exist)")
    args = parser.parse_args()

    copy_checkpoint(args.src, args.dst)
    patched = 0
    for pipeline in PIPELINES:
        for step in json.loads((args.dst / pipeline).read_text())["steps"]:
            if step["registry_name"] in HISTORY_STEPS and "state_file" in step:
                print(f"{pipeline} / {step['registry_name']}")
                patch_state_file(args.dst / step["state_file"])
                patched += 1
    if patched != len(HISTORY_STEPS):
        raise SystemExit(f"Expected {len(HISTORY_STEPS)} FLUX normalization steps, patched {patched}")

    # Informational only (never read by LeRobot), but keeps the copy self-describing.
    config_path = args.dst / "config.json"
    config = json.loads(config_path.read_text())
    config["normalization_stats"] = NORMALIZATION
    config_path.write_text(json.dumps(config, indent=4))
    stats_path = args.dst / "dataset_statistics.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        for stream, quantiles in NORMALIZATION.items():
            stats[stream].update(quantiles)
        stats_path.write_text(json.dumps(stats, indent=2))

    print(f"\nPatched {patched} processor steps -> {args.dst}")
    print("Use this folder as --policy.path for zero-shot, training and rollout.")


if __name__ == "__main__":
    main()
