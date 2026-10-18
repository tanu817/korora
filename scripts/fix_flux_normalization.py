"""Write the corrected SO-101 joint normalization into a copy of the FLUX checkpoint.

The released flux-3-action-so101 processors ship quantiles that do not match raw
LeRobot SO-101 degrees (--robot.use_degrees=true). The organizers' fix replaces them:
https://huggingface.co/spaces/multimodalart/flux-3-action-so101-sim/commit/09edebb006d793cb215ce8c0fccfc9eacd71dfc6

Training (--policy.path) and rollout both load normalization from the checkpoint's
processor files, so pointing both at the patched copy applies the fix everywhere.
Big weight files are symlinked, not copied. The original checkpoint is not modified.

Usage:
    python scripts/fix_flux_normalization.py models/flux-3-action-so101 models/flux-3-action-so101-normfix
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

from lerobot.policies.flux3.processor_flux3 import _HistoryQuantiles
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

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


def patch_pipeline(directory: Path, name: str) -> int:
    filename = f"{name}.json"
    pipeline = PolicyProcessorPipeline.from_pretrained(directory, config_filename=filename)
    patched = 0
    for step in pipeline.steps:
        if not isinstance(step, _HistoryQuantiles):
            continue
        old = step.state_dict()
        new = {
            f"{stream}.{q}": torch.tensor(NORMALIZATION[stream][q], dtype=torch.float32)
            for stream in step.streams
            for q in ("q01", "q99")
        }
        step.load_state_dict(new)
        print(f"{name} / {type(step).__name__}")
        for key in sorted(new):
            print(f"  {key:11s} old {[round(v, 3) for v in old[key].tolist()]}")
            print(f"  {'':11s} new {[round(v, 3) for v in new[key].tolist()]}")
        patched += 1
    pipeline.save_pretrained(directory, config_filename=filename)
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="downloaded flux-3-action-so101 checkpoint")
    parser.add_argument("dst", type=Path, help="new folder for the patched copy (must not exist)")
    args = parser.parse_args()

    copy_checkpoint(args.src, args.dst)
    patched = sum(
        patch_pipeline(args.dst, name)
        for name in (POLICY_PREPROCESSOR_DEFAULT_NAME, POLICY_POSTPROCESSOR_DEFAULT_NAME)
    )
    if patched == 0:
        raise SystemExit("No FLUX history normalization steps found; is this a flux3 checkpoint?")

    # Informational only (the model never reads it), but keeps the copy self-describing.
    config_path = args.dst / "config.json"
    config = json.loads(config_path.read_text())
    config["normalization_stats"] = NORMALIZATION
    config_path.write_text(json.dumps(config, indent=4))

    print(f"\nPatched {patched} processor steps -> {args.dst}")
    print("Use this folder as --policy.path for zero-shot, training and rollout.")


if __name__ == "__main__":
    main()
