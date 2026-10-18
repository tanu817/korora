"""Open-loop displacement error of FLUX checkpoints on the recorded demos (runs on the GPU pod).

For each skill, FLUX gets the real 8-tick history of a demo at several moments (states, leader commands,
scene + wrist frames) and predicts the next H commands; we compare them with what the operator did.
  ADE = mean |predicted - recorded| over the H steps, FDE = the error at step H (degrees, 5 arm joints).
A "stand still" baseline (repeat the last command) shows how much of the error is just motion.
All episodes were used for training (eval_split=0), so this measures fit to the demos, not generalization.

    python scripts/eval_displacement.py --out results/displacement.json \
        zero-shot=/workspace/models/flux-3-action-so101-normfix \
        step-1000=/workspace/outputs/korora_lora/checkpoints/001000/pretrained_model:peft ...
"""
import argparse
import gc
import json
import time

import numpy as np
import pandas as pd
import torch

from feeder.flux_remote import encode_jpeg
from feeder.flux_server import Flux
from lerobot.datasets.lerobot_dataset import LeRobotDataset

N_OBS, H = 8, 20
TASKS = ["Scoop food from the bowl", "Bring the spoon to the mouth", "Tip the food into the mouth",
         "Move the spoon back to the rest position"]


def build_queries(ds, episodes_per_task, moments):
    ep = ds.meta.episodes.to_pandas() if hasattr(ds.meta.episodes, "to_pandas") else pd.DataFrame(ds.meta.episodes)
    queries = []
    for task in TASKS:
        eps = ep[ep.tasks.str[0] == task]
        picks = np.linspace(0, len(eps) - 1, episodes_per_task).round().astype(int)
        for _, e in eps.iloc[picks].iterrows():
            i0, n = int(e.dataset_from_index), int(e.length)
            for t in np.linspace(0, n - H - 1, moments).round().astype(int):
                hist = list(range(i0 + max(t - N_OBS + 1, 0), i0 + t + 1))
                hist = [hist[0]] * (N_OBS - len(hist)) + hist
                frame = lambda i, k: (ds[i][f"observation.images.{k}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                req = {
                    "task": task,
                    "state": [ds[i]["observation.state"].tolist() for i in hist],
                    "command": [(ds[i - 1]["action"] if i > i0 else ds[i]["observation.state"]).tolist() for i in hist],
                    "scene": [encode_jpeg(frame(hist[0], "scene"), 90), encode_jpeg(frame(hist[-1], "scene"), 90)],
                    "wrist": [encode_jpeg(frame(hist[0], "wrist"), 90), encode_jpeg(frame(hist[-1], "wrist"), 90)],
                }
                true = np.stack([ds[i0 + t + k]["action"].numpy() for k in range(H)])[:, :5]
                hold = np.repeat(np.array(req["command"][-1])[None, :5], H, 0)
                queries.append({"task": task, "episode": int(e.episode_index), "t": int(t), "req": req,
                                "true": true, "hold": hold})
    return queries


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", help="name=path[:peft]")
    p.add_argument("--root", default="/workspace/hf/lerobot/ArnavG07/korora_spoon_feed")
    p.add_argument("--episodes", type=int, default=6, help="episodes per skill")
    p.add_argument("--moments", type=int, default=5, help="moments per episode")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ds = LeRobotDataset("ArnavG07/korora_spoon_feed", root=args.root)
    queries = build_queries(ds, args.episodes, args.moments)
    print(f"{len(queries)} queries", flush=True)
    results = {"H": H, "joints": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
               "models": {}}
    base = {}
    for q in queries:
        err = np.abs(q["hold"] - q["true"])
        base.setdefault(q["task"], []).append((err.mean(0), err[-1]))
    results["models"]["stand still"] = {
        t: {"ade": float(np.mean([a for a, _ in v])), "fde": float(np.mean([f for _, f in v])),
            "ade_joint": np.mean([a for a, _ in v], 0).round(3).tolist()} for t, v in base.items()}

    for spec in args.models:
        name, path = spec.split("=", 1)
        peft = path.endswith(":peft")
        path = path.removesuffix(":peft")
        t0 = time.time()
        flux = Flux(path, peft, fixed_seed=True)
        print(f"{name}: loaded in {time.time() - t0:.0f} s", flush=True)
        per_task = {}
        for q in queries:
            actions, _ = flux.act(q["req"])
            pred = np.array(actions)[:H, :5]
            err = np.abs(pred - q["true"])
            per_task.setdefault(q["task"], []).append((err.mean(0), err[-1]))
        results["models"][name] = {
            t: {"ade": float(np.mean([a for a, _ in v])), "fde": float(np.mean([f for _, f in v])),
                "ade_joint": np.mean([a for a, _ in v], 0).round(3).tolist()} for t, v in per_task.items()}
        print(name, {t[:12]: round(r["ade"], 2) for t, r in results["models"][name].items()}, flush=True)
        json.dump(results, open(args.out, "w"), indent=1)
        del flux
        gc.collect()
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
