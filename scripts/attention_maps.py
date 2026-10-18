"""Attention maps: which image regions FLUX's predicted-action tokens attend to (runs on the GPU pod).

FLUX 3 Action runs 28 joint ("single stream") blocks over one sequence [text | video | actions]. We hook
the attention of those blocks, take the softmax rows of the predicted-action tokens, average them over
heads, action tokens, blocks and denoising steps, and map the mass on the video-conditioning tokens back
to their (row, column) in the camera canvas (scene left, wrist right). We also report how the attention
splits between text, scene camera, wrist camera, state/past commands and the actions themselves.

    python scripts/attention_maps.py --policy <ckpt>/pretrained_model --peft --out results/attention
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from feeder.flux_remote import encode_jpeg
from feeder.flux_server import Flux
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flux3.f3 import transformer as T

N_OBS = 8
STATE = {"on": False, "layout": None, "acc": None, "count": 0}

_orig_seq_forward = T.JointSingleSeq.forward
_orig_attn_forward = T.Attention.forward


def _seq_forward(self, ctx, ctx_ids, vector, timesteps_ctx, pag_single_layers=None, pag_strength=1.0, **kw):
    if STATE["on"]:
        order, lens, ids = [], {}, {}
        for key, mod in self.sequence.items():
            if key in kw:
                if mod not in order:
                    order.append(mod)
                    lens[mod], ids[mod] = 0, []
                lens[mod] += kw[key].shape[1]
                ids[mod].append(kw[key + "_ids"][0].cpu())
        STATE["layout"] = {"txt": ctx.shape[1], "order": order, "lens": lens,
                           "ids": {m: torch.cat(v, 0) for m, v in ids.items()}}
    return _orig_seq_forward(self, ctx, ctx_ids, vector, timesteps_ctx, pag_single_layers, pag_strength, **kw)


def _segments(layout):
    seg, off = {"text": (0, layout["txt"])}, layout["txt"]
    for m in layout["order"]:
        seg[m] = (off, off + layout["lens"][m])
        off += layout["lens"][m]
    return seg, off


def _attn_forward(self, q, k, v, attn_mode="cudnn"):
    layout = STATE["layout"]
    if STATE["on"] and layout is not None:
        seg, total = _segments(layout)
        act = [m for m in layout["order"] if m not in ("video", "video_cond") and not m.endswith("_cond")]
        if q.shape[2] == total and act:
            a0, a1 = seg[act[0]]
            scores = (q[:, :, a0:a1].float() @ k.float().transpose(-1, -2)) * q.shape[-1] ** -0.5
            p = scores.softmax(-1).mean(dim=(1, 2))[0]
            STATE["acc"] = p if STATE["acc"] is None else STATE["acc"] + p
            STATE["count"] += 1
    return _orig_attn_forward(self, q, k, v, attn_mode)


T.JointSingleSeq.forward = _seq_forward
T.Attention.forward = _attn_forward


def demo_request(ds, task, i0, t):
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
    canvas = np.hstack([cv2.resize(frame(hist[-1], "scene"), (256, 256)), cv2.resize(frame(hist[-1], "wrist"), (256, 256))])
    return req, canvas


def attention_grid(layout, acc):
    """Attention mass on the latest video-conditioning frame as a (rows, cols) grid."""
    seg, _ = _segments(layout)
    c0, c1 = seg["video_cond"]
    mass = acc[c0:c1].cpu().numpy()
    ids = layout["ids"]["video_cond"].numpy()
    # ids columns: find time / row / column axes by their value ranges
    spans = [len(np.unique(ids[:, j])) for j in range(ids.shape[1])]
    tcol = int(np.argmax([s if 1 < s < 8 else 0 for s in spans])) if any(1 < s < 8 for s in spans) else None
    big = [j for j in range(ids.shape[1]) if spans[j] >= 8]
    rcol, ccol = big[0], big[1]
    keep = ids[:, tcol] == ids[:, tcol].max() if tcol is not None else np.ones(len(ids), bool)
    rows, cols = ids[keep, rcol], ids[keep, ccol]
    grid = np.zeros((rows.max() - rows.min() + 1, cols.max() - cols.min() + 1))
    np.add.at(grid, (rows - rows.min(), cols - cols.min()), mass[keep])
    return grid, {"id_spans": spans, "time_col": tcol, "row_col": rcol, "col_col": ccol}


def overlay(canvas, grid):
    heat = cv2.resize(grid / grid.max(), (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    out = (0.55 * canvas[..., ::-1] + 0.45 * heat).astype(np.uint8)
    cv2.line(out, (256, 0), (256, 255), (255, 255, 255), 1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True)
    p.add_argument("--peft", action="store_true")
    p.add_argument("--name", default="model")
    p.add_argument("--root", default="/workspace/hf/lerobot/ArnavG07/korora_spoon_feed")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = LeRobotDataset("ArnavG07/korora_spoon_feed", root=args.root)
    ep = ds.meta.episodes.to_pandas() if hasattr(ds.meta.episodes, "to_pandas") else pd.DataFrame(ds.meta.episodes)
    moments = [  # (label, task, episode position in that task, seconds into the episode)
        ("scoop_start", "Scoop food from the bowl", 30, 0.0),
        ("scoop_turn_to_bowl", "Scoop food from the bowl", 30, 1.8),
        ("scoop_in_food", "Scoop food from the bowl", 30, 3.6),
        ("to_mouth", "Bring the spoon to the mouth", 20, 2.0),
        ("tip", "Tip the food into the mouth", 10, 2.0),
        ("back_to_rest", "Move the spoon back to the rest position", 10, 1.5),
    ]
    flux = Flux(args.policy, args.peft, fixed_seed=True)
    summary = {}
    for label, task, k, sec in moments:
        e = ep[ep.tasks.str[0] == task].iloc[k]
        req, canvas = demo_request(ds, task, int(e.dataset_from_index), min(int(sec * 30), int(e.length) - 1))
        STATE.update(on=True, acc=None, count=0, layout=None)
        flux.act(req)
        STATE["on"] = False
        layout, acc = STATE["layout"], STATE["acc"] / STATE["count"]
        seg, _ = _segments(layout)
        split = {name: float(acc[a:b].sum()) for name, (a, b) in seg.items()}
        grid, axes = attention_grid(layout, acc)
        half = grid.shape[1] // 2
        on_cond = split["video_cond"]
        split["scene_camera"] = on_cond * float(grid[:, :half].sum() / grid.sum())
        split["wrist_camera"] = on_cond * float(grid[:, half:].sum() / grid.sum())
        cv2.imwrite(str(out / f"{args.name}_{label}.png"), overlay(canvas, grid))
        np.save(out / f"{args.name}_{label}_grid.npy", grid)
        summary[label] = {"task": task, "episode": int(e.episode_index), "seconds": sec, "split": split,
                          "blocks_x_steps": STATE["count"], "layout": {m: layout["lens"][m] for m in layout["order"]},
                          "text_tokens": layout["txt"], "id_axes": axes}
        print(label, {n: round(v, 3) for n, v in split.items()}, flush=True)
    json.dump(summary, open(out / f"{args.name}_summary.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
