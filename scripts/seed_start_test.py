"""Fixed vs random sampling seed from a still start (issue 2 in docs/flux_findings.md). Run on the pod with the server up."""
"""Does FLUX leave the waiting phase from a still start? Query the first frame of scoop demos repeatedly."""
import json
import urllib.request

import numpy as np
import pandas as pd

from feeder.flux_remote import encode_jpeg
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("ArnavG07/korora_spoon_feed", root="/workspace/hf/lerobot/ArnavG07/korora_spoon_feed")
ep = ds.meta.episodes.to_pandas() if hasattr(ds.meta.episodes, "to_pandas") else pd.DataFrame(ds.meta.episodes)
sc = ep[ep.tasks.str[0] == "Scoop food from the bowl"]


def img(i, k):
    return (ds[i][f"observation.images.{k}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)


for k in (15, 30):
    i = int(sc.iloc[k].dataset_from_index)
    state = ds[i]["observation.state"].numpy()
    cmd = state.copy()
    cmd[:5] += np.array([0.1, -2.0, -0.4, -0.35, -0.1])
    cmd[5] = 0.0
    req = {"task": "Scoop food from the bowl", "state": [state.tolist()] * 8, "command": [cmd.tolist()] * 8,
           "scene": [encode_jpeg(img(i, "scene"), 90)] * 2, "wrist": [encode_jpeg(img(i, "wrist"), 90)] * 2}
    moves = []
    for _ in range(10):
        r = urllib.request.Request("http://localhost:8000/act", data=json.dumps(req).encode(),
                                   headers={"Content-Type": "application/json"})
        a = np.array(json.load(urllib.request.urlopen(r, timeout=120))["actions"])
        moves.append(np.abs(a[-1, :5] - cmd[:5]).max())
    print(f"demo {k}: biggest joint change within the 32-step plan, 10 random seeds: {np.round(moves, 1)}")
