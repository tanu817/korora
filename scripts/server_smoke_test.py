"""Smoke test for feeder/flux_server.py (run on the pod): one real demo frame in, 32 commands out."""
import json
import time
import urllib.request

import numpy as np

from feeder.flux_remote import encode_jpeg
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("ArnavG07/korora_spoon_feed", root="/workspace/hf/lerobot/ArnavG07/korora_spoon_feed")
s = ds[0]
state = s["observation.state"].numpy()


def img(k):
    return (s[k].permute(1, 2, 0).numpy() * 255).astype(np.uint8)


req = {
    "task": "Scoop food from the bowl",
    "state": [state.tolist()] * 8,
    "command": [state.tolist()] * 8,
    "scene": [encode_jpeg(img("observation.images.scene"), 90)] * 2,
    "wrist": [encode_jpeg(img("observation.images.wrist"), 90)] * 2,
}
for i in range(2):
    t = time.time()
    http = urllib.request.Request(
        "http://localhost:8000/act", data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        r = json.load(urllib.request.urlopen(http, timeout=120))
    except urllib.error.HTTPError as e:
        print("ERROR", json.load(e))
        raise SystemExit(1)
    a = np.array(r["actions"])
    print(f"call {i}: {time.time() - t:.2f}s (gpu {r['infer_s']:.2f}s) shape {a.shape}")
print("start state   ", np.round(state, 1))
print("action 1      ", np.round(a[0], 1))
print("action 32     ", np.round(a[-1], 1))
print("max step/tick ", np.round(np.abs(np.diff(np.vstack([state, a]), axis=0)).max(0), 1))
print("demo @ tick 32", np.round(ds[31]["action"].numpy(), 1))
