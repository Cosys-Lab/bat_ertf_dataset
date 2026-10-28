"""Write the best-matching receiver and nostril positions of a placement study into config/species.toml as the
species' landmarks ([species.<key>.landmarks], CT frame, metres).
Usage: python scripts/set_landmarks_from_placement.py key"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import paths

key = sys.argv[1]
d = np.load(paths.work_dir() / "placement" / f"{key}.npz")
names, pos = list(d["names"]), d["pos"]
r = json.loads((paths.RESULTS / "placement" / f"{key}.json").read_text())["refit"]
p = lambda n: "[" + ", ".join(f"{x:.6e}" for x in pos[names.index(n)]) + "]"
a, b, _ = r["emitter_best_pair"]
block = (f"\n[species.{key}.landmarks]   # best match to the reference ERTFs (scripts/placement_study.py; corr L/R/E "
         f"{r['left_corr_best_median_p10'][0]:.2f}/{r['right_corr_best_median_p10'][0]:.2f}/{r['emitter_corr_best_median'][0]:.2f})\n"
         f"ear_left = {p(r['best_left'])}\near_right = {p(r['best_right'])}\nnostrils = [{p(a)}, {p(b)}]\n")
f = paths.ROOT / "config" / "species.toml"
s = f.read_text()
hdr = f"[species.{key}.frames.ct]"
assert f"[species.{key}.landmarks]" not in s, "landmarks already set"
s = s.replace(hdr, block.lstrip("\n") + "\n" + hdr)
f.write_text(s)
print(block)
