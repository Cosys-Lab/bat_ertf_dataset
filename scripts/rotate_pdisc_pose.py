"""One-off (2026-10-29): put the P. discolor head in the pose of De Mey et al. 2008, in which its ERTFs are usually
shown, so that every species has exactly one frame. The pose is the rotation fitted to their reference ERTFs
([species.pdisc.frames.reference] matrix M, u_mesh = M u_frame, so v_new = v_ct @ M), about 33 deg from the CT scan.

Rotates: the dataset mesh, the configured landmarks and nostril search centre, the positions in
results/placement/pdisc_rule.json (the placement ensemble), and the cached convergence remeshes; sets the reference
frame to the identity and records the rotation in meshes.json and in scripts/remesh_ct.py (POSE). Not rotated
(historical outputs, internally consistent in the CT scan frame): <work>/placement/pdisc.npz, results/placement/pdisc.json,
<work>/occlusion/pdisc.npz (per-vertex values, valid in any frame). Refuses to run twice (meshes.json marker).
Afterwards: build_rigs.py pdisc, rig_ranges.py pdisc, run_ertf.py pdisc ct (full band).
Usage: python scripts/rotate_pdisc_pose.py"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import paths

KEY = "pdisc"


def rot_tree(x, M):
    """Rotate every 3-vector (list of 3 numbers) in a nested JSON structure."""
    if isinstance(x, list) and len(x) == 3 and all(isinstance(v, (int, float)) for v in x):
        return (np.asarray(x, float) @ M).tolist()
    if isinstance(x, list):
        return [rot_tree(v, M) for v in x]
    if isinstance(x, dict):
        return {k: rot_tree(v, M) for k, v in x.items()}
    return x


def main():
    sp = paths.species()[KEY]
    M = np.array(sp["frames"]["reference"]["matrix"], float)
    mj = paths.ROOT / "data" / "meshes" / "dataset" / "meshes.json"
    meta = json.loads(mj.read_text())
    if "pose_rotation" in meta[KEY]:
        sys.exit("already rotated")
    print(f"rotation {np.degrees(np.arccos((np.trace(M) - 1) / 2)):.1f} deg")

    fn = paths.ROOT / "data" / "meshes" / "dataset" / meta[KEY]["file"]
    m = trimesh.load(fn, process=False)
    m.vertices = m.vertices @ M
    m.export(fn)
    meta[KEY]["pose_rotation"] = {"matrix_v_new_eq_v_ct_times_M": M.tolist(),
                                  "note": "head in the pose of De Mey et al. 2008 (scripts/rotate_pdisc_pose.py)"}
    meta[KEY]["frame"] = meta[KEY]["frame"] + "; rotated into the pose of De Mey et al. 2008 (pose_rotation)"
    mj.write_text(json.dumps(meta, indent=1))

    for f in (paths.work_dir() / "convergence").glob(f"{KEY}_L*.stl"):
        c = trimesh.load(f, process=False)
        c.vertices = c.vertices @ M
        c.export(f)

    rj = paths.RESULTS / "placement" / f"{KEY}_rule.json"
    r = json.loads(rj.read_text())
    for k in ("left", "right", "nostrils"):
        if k in r:
            r[k] = rot_tree(r[k], M)
    r["pose_rotation"] = "positions rotated into the pose of De Mey et al. 2008 (scripts/rotate_pdisc_pose.py)"
    rj.write_text(json.dumps(r, indent=1))

    cfg = paths.ROOT / "config" / "species.toml"
    s = cfg.read_text()
    i = s.index(f"[species.{KEY}]")
    j = s.index("[species.mmicr]", i)                  # all pdisc sections (header, landmarks, frames)
    blk = s[i:j]
    fmt = lambda p: "[" + ", ".join(f"{x:.6e}" for x in p) + "]"
    lm = sp["landmarks"]
    for k in ("ear_left", "ear_right"):
        blk = re.sub(rf"^{k} = .*$", f"{k} = {fmt(np.asarray(lm[k]) @ M)}", blk, flags=re.M)
    n = np.asarray(lm["nostrils"]) @ M
    blk = re.sub(r"^nostrils = .*$", f"nostrils = [{fmt(n[0])}, {fmt(n[1])}]", blk, flags=re.M)
    blk = re.sub(r"^nose_centre = (\[[^\]]*\])", lambda mm: f"nose_centre = {fmt(np.asarray(json.loads(mm.group(1))) @ M)}",
                 blk, flags=re.M)
    blk = re.sub(r"(\[species\.pdisc\.frames\.reference\][^\n]*\n)matrix = .*\n",
                 "[species.pdisc.frames.reference]   # = the dataset frame: the mesh is in the pose of De Mey et al. "
                 "2008 (scripts/rotate_pdisc_pose.py)\nrotation = [0.0, 0.0, 0.0]\n", blk)
    blk = blk.replace(f"[species.{KEY}]\n", f"[species.{KEY}]   # head in the pose of De Mey et al. 2008, "
                      "about 33 deg from the CT scan (scripts/rotate_pdisc_pose.py)\n", 1)
    cfg.write_text(s[:i] + blk + s[j:])
    print("done: rebuild the rig, the ranges and the production file of pdisc")


if __name__ == "__main__":
    main()
