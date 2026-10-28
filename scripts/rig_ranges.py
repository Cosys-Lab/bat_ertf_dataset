"""Valid pose range of every rig joint and axis (batertf/rig.py, docs/rigging.md).

For each joint and axis (flex, abduct, twist) alone, the angle is increased in 5 deg steps (up to 60 deg) in both
directions until the deformed mesh fails Rig.check (self-intersection, inverted face, edge stretch outside
rig.STRETCH_LIMITS); the last valid angle is reported, together with the reason of the first failure.
Usage: python scripts/rig_ranges.py [key ...]  -> results/rig/valid_ranges.json (merged per species)"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import paths, rig

STEP, MAX = 5.0, 60.0


def reason(c):
    r = []
    if c.get("self_intersecting_faces"):
        r.append(f"{c['self_intersecting_faces']} self-intersecting faces")
    if c.get("inverted_faces"):
        r.append(f"{c['inverted_faces']} inverted faces")
    lo, hi = c["edge_stretch_min_max"]
    if lo < rig.STRETCH_LIMITS[0] or hi > rig.STRETCH_LIMITS[1]:
        r.append(f"edge stretch {lo:.2f}-{hi:.2f}")
    return ", ".join(r)


def scan(r, joint, axis):
    out = {}
    for sgn in (1, -1):
        last, why, a = 0.0, "", STEP
        while a <= MAX + 1e-9:
            pose = {joint: {axis: sgn * a}}
            c = r.check(r.deform(pose), pose)
            if not c["valid"]:
                why = f"{sgn * a:+.0f} deg: {reason(c)}"
                break
            last = a
            a += STEP
        out["+" if sgn > 0 else "-"] = (sgn * last, why)
    return {"range_deg": [out["-"][0], out["+"][0]], "limit": [out["-"][1], out["+"][1]]}


def main():
    keys = sys.argv[1:] or list(paths.species())
    fn = rig.RIG_DIR / "valid_ranges.json"
    res = json.loads(fn.read_text()) if fn.exists() else {}
    for key in keys:
        r = rig.load_rig(key)
        t0 = time.time()
        res[key] = {"step_deg": STEP, "max_deg": MAX, "stretch_limits": rig.STRETCH_LIMITS, "joints": {}}
        for b in r.bones:
            res[key]["joints"][b] = {a: scan(r, b, a) for a in rig.AXES}
            print(key, b, {a: v["range_deg"] for a, v in res[key]["joints"][b].items()}, flush=True)
        print(f"{key}: {time.time() - t0:.0f} s", flush=True)
        fn.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
