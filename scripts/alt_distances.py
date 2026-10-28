"""Distance of every alternative source position (<name>_alt<i> of the production files) from the nominal landmark
(referee item M7). The alternatives are reproduced with run_ertf.ensemble (read-only: it reads
results/placement/<key>_rule.json and the configured landmarks) and checked against the production HDF5 files: the
source patch rebuilt from each position must equal the stored patch_faces (all production frames are the CT frame).
Ears: distance between the alternative and the nominal receiver. Nostril pairs: distance between the pair midpoints,
and the mean per-nostril distance (nostrils matched so that this mean is smallest).
Usage: python scripts/alt_distances.py  -> results/validation/alt_distances.json"""
import json
import os
import sys
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batertf import mesh, paths, project, sources
from run_ertf import ensemble

ORDER = ["pdisc", "mmicr", "mmacr", "rrouxi", "rform", "efusc", "mdaub"]
OUT = paths.RESULTS / "validation" / "alt_distances.json"


def main():
    sp = paths.species()
    res = {}
    for key in ORDER:
        lm = sp[key]["landmarks"]
        assert sp[key]["frames"]["ct"]["rotation"] == [0.0, 0.0, 0.0]
        ens = ensemble(key, lm)
        m = mesh.load(sp[key]["stl"])
        r = {}
        with h5py.File(paths.work_dir() / "ertf" / f"{key}_ct.h5", "r") as h:
            for name, alts in ens.items():
                nom = np.atleast_2d(lm["nostrils"] if name == "emitter" else lm[f"ear_{name}"])
                rad = project.NOSTRIL_RADIUS if name == "emitter" else project.EAR_RADIUS
                rows = []
                for i, a in enumerate(alts):
                    a = np.atleast_2d(a)
                    g = f"{name}_alt{i + 1}"
                    faces = np.sort(np.concatenate([sources.disc_patch(m, p, rad) for p in a]))
                    same = g in h and np.array_equal(np.sort(h[g]["patch_faces"][:]), faces)
                    row = {"source": g, "in_production_file": g in h, "patch_matches_file": bool(same),
                           "position_m": a.tolist(),
                           "distance_mm": float(np.linalg.norm(a.mean(0) - nom.mean(0)) * 1e3)}
                    if name == "emitter":
                        d = min(np.linalg.norm(a - nom[[0, 1]], axis=1).mean(), np.linalg.norm(a - nom[[1, 0]], axis=1).mean())
                        row["midpoint_distance_mm"] = row.pop("distance_mm")
                        row["mean_per_nostril_distance_mm"] = float(d * 1e3)
                        row["pair_spacing_mm"] = float(np.linalg.norm(a[0] - a[1]) * 1e3)
                    rows.append(row)
                extra = sorted(n for n in h if n.startswith(name + "_alt") and n not in [x["source"] for x in rows])
                assert not extra, f"{key}: production file has {extra} not reproduced by ensemble()"
                r[name] = rows
        res[key] = r
        print(key, {n: [round(x.get("distance_mm", x.get("midpoint_distance_mm")), 2) for x in v] for n, v in r.items()},
              "nostril mean:", [round(x["mean_per_nostril_distance_mm"], 2) for x in r.get("emitter", [])],
              "all patches match:", all(x["patch_matches_file"] for v in r.values() for x in v), flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(OUT)


if __name__ == "__main__":
    main()
