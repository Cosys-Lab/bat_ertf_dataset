"""Place ears and nostrils of a species without reference ERTFs with the rule learned on pdisc and mmicr
(scripts/learn_placement_rule.py, docs/placement_rule.md):

- ears: candidates as in placement_study.py; score = z(front/back ratio) + z(occlusion), averaged over SMOOTH;
  the receiver is the maximum (on pdisc/mmicr this gives pattern correlations 0.77-0.86 with the references,
  occlusion alone 0.62-0.77);
- nostrils (nasal emitters): mirror-symmetric pairs (mismatch < 1 mm across the symmetry plane), 1.5-2.5 mm apart;
  the pair with the highest front/back ratio of the summed field (matches the reference-best pair on mmicr, 0.74 vs a
  0.81 best on pdisc).
Front/back ratio: mean power over the frontal half-space (X > 0 of the CT frame) over that of the rear, in dB,
averaged over frequency. The next ENSEMBLE-1 distinct positions (>= MIN_SEP apart) are kept as alternatives.
Usage: python scripts/apply_placement_rule.py key [--from-placement]
  -> results/placement/<key>_rule.json and config landmarks (printed).
  --from-placement: reuse the solved candidates of placement_study.py (calibrated species); writes only the json
  (alternatives for the ensemble), the calibrated landmarks in the config stay."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batertf import gpubem, grid, mesh, paths, sources, symmetry
from ear_landmarks import ear_candidates
from learn_placement_rule import descriptors
from placement_study import EAR_PATCH, EAR_REGION, NOSE_PATCH, SPACING, sample
from test_nostril_rule import mirror

N_FREQ = 10
NOSE_REGION = 4.0e-3
SMOOTH = 0.75e-3
ENSEMBLE, MIN_SEP = 5, 1.0e-3


def distinct_top(P, score, k, sep):
    out = []
    for i in np.argsort(-score):
        if all(np.linalg.norm(P[i] - P[j]) >= sep for j in out):
            out.append(int(i))
        if len(out) == k:
            break
    return out


def main():
    key = sys.argv[1]
    reuse = "--from-placement" in sys.argv
    sp = paths.species()[key]
    m = mesh.load(sp["stl"])
    ao = np.load(paths.work_dir() / "occlusion" / f"{key}.npz")["ao"]
    band = np.array(sp["band_khz"][:2]) * 1e3
    freqs = np.linspace(band[0], band[1], N_FREQ + 2)[1:-1]
    c = ear_candidates(m, ao)
    cand = {}
    for side in ("L", "R"):
        for k, p in enumerate(sample(m, c[side][0], EAR_REGION, SPACING)):
            cand[f"ear{side}{k:03d}"] = (p, EAR_PATCH)
    if sp["emitter"] == "nasal":
        fwd = m.vertex_normals[:, 0] > 0.2
        for k, p in enumerate(sample(m, np.array(sp["nose_centre"]), NOSE_REGION, SPACING, fwd)):
            cand[f"nose{k:03d}"] = (p, NOSE_PATCH)
    if reuse:
        d = np.load(paths.work_dir() / "placement" / f"{key}.npz")
        cand = {n: (p, EAR_PATCH if n.startswith("ear") else NOSE_PATCH) for n, p in zip(map(str, d["names"]), d["pos"])}
        H, freqs = {n: h for n, h in zip(cand, d["H"])}, d["freqs"]
    patches = {n: [sources.disc_patch(m, p, r)] for n, (p, r) in cand.items()}
    names = list(patches)
    print(f"{key}: {len(patches)} candidates, {len(freqs)} frequencies {np.round(freqs/1e3, 1)} kHz", flush=True)
    if not reuse:
        pts, _ = grid.unique_points("sphere")
        h = gpubem.solve_lu(m, patches, freqs, pts, log=lambda s: print(s, flush=True))
        H = {n: gpubem.to_grid(v, "sphere").astype(np.complex64) for n, v in h.items()}
    area = {n: float(m.area_faces[patches[n][0]].sum()) for n in patches}
    tree = cKDTree(m.vertices)
    z = lambda v: (v - v.mean()) / v.std()
    out = {"freqs": freqs.tolist(), "rule": "docs/placement_rule.md"}
    lm = {}
    for side, s in (("L", "left"), ("R", "right")):
        ns = [n for n in names if n.startswith("ear" + side)]
        P = np.array([cand[n][0] for n in ns])
        fb = np.array([descriptors(H[n])["front_back_db"] for n in ns])
        oc = np.array([ao[tree.query(p)[1]] for p in P])
        sc = z(fb) + z(oc)
        t = cKDTree(P)
        sc = np.array([sc[t.query_ball_point(p, SMOOTH)].mean() for p in P])
        top = distinct_top(P, sc, ENSEMBLE, MIN_SEP)
        lm[f"ear_{s}"] = P[top[0]].tolist()
        out[s] = {"nominal": P[top[0]].tolist(), "alternatives": P[top[1:]].tolist(), "score": sc[top].tolist(),
                  "front_back_db": fb[top].tolist(), "occlusion": oc[top].tolist()}
    if sp["emitter"] == "nasal":
        sym, _ = symmetry.fit(m)
        nose = [n for n in names if n.startswith("nose")]
        pairs = []
        for a, b in itertools.combinations(nose, 2):
            pa, pb = cand[a][0], cand[b][0]
            if 1.5e-3 <= np.linalg.norm(pa - pb) <= 2.5e-3 and np.linalg.norm(mirror(pa, *sym) - pb) < 1.0e-3:
                h = H[a] * area[a] + H[b] * area[b]
                pairs.append((descriptors(h)["front_back_db"], a, b))
        pairs.sort(reverse=True)
        chosen = []
        for fbv, a, b in pairs:                          # distinct alternatives: midpoints >= MIN_SEP apart
            mid = (cand[a][0] + cand[b][0]) / 2
            if all(np.linalg.norm(mid - mm) >= MIN_SEP for _, _, _, mm in chosen):
                chosen.append((fbv, a, b, mid))
            if len(chosen) == ENSEMBLE:
                break
        lm["nostrils"] = [cand[chosen[0][1]][0].tolist(), cand[chosen[0][2]][0].tolist()]
        out["nostrils"] = {"n_symmetric_pairs": len(pairs), "nominal": lm["nostrils"],
                           "alternatives": [[cand[a][0].tolist(), cand[b][0].tolist()] for _, a, b, _ in chosen[1:]],
                           "front_back_db": [c[0] for c in chosen]}
    d = paths.RESULTS / "placement"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{key}_rule.json").write_text(json.dumps(out, indent=1))
    if reuse:
        print(json.dumps({k: v["nominal"] if isinstance(v, dict) and "nominal" in v else None for k, v in out.items()}))
        return
    f = paths.ROOT / "config" / "species.toml"
    s = f.read_text()
    assert f"[species.{key}.landmarks]" not in s, "landmarks already set"
    fmt = lambda p: "[" + ", ".join(f"{x:.6e}" for x in p) + "]"
    block = f"[species.{key}.landmarks]   # placement rule learned on pdisc/mmicr (scripts/apply_placement_rule.py)\n"
    block += "".join(f"{k} = {fmt(v)}\n" for k, v in lm.items() if k != "nostrils")
    if "nostrils" in lm:
        block += f"nostrils = [{fmt(lm['nostrils'][0])}, {fmt(lm['nostrils'][1])}]\n"
    if "mouth" in sp:
        block += f"mouth = {fmt(sp['mouth'])}\n"
    s = s.replace(f"[species.{key}.frames.ct]", block + "\n" + f"[species.{key}.frames.ct]")
    f.write_text(s)
    print(block)


if __name__ == "__main__":
    main()
