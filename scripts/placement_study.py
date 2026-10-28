"""Receiver and emitter placement study against the reference ERTFs (Jan: "find various placements of receive elements
and see if match improves").

On the dataset mesh (CT frame, = frame of the original lab simulations), dense candidate source patches:
- ears: surface points within EAR_REGION of the most occluded spot of each pinna base, at least SPACING apart,
  patch radius EAR_PATCH;
- nose: forward-facing points within NOSE_REGION of the nose centre, SPACING apart, patch radius NOSE_PATCH.
All candidates are right-hand sides of one GPU solve per frequency (N_FREQ frequencies across the reference band),
evaluated on the full sphere. Pattern correlation with the reference (frontal hemisphere, dB clipped at -30) is computed
(a) in the start frame (CT frame, or [frames.reference] matrix if the reference used another orientation), and
(b) after a small orientation refit around it (yaw/pitch/roll within +-MAX_ROT deg); label "ct_frame" = start frame.
Usage: python scripts/placement_study.py key   -> <work>/placement/<key>.npz, results/placement/<key>.{json,png}"""
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batertf import gpubem, grid, mesh, paths, reference, rotate, sources
from ear_landmarks import ear_candidates
from occlusion_maps import draw

EAR_REGION, NOSE_REGION, SPACING = 4.0e-3, 3.0e-3, 0.5e-3
EAR_PATCH, NOSE_PATCH = 0.4e-3, 0.3e-3
N_FREQ = 10
MAX_ROT = 15.0
BATCH = 40


def sample(m, centre, radius, spacing, mask=None):
    v = m.vertices
    ok = np.linalg.norm(v - centre, axis=1) < radius
    if mask is not None:
        ok &= mask
    idx = np.where(ok)[0]
    out = [idx[np.argmin(np.linalg.norm(v[idx] - centre, axis=1))]]
    d = np.linalg.norm(v[idx] - v[out[0]], axis=1)
    while d.max() >= spacing:
        j = idx[np.argmax(d)]
        out.append(j)
        d = np.minimum(d, np.linalg.norm(v[idx] - v[j], axis=1))
    return v[out]


def corr_mean(sim, ref):
    return reference.metrics(sim, ref)["corr"].mean()


def main():
    key = sys.argv[1]
    sp = paths.species()[key]
    m = mesh.load(sp["stl"])
    ao = np.load(paths.work_dir() / "occlusion" / f"{key}.npz")["ao"]
    ref_key = sp.get("reference_key", key)
    ref = reference.load(ref_key)
    rf = ref["freqs"]
    freqs = rf[np.linspace(0, len(rf) - 1, N_FREQ + 2).round().astype(int)[1:-1]]
    c = ear_candidates(m, ao)
    cand = {}
    for side, s in (("L", "left"), ("R", "right")):
        for k, p in enumerate(sample(m, c[side][0], EAR_REGION, SPACING)):
            cand[f"ear{side}{k:03d}"] = (p, EAR_PATCH)
    fwd = m.vertex_normals[:, 0] > 0.2
    for k, p in enumerate(sample(m, np.array(sp["nose_centre"]), NOSE_REGION, SPACING, fwd)):
        cand[f"nose{k:03d}"] = (p, NOSE_PATCH)
    patches = {n: [sources.disc_patch(m, p, r)] for n, (p, r) in cand.items()}
    print(f"{key}: {len(patches)} candidates ({sum(n.startswith('earL') for n in cand)} left, "
          f"{sum(n.startswith('earR') for n in cand)} right, {sum(n.startswith('nose') for n in cand)} nose), "
          f"{len(freqs)} frequencies", flush=True)
    pts, _ = grid.unique_points("sphere")
    out = paths.work_dir() / "placement"
    out.mkdir(exist_ok=True)
    H = {}
    names_all = list(patches)
    for b0 in range(0, len(names_all), BATCH):          # sources in batches: the Krylov basis grows with their number
        batch = {n: patches[n] for n in names_all[b0:b0 + BATCH]}
        h = gpubem.solve(m, batch, freqs, pts, log=lambda s: print(s, flush=True))
        for n in h:
            H[n] = gpubem.to_grid(h[n], "sphere").astype(np.complex64)
    area = {n: float(m.area_faces[patches[n][0]].sum()) for n in patches}
    names = list(H)
    np.savez(out / f"{key}.npz", freqs=freqs, names=np.array(names), H=np.stack([H[n] for n in names]),
             pos=np.array([cand[n][0] for n in names]), area=np.array([area[n] for n in names]))
    ki = [int(np.argmin(abs(rf - f))) for f in freqs]
    R0 = {s: ref[s][..., ki] for s in ("left", "right", "emitter")}

    def hemi(h, R):
        return rotate.hemisphere_from_sphere(np.abs(h), R)

    fr = sp.get("frames", {}).get("reference", {})
    R0m = np.array(fr["matrix"]) if "matrix" in fr else np.eye(3)    # start: the reference frame if known, else CT
    res = {"freqs": freqs.tolist(), "start_matrix": R0m.tolist()}
    for label, angles in (("ct_frame", (0.0, 0.0, 0.0)), ("refit", None)):
        if angles is None:                             # small orientation refit on the best ears of the CT frame
            bl, br = res["ct_frame"]["best_left"], res["ct_frame"]["best_right"]
            f = lambda a: -(corr_mean(hemi(H[bl], R0m @ rotate.matrix(*a)), R0["left"]) +
                            corr_mean(hemi(H[br], R0m @ rotate.matrix(*a)), R0["right"])) / 2 if np.all(np.abs(a) <= MAX_ROT) else 1.0
            angles = minimize(f, np.zeros(3), method="Nelder-Mead", options={"initial_simplex": np.array(
                [[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]]), "xatol": 0.3, "fatol": 1e-4}).x
        R = R0m @ rotate.matrix(*angles)
        ears = {s: {n: corr_mean(hemi(H[n], R), R0[s]) for n in names if n.startswith("ear" + s[0].upper())}
                for s in ("left", "right")}
        nose = [n for n in names if n.startswith("nose")]
        hn = {n: rotate.hemisphere_from_sphere(H[n].real, R) + 1j * rotate.hemisphere_from_sphere(H[n].imag, R) for n in nose}
        pairs = []
        for a, b in itertools.combinations(nose, 2):
            d = np.linalg.norm(cand[a][0] - cand[b][0])
            if 0.8e-3 <= d <= 4e-3:
                pairs.append((corr_mean(np.abs(hn[a] * area[a] + hn[b] * area[b]), R0["emitter"]), a, b, d))
        pairs.sort(reverse=True)
        r = {"angles": list(map(float, angles))}
        for s in ("left", "right"):
            v = np.array(list(ears[s].values()))
            r[f"best_{s}"] = max(ears[s], key=ears[s].get)
            r[f"{s}_corr_best_median_p10"] = [float(v.max()), float(np.median(v)), float(np.percentile(v, 10))]
            r[f"{s}_corr_rule"] = float(ears[s][f"ear{s[0].upper()}000"])     # the occlusion rule's own point
        r["emitter_best_pair"] = [pairs[0][1], pairs[0][2], float(pairs[0][3])]
        r["emitter_corr_best_median"] = [float(pairs[0][0]), float(np.median([p[0] for p in pairs]))]
        r["ear_corr"] = {s: {n: float(v) for n, v in ears[s].items()} for s in ears}
        res[label] = r
        print(f"{key} [{label}] angles {np.round(angles, 1)}: left best/median/p10 {np.round(r['left_corr_best_median_p10'], 3)} "
              f"(rule {r['left_corr_rule']:.3f}); right {np.round(r['right_corr_best_median_p10'], 3)} (rule {r['right_corr_rule']:.3f}); "
              f"emitter best/median {np.round(r['emitter_corr_best_median'], 3)}, best pair spacing {pairs[0][3]*1e3:.2f} mm", flush=True)
    d = paths.RESULTS / "placement"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{key}.json").write_text(json.dumps({k: v for k, v in res.items()}, indent=1, default=str))
    # maps: candidate positions coloured by correlation (CT frame), side views of each ear
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for ax, s, R in ((axs[0], "left", [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]), (axs[1], "right", [[1, 0, 0], [0, 0, 1], [0, -1, 0]])):
        draw(ax, m, np.array(R, float), np.zeros(len(m.vertices)) + 0.15, f"{s} ear: candidate receivers")
        e = res["ct_frame"]["ear_corr"][s]
        p = np.array([cand[n][0] for n in e]) @ np.array(R, float).T * 1e3
        sc = ax.scatter(p[:, 0], p[:, 1], c=list(e.values()), cmap="viridis", s=18, vmin=0.4, vmax=0.95, zorder=5)
        b = cand[res["ct_frame"][f"best_{s}"]][0] @ np.array(R, float).T * 1e3
        ax.plot(b[0], b[1], "r*", ms=14, zorder=6)
    fig.colorbar(sc, ax=axs, label="Pattern Correlation with Reference (mean over frequency)")
    fig.suptitle(f"{sp['name']}: Receiver Placement vs Agreement with the Reference (CT frame)")
    fig.savefig(d / f"{key}.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
