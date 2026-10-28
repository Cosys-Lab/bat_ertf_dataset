"""Learn a transferable placement rule from the calibrated species (Jan: get close to the pdisc and mmicr references,
learn from this, and know how to adjust the other models).
For every ear candidate of a placement study: acoustic descriptors (mean gain re free field over the frontal
hemisphere and frequency, peak gain, directivity index, gain toward the front) and geometric ones (occlusion), next to
its pattern correlation with the reference. Prints the Spearman rank correlation of each descriptor with the match and
where the descriptor's maximum ranks in the match.
Usage: python scripts/learn_placement_rule.py key [key ...]"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import grid, mesh, paths

AZ, EL = grid.GRIDS["sphere"]


def descriptors(H, R=None):
    """H: [el, az, f] complex on the full sphere (CT frame). Directions in the frontal hemisphere of frame R."""
    u = grid.unit_vectors(AZ, EL)
    fwd = (u @ (R[:, 0] if R is not None else np.array([1.0, 0, 0]))) > 0
    w = np.cos(np.deg2rad(EL))[:, None] * np.ones(len(AZ))
    P = np.abs(H) ** 2
    mean_all = (P * w[..., None]).sum((0, 1)) / w.sum()
    mean_front = (P * (w * fwd)[..., None]).sum((0, 1)) / (w * fwd).sum()
    return {"gain_front_db": 10 * np.log10(mean_front).mean(),
            "gain_peak_db": 10 * np.log10(P.max((0, 1))).mean(),
            "directivity_db": 10 * np.log10(P.max((0, 1)) / mean_all).mean(),
            "front_back_db": 10 * np.log10(mean_front / ((P * (w * ~fwd)[..., None]).sum((0, 1)) / (w * ~fwd).sum())).mean()}


def main():
    for key in sys.argv[1:]:
        d = np.load(paths.work_dir() / "placement" / f"{key}.npz")
        names, pos, H = list(d["names"]), d["pos"], d["H"]
        res = json.loads((paths.RESULTS / "placement" / f"{key}.json").read_text())["refit"]
        m = mesh.load(paths.species()[key]["stl"])
        ao = np.load(paths.work_dir() / "occlusion" / f"{key}.npz")["ao"]
        tree = cKDTree(m.vertices)
        for s in ("left", "right"):
            ec = res["ear_corr"][s]
            cand = list(ec)
            match = np.array([ec[n] for n in cand])
            desc = {n: descriptors(H[names.index(n)]) for n in cand}
            desc_ao = {n: ao[tree.query(pos[names.index(n)])[1]] for n in cand}
            order = np.argsort(-match)
            print(f"{key} {s}: {len(cand)} candidates, best match {match.max():.3f}")
            for k in list(desc[cand[0]]) + ["occlusion"]:
                v = np.array([desc_ao[n] if k == "occlusion" else desc[n][k] for n in cand])
                rho = spearmanr(v, match).correlation
                top = int(np.argmax(v))
                print(f"   {k:16s} spearman {rho:+.2f}; its maximum has match {match[top]:.3f} "
                      f"(rank {int(np.where(order == top)[0][0]) + 1}/{len(cand)}), "
                      f"{np.linalg.norm(pos[names.index(cand[top])] - pos[names.index(cand[order[0]])])*1e3:.1f} mm from the best")


if __name__ == "__main__":
    main()
