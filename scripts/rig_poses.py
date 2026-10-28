"""Visual check of rig deformations: rest pose and a few poses per species (both pinnae symmetric, noseleaf).
Usage: python scripts/rig_poses.py [key ...]   (default mmicr pdisc)  -> results/rig/poses.png"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import rig, rigplot

POSES = {"rest": {},
         "pinnae base flex +20": {"pinna_both.base": {"flex": 20}},
         "pinnae base flex -20": {"pinna_both.base": {"flex": -20}},
         "pinnae base abduct +15": {"pinna_both.base": {"abduct": 15}},
         "pinnae tip bent: distal flex +30": {"pinna_both.distal": {"flex": 30}},
         "noseleaf base flex +15": {"noseleaf.base": {"flex": 15}},
         "noseleaf base flex -15": {"noseleaf.base": {"flex": -15}}}


def main():
    keys = sys.argv[1:] or ["mmicr", "pdisc"]
    views = ["front-left 40 deg", "left side"]
    fig, axs = plt.subplots(len(keys) * len(views), len(POSES), figsize=(3.2 * len(POSES), 3.0 * len(keys) * len(views)))
    for i, key in enumerate(keys):
        r = rig.load_rig(key)
        cols = rigplot.weight_colours(r)
        ms = {n: r.deform(p) for n, p in POSES.items()}
        for j, (n, p) in enumerate(POSES.items()):
            c = r.check(ms[n], p)
            for v, view in enumerate(views):
                ax = axs[i * len(views) + v, j]
                R = rigplot.VIEWS[view]
                if n != "rest":                               # rest outline for reference
                    w = r.rest.vertices @ R.T * 1e3
                    ax.scatter(w[::7, 0], w[::7, 1], s=0.05, c="0.85", zorder=0)
                rigplot.render(ax, ms[n], R, cols)
                rigplot.draw_skeleton(ax, r, R, p)
                rigplot.frame_axes(ax, list(ms.values()), R)
                ax.set_xticks([])
                ax.set_yticks([])
                if v == 0:
                    ok = "valid" if c["valid"] else (f"INVALID: {c['self_intersecting_faces']} self-int., "
                                                     f"{c.get('inverted_faces', 0)} inverted")
                    ax.set_title(f"{key}: {n}\n{ok}", fontsize=8, color="k" if c["valid"] else "r")
                if j == 0:
                    ax.set_ylabel(view, fontsize=8)
    fig.suptitle("Rig poses (angles in degrees; flex + = backward / up, abduct + = outward; grey dots = rest pose)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(rig.RIG_DIR / "poses.png", dpi=80)


if __name__ == "__main__":
    main()
