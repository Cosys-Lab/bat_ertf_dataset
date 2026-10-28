"""Build the pinna / noseleaf rigs of the dataset meshes (batertf/rig.py, docs/rigging.md).

Per species: results/rig/<key>_rig.json (joints, bones, axes, segmentation, landmark weights),
results/rig/<key>_weights.npz (skinning weights, appendage weights, chain coordinates), results/rig/<key>_rig.png
(mesh coloured by bone weight, skeleton), and a skinned GLB for Blender in <work>/rig/<key>.glb (millimetres).
Usage: python scripts/build_rigs.py [key ...]"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import paths, rig, rigplot


def figure(r, fn):
    cols = rigplot.weight_colours(r)
    nasal = "noseleaf" in {b.split(".")[0] for b in r.bones}
    views = ["front", "left side", "top", "front-left 40 deg"]
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.4))
    for ax, v in zip(axs, views):
        R = rigplot.VIEWS[v]
        rigplot.render(ax, r.rest, R, cols)
        rigplot.draw_skeleton(ax, r, R)
        for k, a in r.meta["landmarks"].items():
            if k == "nose_centre":
                continue
            q = np.array(a["rest"]) @ R.T * 1e3
            ax.plot(q[0], q[1], "+", color="r", ms=7, mew=1.5, zorder=12)
        rigplot.frame_axes(ax, [r.rest], R)
        ax.set_title(v, fontsize=10)
        ax.tick_params(labelsize=7)
    ax = axs[3]                                           # zoom: noseleaf (nasal) or left ear (oral)
    R = rigplot.VIEWS["front-left 40 deg"]
    if nasal:
        c = np.mean([np.array(r.joints[b]["position"]) for b in r.bones if b.startswith("noseleaf")], 0)
        hw = 1.3 * max(r.joints[b]["length_mm"] for b in r.bones if b.startswith("noseleaf")) + 2
        ax.set_title("front-left 40 deg, noseleaf", fontsize=10)
    else:
        c = np.array(r.joints["pinna_left.base"]["position"])
        hw = 6.0
        ax.set_title("front-left 40 deg, left ear base", fontsize=10)
    q = c @ R.T * 1e3
    ax.set_xlim(q[0] - hw, q[0] + hw)
    ax.set_ylim(q[1] - hw, q[1] + hw)
    handles = [plt.Line2D([], [], color=c, lw=6, label=b) for b, c in zip(r.bones, rigplot.bone_colours(r))]
    handles += [plt.Line2D([], [], color="r", marker="+", ls="", label="landmarks (ears, nostrils / mouth)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    seg = "; ".join(f"{k}: {v.split(';')[0]}" for k, v in r.meta["segmentation"].items())
    fig.suptitle(f"{r.meta['species']} ({r.key}): rig, colour = skinning weight per bone (grey = fixed head), mm\n"
                 f"{seg}", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(fn, dpi=80)
    plt.close(fig)


def main():
    keys = sys.argv[1:] or list(paths.species())
    out = paths.work_dir() / "rig"
    out.mkdir(parents=True, exist_ok=True)
    for key in keys:
        meta, arrays, m = rig.build_rig(key)
        rig.save_rig(meta, arrays)
        r = rig.load_rig(key)
        figure(r, rig.RIG_DIR / f"{key}_rig.png")
        rig.export_gltf(r, out / f"{key}.glb")
        lw = {k: round(v["max_bone_weight"], 3) for k, v in meta["landmarks"].items()}
        print(f"{key}: bones {len(meta['bones'])}, " + ", ".join(
            f"{n} {a['n_vertices_moving']} v moving, chain {a['chain_length_mm']:.1f} mm" for n, a in meta["appendages"].items())
            + f"; landmark bone weights {lw}", flush=True)


if __name__ == "__main__":
    main()
