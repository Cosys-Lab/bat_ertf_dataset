"""Demo of the rig: how much does the left-ear ERTF of M. microtis change when the left pinna is rotated about its base?

Poses: rest, left pinna base flex +20 deg (tilted backward / up) and -20 deg (forward / down). Each deformed mesh is
checked (Rig.check), the left-ear patch is built as in scripts/run_ertf.build (disc_patch at the deformed ear
landmark, radius project.EAR_RADIUS), and gpubem.solve_lu gives the left-ear transfer function on the full-sphere grid
at 40/80/120 kHz. The deformed patterns are compared with the rest pose (same metric as the reference comparison:
dB patterns normalised per frequency, -30 dB floor, cos(el)-weighted correlation, angle between peak directions).
Usage: python scripts/rig_demo_bem.py [key] [joint] [axis] [deg ...]
Output: results/rig/demo_<key>.json, results/rig/demo_<key>.png, fields in <work>/rig/demo_<key>.npz"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import gpubem, grid, paths, plotting, project, reference, rig, sources

FREQS = np.array([40e3, 80e3, 120e3])


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "mmicr"
    joint = sys.argv[2] if len(sys.argv) > 2 else "pinna_left.base"
    axis = sys.argv[3] if len(sys.argv) > 3 else "flex"
    degs = [float(x) for x in sys.argv[4:]] or [20.0, -20.0]
    r = rig.load_rig(key)
    pts, back = grid.unique_points("sphere")
    az, el = grid.GRIDS["sphere"]
    wcell = np.cos(np.deg2rad(el))[:, None] * np.ones(len(az))[None, :]
    wpt = np.bincount(back.ravel(), wcell.ravel(), minlength=len(pts))       # solid-angle weight per unique point
    u = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    front = u[:, 0] >= 0
    side = "left" if "left" in joint else "right"
    poses = {"rest": {}} | {f"{axis} {d:+.0f}": {joint: {axis: d}} for d in degs}
    H, info = {}, {}
    for name, pose in poses.items():
        m = r.deform(pose)
        chk = r.check(m, pose)
        lm = r.deform_landmarks(pose, m)
        ear = lm[f"ear_{side}"]
        patch = sources.disc_patch(m, ear, project.EAR_RADIUS)
        t0 = time.time()
        H[name] = gpubem.solve_lu(m, {side: [patch]}, FREQS, pts, log=print)[side]
        jp = r.joint_positions(pose)
        tip = jp[joint.split(".")[0] + ".end"]
        info[name] = {"pose": pose, "check": chk, "ear_landmark_mm": (ear * 1e3).round(4).tolist(),
                      "ear_landmark_shift_mm": float(np.linalg.norm(ear - r.deform_landmarks({}, r.rest)[f"ear_{side}"]) * 1e3),
                      "patch_faces": int(len(patch)), "patch_faces_same_as_rest": None,
                      "tip_joint_mm": (tip * 1e3).round(3).tolist(), "solve_s": round(time.time() - t0, 1)}
        info[name]["_patch"] = patch
        print(name, chk, f"{time.time() - t0:.0f} s", flush=True)
    rest_patch = set(info["rest"]["_patch"].tolist())
    for name in info:
        info[name]["patch_faces_same_as_rest"] = set(info[name].pop("_patch").tolist()) == rest_patch
    res = {"key": key, "species": r.meta["species"], "source": f"ear_{side}", "freqs_khz": (FREQS / 1e3).tolist(),
           "metric": "dB patterns normalised per frequency, -30 dB floor, cos(el)-weighted correlation; peak = angle "
                     "between maximum directions; max_gain_change_db = change of the pattern maximum (absolute level)",
           "poses": info, "comparison_to_rest": {}}
    a0 = np.abs(H["rest"])
    for name in poses:
        if name == "rest":
            continue
        a1 = np.abs(H[name])
        full = reference.metrics_points(a1, a0, wpt, u)
        fr = reference.metrics_points(a1[front], a0[front], wpt[front], u[front])
        pk0, pk1 = u[np.argmax(a0, 0)], u[np.argmax(a1, 0)]
        from batertf.rotate import to_az_el
        res["comparison_to_rest"][name] = {
            "corr_full_sphere": full["corr"].round(4).tolist(), "corr_frontal": fr["corr"].round(4).tolist(),
            "rms_db_full_sphere": full["rms_db"].round(2).tolist(), "rms_db_frontal": fr["rms_db"].round(2).tolist(),
            "peak_shift_deg": full["peak_deg"].round(1).tolist(),
            "peak_az_el_rest": np.round(np.stack(to_az_el(pk0), 1), 1).tolist(),
            "peak_az_el_pose": np.round(np.stack(to_az_el(pk1), 1), 1).tolist(),
            "max_gain_change_db": (20 * np.log10(a1.max(0) / a0.max(0))).round(2).tolist()}
        print(name, res["comparison_to_rest"][name], flush=True)
    rig.RIG_DIR.mkdir(exist_ok=True)
    (rig.RIG_DIR / f"demo_{key}.json").write_text(json.dumps(res, indent=1))
    out = paths.work_dir() / "rig"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / f"demo_{key}.npz", freqs=FREQS, points=pts, **{k.replace(" ", "_"): v for k, v in H.items()})

    # figure: rows = poses, columns = frequency x (front, back) LAEA; a mesh thumbnail per pose
    from batertf import rigplot
    n = len(poses)
    fig = plt.figure(figsize=(3.0 + 2.6 * 2 * len(FREQS), 2.7 * n))
    gs = fig.add_gridspec(n, 1 + 2 * len(FREQS), width_ratios=[1.3] + [1] * (2 * len(FREQS)))
    for i, (name, pose) in enumerate(poses.items()):
        ax = fig.add_subplot(gs[i, 0])
        m = r.deform(pose)
        R = rigplot.VIEWS["left side"] if axis == "flex" else rigplot.VIEWS["front"]
        rigplot.render(ax, m, R, rigplot.weight_colours(r))
        rigplot.draw_skeleton(ax, r, R, pose)
        rigplot.frame_axes(ax, [r.rest, r.deform(list(poses.values())[1]), r.deform(list(poses.values())[-1])], R)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
        Hg = gpubem.to_grid(H[name], "sphere")
        for kf, f in enumerate(FREQS):
            db = 20 * np.log10(np.abs(Hg[..., kf]) / np.abs(Hg[..., kf]).max())
            for s, sd in enumerate(("front", "back")):
                axm = fig.add_subplot(gs[i, 1 + 2 * kf + s])
                im = plotting.laea_hemi_map(axm, az, el, db, side=sd)
                t = f"{f/1e3:.0f} kHz {sd}"
                if name != "rest" and sd == "front":
                    c = res["comparison_to_rest"][name]
                    t += f"\nr={c['corr_full_sphere'][kf]:.2f}, peak {c['peak_shift_deg'][kf]:.0f} deg"
                axm.set_title(t, fontsize=8)
    cb = fig.colorbar(im, ax=fig.axes, shrink=0.5, pad=0.01)
    cb.set_label("dB re max (per frequency)")
    fig.suptitle(f"{r.meta['species']}: {side}-ear ERTF for {joint} {axis} rotations (LAEA, bat's left on the left; "
                 f"r = full-sphere pattern correlation with rest)", fontsize=10)
    fig.savefig(rig.RIG_DIR / f"demo_{key}.png", dpi=90)
    print("done", flush=True)


if __name__ == "__main__":
    main()
