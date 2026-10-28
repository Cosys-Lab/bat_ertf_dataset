"""Video: moving pinnae and the resulting change of the ERTF, for one species (Cosys-Lab style, scripts/cosys_video.py).

Data: <work>/rig/sweep_<key>.npz from scripts/rig_sweep_bem.py (both pinnae flexed symmetrically about their base
joints; left ear, right ear and emitter at three frequencies). Frame: title slide, a "what is shown" slide, the sweep
played forward and back twice while the view of the head turns slowly, end slide. Left: the deformed head (moving parts
tinted) in two views; right: per frequency (rows), front/back Lambert maps of the left ear, the right ear and the
two-way ERTF of the left ear (emitter x left ear), each in dB re its maximum over the sphere, 30 dB range.
Usage: python scripts/make_rig_video.py [key] [--test]  -> paper/video/<key>_pinna_sweep.mp4"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cosys_video as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from batertf import paths, rotate
from make_video import disc_dirs, N_PIX, VMIN

COLS = [("Left ear", lambda H: H[0]), ("Right ear", lambda H: H[1]), ("Two-way, left", lambda H: H[2] * H[0])]


def head_axes(fig, rect, V, F, moving, elev, azim, lim):
    ax = fig.add_axes(rect, projection="3d")
    light = np.ones(len(F))
    pc = Poly3DCollection(V[F], edgecolors="none")
    ax.add_collection3d(pc)
    c0 = (V.max(0) + V.min(0)) / 2                    # centre on the head, not on the origin
    ax.set(xlim=(c0[0] - lim, c0[0] + lim), ylim=(c0[1] - lim, c0[1] + lim), zlim=(c0[2] - lim, c0[2] + lim))
    ax.set_box_aspect((1, 1, 1), zoom=1.5)
    ax.axis("off")
    ax.view_init(elev=elev, azim=azim)
    return ax, pc


def shade(V, F, moving):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    lit = np.clip(n @ np.array([0.4, 0.3, 0.85]), 0, 1) * 0.6 + 0.35
    c = plt.cm.gray(lit)
    c[moving, :3] = c[moving, :3] * np.array([1.0, 0.55, 0.85])            # moving parts tinted magenta
    return c


def load(key, cycle):
    d = np.load(paths.work_dir() / "rig" / f"{'cycle' if cycle else 'sweep'}_{key}.npz", allow_pickle=True)
    fin = d["finished"]
    return (d["angles"][fin], np.abs(d["H"][fin]), d["vertices"][fin] * 1e3, d["faces"], d["freqs"], d["valid"][fin])


def motion_text(ang, cycle):
    if cycle:
        return (f"the two pinnae swing in antiphase over one cycle of {len(ang)} poses: the base of each pinna rotates "
                f"between {ang[:, 0].min():+.0f} and {ang[:, 0].max():+.0f} degrees (positive: backward) and its tip "
                f"bends by up to {abs(ang[:, 2]).max():.0f} degrees, while the other pinna moves the opposite way")
    return (f"both pinnae are rotated together about their base, from {ang.min():+.0f} to {ang.max():+.0f} degrees "
            f"(positive: backward), in steps of {ang[1] - ang[0]:.1f} degrees")


def pinna_sequence(wr, key, cycle=True, test=False, title=None, seq_no=None):
    """Data frames of one pinna sweep or antiphase cycle, appended to the writer wr (fades in and out).
    title: frame title (default: species-specific); seq_no: (i, n) adds a "Sequence: i of n" status line."""
    ang, H, Vall, F, freqs, valid = load(key, cycle)
    sys.path.insert(0, str(paths.ROOT / "paper" / "figures"))
    from fig_posing import level_matrix                # display only: ear-nostril line horizontal
    Vall = Vall @ level_matrix(key).T
    sp = paths.species()[key]
    moving = (np.linalg.norm(Vall.max(0) - Vall.min(0), axis=1)[F].max(1) > 0.05)
    lim = 0.55 * np.ptp(Vall[0], axis=0).max()
    head = "Alternating Pinnae" if cycle else "Moving Pinnae"
    title = title or f"{head} and the Echolocation-Related Transfer Function\nof {sp['taxon']}"
    fig = cv.new_fig()
    fig.text(0.5, 0.925, title, ha="center", va="center", fontsize=28, linespacing=1.2)
    st = {}
    labels = (("Species", "Left pinna", "Right pinna", "Cycle") if cycle else ("Species", "Pinna rotation", "Sweep"))
    labels += (("Sequence",) if seq_no else ())
    for i, lab in enumerate(labels):
        fig.text(0.015, 0.80 - 0.036 * i, f"{lab}:", fontsize=15, fontweight="bold")
        st[lab] = fig.text(0.105, 0.80 - 0.036 * i, "", fontsize=15)
    st["Species"].set_text(sp["taxon"])
    st["Species"].set_fontstyle("italic")
    if seq_no:
        st["Sequence"].set_text(f"{seq_no[0]} of {seq_no[1]}")
    y0 = 0.36 if seq_no or cycle else 0.40
    ax_side, pc_side = head_axes(fig, [0.0, y0, 0.27, 0.30], Vall[0], F, moving, 10, -90, lim)
    ax_front, pc_front = head_axes(fig, [0.0, 0.07, 0.27, 0.30], Vall[0], F, moving, 15, 20, lim)
    dirs = {s: disc_dirs(s) for s in ("front", "back")}
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("white")
    x0, x1, top, bot = 0.29, 0.93, 0.83, 0.12
    pw, ph = (x1 - x0) / 3, (top - bot) / len(freqs)
    ims = {}
    t = np.linspace(0, 2 * np.pi, 200)
    r = (N_PIX - 1) / 2
    for i, f in enumerate(freqs):
        for c, (name, _) in enumerate(COLS):
            ax = fig.add_axes([x0 + c * pw + 0.004, top - (i + 1) * ph + 0.012, pw - 0.008, ph - 0.03])
            ims[i, c] = ax.imshow(np.full((N_PIX, 2 * N_PIX + 8), np.nan), cmap=cmap, vmin=VMIN, vmax=0,
                                  origin="lower", interpolation="bilinear")
            for cx in (r, N_PIX + 8 + r):
                ax.plot(cx + r * np.cos(t), r + r * np.sin(t), color="0.4", lw=1)
            ax.set_xlim(-2, 2 * N_PIX + 10)
            ax.set_ylim(-14, N_PIX + 2)
            ax.axis("off")
            ax.text(r, -12, "front", ha="center", fontsize=11, color="0.4")
            ax.text(N_PIX + 8 + r, -12, "back", ha="center", fontsize=11, color="0.4")
            if i == 0:
                ax.set_title(name, fontsize=19, fontweight="bold", pad=4)
        fig.text(x0 - 0.008, top - (i + 0.5) * ph, f"{f / 1e3:.0f} kHz", ha="right", va="center", fontsize=17,
                 fontweight="bold")
    cax = fig.add_axes([0.94, 0.3, 0.009, 0.4])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(VMIN, 0), cmap=cmap), cax=cax)
    cb.ax.tick_params(labelsize=11)
    cb.set_label("Magnitude (dB re max)", fontsize=11)
    fig.text(0.56, 0.06, ("Pinnae in antiphase, base and tip joints" if cycle else "Both pinnae rotated about their base joint") + " (rig of the dataset mesh, dual-quaternion "
             "skinning); BEM at every pose;\nmaps: front and back hemispheres, normalized per map, 30 dB range",
             ha="center", fontsize=13, color="0.4")
    cv.logos(fig)

    def maps(k):
        for i in range(len(freqs)):
            for c, (_, fn) in enumerate(COLS):
                X = fn(H[k, :, :, :, i])
                smp = rotate.SphereSampler(20 * np.log10(X / X.max()))
                img = np.full((N_PIX, 2 * N_PIX + 8), np.nan)
                for j, side in enumerate(("front", "back")):
                    u, outside = dirs[side]
                    v = np.maximum(smp(u), VMIN)
                    v[outside] = np.nan
                    img[:, j * (N_PIX + 8):j * (N_PIX + 8) + N_PIX] = v
                ims[i, c].set_data(img)

    if cycle:                                                                 # periodic: loop the cycle
        seq = list(range(len(ang)))
        reps, hold = (1, 1) if test else (3, 3)
    else:                                                                     # forward and back
        seq = list(range(len(ang))) + list(range(len(ang) - 2, 0, -1))
        reps, hold = (1, 1) if test else (2, 3)
    frames = [k for _ in range(reps) for k in seq for _ in range(hold)]
    if test:
        frames = frames[::max(1, len(frames) // 12)]
    cache = {}
    n = len(frames)
    for fi, k in enumerate(frames):
        if k not in cache:
            maps(k)
            cache[k] = {key: im.get_array().copy() for key, im in ims.items()}
        else:
            for key, im in ims.items():
                im.set_data(cache[k][key])
        V = Vall[k]
        col = shade(V, F, moving)
        for pc in (pc_side, pc_front):
            pc.set_verts(V[F])
            pc.set_facecolor(col)
        turn = 360 * fi / n * reps / 2                                            # one slow turn per two sweeps
        ax_side.view_init(elev=10, azim=-90 + 0.15 * turn)
        ax_front.view_init(elev=15, azim=20 + turn)
        if cycle:
            g = lambda x: f"{round(x) + 0:+d}"            # no "-0"
            st["Left pinna"].set_text(f"base {g(ang[k, 0])}°, tip {g(ang[k, 2])}°")
            st["Right pinna"].set_text(f"base {g(ang[k, 1])}°, tip {g(ang[k, 3])}°")
            st["Cycle"].set_text(f"{fi // (len(seq) * hold) + 1} of {reps}")
        else:
            st["Pinna rotation"].set_text(f"{ang[k]:+.1f}°  (valid mesh)" if valid[k] else f"{ang[k]:+.1f}°")
            st["Sweep"].set_text("forward" if fi % len(seq * hold) < len(ang) * hold else "back")
        wr.append_data(cv.fade(cv.grab(fig), min(1.0, (fi + 1) / 5, (n - fi) / 5)))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key", nargs="?", default="mmicr")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--cycle", action="store_true", help="antiphase cycle of scripts/rig_cycle_bem.py")
    a = ap.parse_args()
    ang, _, _, _, freqs, _ = load(a.key, a.cycle)
    sp = paths.species()[a.key]
    head = "Alternating Pinnae" if a.cycle else "Moving Pinnae"
    out = paths.ROOT / "paper" / "video" / f"{a.key}_pinna_{'cycle' if a.cycle else 'sweep'}.mp4"
    wr = cv.writer(out)
    cv.hold(wr, cv.slide_title(f"{head} and the Echolocation-Related\nTransfer Function of " + sp["taxon"]),
            1 if a.test else 4, fade_in=False)
    cv.hold(wr, cv.slide_text("What is shown", [
        f"The head mesh of {sp['taxon']} from the dataset, with a skeleton in each pinna: " + motion_text(ang, a.cycle) + ".",
        "For every pose, the left ear, the right ear and the emitter (the two nostrils) were computed with the boundary "
        f"element method at {', '.join(f'{f / 1e3:.0f}' for f in freqs)} kHz on the full sphere.",
        "The maps show the front and the back hemisphere in Lambert azimuthal equal-area projection (the bat's left on "
        "the left in both), each normalized to its maximum, with a 30 dB range. The magenta parts of the head move.",
        "Every frame is a computed pose; nothing is interpolated."]), 1 if a.test else 12)
    pinna_sequence(wr, a.key, a.cycle, a.test)
    cv.hold(wr, cv.slide_end(), 1 if a.test else 3.5, fade_out=False)
    wr.close()
    print(out)


if __name__ == "__main__":
    main()
