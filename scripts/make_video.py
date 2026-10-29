"""Supplementary video of the paper (Cosys-Lab style, scripts/cosys_video.py).

Structure: title slide; two context slides (the dataset, with Fig. 1 cropped from the compiled paper; what is shown);
a turntable of the seven head meshes with their landmarks; frequency sweeps (20 kHz to the top of the band) of the
left ear, the emitter and the two-way ERTF of all species, as front/back hemispheres in Lambert azimuthal equal-area
projection, as in the paper; end slide. Uses every computed frequency on a 1 kHz grid (2 kHz while production runs).
Usage: python scripts/make_video.py [--test] [--out paper/video/bat_directivities_supplementary.mp4]"""
import argparse
import os
import sys
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cosys_video as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from batertf import grid, mesh, paths, rotate

ORDER = ["pdisc", "mmicr", "mmacr", "rrouxi", "rform", "efusc", "mdaub"]
TITLE = "Full-Sphere Emission and Reception Directivities\nof Seven Echolocating Bat Species"
N_PIX = 121                                         # pixels per disc diameter
VMIN = -30


def disc_dirs(side, n=N_PIX):
    """Unit vectors of the pixels of one LAEA hemisphere disc (as plotting.laea_hemi_map) and the outside mask."""
    sgn = 1.0 if side == "front" else -1.0
    x = np.linspace(-np.sqrt(2), np.sqrt(2), n)
    X, Y = np.meshgrid(x, x)
    rho2 = X ** 2 + Y ** 2
    c = 1 - np.minimum(rho2, 2.0) / 2
    k = np.sqrt(2 / (1 + c))
    u = np.stack([sgn * c, -X / k, Y / k], -1)
    return u / np.linalg.norm(u, axis=-1, keepdims=True), rho2 > 2.0


def load(key, step_khz):
    """Magnitudes of left ear and emitter [el, az, f] at every computed frequency on a step_khz grid."""
    with h5py.File(paths.work_dir() / "ertf" / f"{key}_ct.h5", "r") as h:
        f = h["freqs"][:]
        ok = h["done"][:] & (np.round(f / 1e3 / step_khz, 6) % 1 == 0)
        idx = np.where(ok)[0]
        return f[idx], {s: np.abs(h[s]["H"][..., idx]) for s in ("left", "emitter")}


def step_available():
    with h5py.File(paths.work_dir() / "ertf" / "mmicr_ct.h5", "r") as h:
        f, d = h["freqs"][:], h["done"][:]
    return 1.0 if d[np.round(f / 1e3) == f / 1e3].all() else 2.0


def lambert_legend(ax):
    """Context view: how to read the front/back discs."""
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-2.4, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")
    t = np.linspace(0, 2 * np.pi, 200)
    for cx, lab in ((-1.6, "Front"), (1.6, "Back")):
        ax.plot(cx + 1.41 * np.cos(t), 1.41 * np.sin(t), color="0.3", lw=1.5)
        ax.plot([cx - 1.41, cx + 1.41], [0, 0], color="0.7", lw=0.8)
        ax.plot([cx, cx], [-1.41, 1.41], color="0.7", lw=0.8)
        ax.text(cx, 1.75, lab, ha="center", fontsize=15, fontweight="bold")
        ax.text(cx - 1.15, 0.12, "L", fontsize=13, ha="center", color="0.25")
        ax.text(cx + 1.15, 0.12, "R", fontsize=13, ha="center", color="0.25")
        ax.text(cx + 0.18, 1.0, "Up", fontsize=12, color="0.25")
    ax.text(-1.6, -2.1, "centre: straight ahead", fontsize=12, ha="center")
    ax.text(1.6, -2.1, "centre: straight behind\n(seen from behind)", fontsize=12, ha="center", va="top")
    ax.set_title("How to read the maps", fontsize=17, fontweight="bold")


def sweep(wr, data, which, seq, nseq, test=False):
    """One frequency sweep over all species: panels of front/back discs, updated per frequency."""
    sp = paths.species()
    name = {"left": "Left ear", "emitter": "Emitter", "two-way": "Two-way (emitter x left ear)"}[which]
    lay = cv.DataLayout(TITLE, ["Response", "Frequency", "Sequence"], [sp[k]["taxon"] for k in ORDER] + [""],
                        ncol=4, cbar_label="Magnitude (dB re max)", vmin=VMIN, vmax=0,
                        note="Lambert equal-area front and back hemispheres, frame of each head (P. discolor: pose of De Mey et al.); 30 dB range, normalized\n"
                             "per species and frequency; E. fuscus and M. daubentonii emitters: analytical pistons")
    lambert_legend(lay.ctx)
    lay.axes[-1].axis("off")
    dirs = {s: disc_dirs(s) for s in ("front", "back")}
    ims = {}
    for ax, key in zip(lay.axes, ORDER):
        ax.title.set_fontstyle("italic")
        ax.title.set_fontweight("normal")
        ax.title.set_fontsize(16)
        ax.axis("off")
        blank = np.full((N_PIX, 2 * N_PIX + 8), np.nan)
        ims[key] = ax.imshow(blank, cmap=lay.cmap, vmin=VMIN, vmax=0, origin="lower", interpolation="bilinear")
        t = np.linspace(0, 2 * np.pi, 200)
        r = (N_PIX - 1) / 2
        for cx in (r, N_PIX + 8 + r):
            ax.plot(cx + r * np.cos(t), r + r * np.sin(t), color="0.4", lw=1)
        ax.set_xlim(-2, 2 * N_PIX + 10)
        ax.set_ylim(-2, N_PIX + 2)
        ax.text(r, -12, "front", ha="center", fontsize=12, color="0.4")
        ax.text(N_PIX + 8 + r, -12, "back", ha="center", fontsize=12, color="0.4")
    f_all = np.unique(np.concatenate([data[k][0] for k in ORDER]))
    if test:
        f_all = f_all[::max(1, len(f_all) // 6)]
    lay.status("Response", name)
    lay.status("Sequence", f"{seq} of {nseq}")
    n = len(f_all)
    for i, fq in enumerate(f_all):
        lay.status("Frequency", f"{fq / 1e3:.0f} kHz")
        for key in ORDER:
            f, d = data[key]
            j = np.where(np.isclose(f, fq))[0]
            img = np.full((N_PIX, 2 * N_PIX + 8), np.nan)
            if len(j):
                X = d["left"][..., j[0]] if which == "left" else d["emitter"][..., j[0]] if which == "emitter" \
                    else d["left"][..., j[0]] * d["emitter"][..., j[0]]
                db = 20 * np.log10(X / X.max())
                smp = rotate.SphereSampler(db)
                for c, side in enumerate(("front", "back")):
                    u, out = dirs[side]
                    v = np.maximum(smp(u), VMIN)
                    v[out] = np.nan
                    img[:, c * (N_PIX + 8):c * (N_PIX + 8) + N_PIX] = v
            ims[key].set_data(img)
        nf = 4
        wr.append_data(cv.fade(lay.frame(), min(1.0, (i + 1) / nf, (n - i) / nf)))
    lay.close()


def turntable(wr, seq, nseq, test=False):
    """The seven heads rotating about the vertical axis, with the ear (blue) and emitter (red) landmarks."""
    sp = paths.species()
    lay = cv.DataLayout(TITLE, ["Shown", "View azimuth", "Sequence"], [sp[k]["taxon"] for k in ORDER] + [""],
                        ncol=4, cbar_label="", note="Clean micro-CT head meshes as simulated (X forward, Y left, Z up; P. discolor in the pose of De Mey et al.)\n"
                        "blue: ear receiver patches; red: nostril patches or mouth piston centre")
    for cax in lay.fig.axes:                             # no colorbars for the meshes
        if cax.get_position().x0 > 0.93:
            cax.remove()
    lay.axes[-1].axis("off")
    lay.ctx.axis("off")
    lay.ctx.text(0.5, 0.6, "Each head is shown\nas simulated: the\nsame frame as its\ntransfer functions.",
                 ha="center", va="center", fontsize=17, transform=lay.ctx.transAxes)
    lay.status("Shown", "Head meshes")
    lay.status("Sequence", f"{seq} of {nseq}")
    axs3 = []
    for ax, key in zip(lay.axes, ORDER):
        pos = ax.get_position()
        ttl = ax.get_title()
        ax.remove()
        a3 = lay.fig.add_axes(pos, projection="3d")
        a3.set_title(ttl, fontsize=16, fontstyle="italic", pad=0)
        m = mesh.load(sp[key]["stl"])
        if len(m.faces) > 12000:                         # decimate for drawing only
            m = m.simplify_quadric_decimation(face_count=12000) if hasattr(m, "simplify_quadric_decimation") else m
        v = m.vertices * 1e3
        light = np.clip(m.face_normals @ np.array([0.4, 0.3, 0.85]), 0, 1) * 0.6 + 0.35
        pc = Poly3DCollection(v[m.faces], facecolors=plt.cm.gray(light), edgecolors="none")
        a3.add_collection3d(pc)
        lm = sp[key]["landmarks"]
        ears = np.array([lm["ear_left"], lm["ear_right"]]) * 1e3
        em = np.atleast_2d(lm.get("nostrils", lm.get("mouth"))) * 1e3
        a3.scatter(*ears.T, color="tab:blue", s=40, depthshade=False)
        a3.scatter(*em.T, color="tab:red", s=40, depthshade=False)
        r = 0.72 * np.abs(v).max()
        a3.set(xlim=(-r, r), ylim=(-r, r), zlim=(-r, r))
        a3.set_box_aspect((1, 1, 1))
        a3.axis("off")
        axs3.append(a3)
    n = 12 if test else 72
    for i in range(n):
        azim = -150 + 360 * i / n
        for a3 in axs3:
            a3.view_init(elev=15, azim=azim)
        lay.status("View azimuth", f"{(azim + 180) % 360 - 180:.0f}°")
        wr.append_data(cv.fade(lay.frame(), min(1.0, (i + 1) / 4, (n - i) / 4)))
    lay.close()


def head_polys(key, n_faces=12000):
    """Decimated mesh (mm) as a shaded Poly3DCollection, and the landmarks (mm)."""
    sp = paths.species()[key]
    m = mesh.load(sp["stl"])
    if len(m.faces) > n_faces:
        m = m.simplify_quadric_decimation(face_count=n_faces)
    v = m.vertices * 1e3
    light = np.clip(m.face_normals @ np.array([0.4, 0.3, 0.85]), 0, 1) * 0.6 + 0.35
    lm = sp["landmarks"]
    return (Poly3DCollection(v[m.faces], facecolors=plt.cm.gray(light), edgecolors="none"), v,
            np.array([lm["ear_left"], lm["ear_right"]]) * 1e3, np.atleast_2d(lm.get("nostrils", lm.get("mouth"))) * 1e3)


def balloons(wr, data, f_show, seq, nseq, test=False):
    """Per species: the head rotating together with the 3D directivity balloons of the left ear and the emitter at
    f_show (radius and colour: dB re max, 30 dB range), so the beams stay attached to the anatomy."""
    sp = paths.species()
    lay = cv.DataLayout(TITLE, ["Species", "Frequency", "View azimuth", "Sequence"],
                        ["Head mesh", "Left ear", "Emitter"], ncol=3, cbar_label="Magnitude (dB re max)",
                        vmin=VMIN, vmax=0, note="Balloons: radius shows the linear magnitude, colour the level in dB re max (30 dB "
                        "range) per direction, centred on the origin of the frame;\nthe head is drawn to scale "
                        "inside; E. fuscus and M. daubentonii emitters: analytical pistons")
    lambert = lay.ctx
    lambert.axis("off")
    lay.status("Frequency", f"{f_show / 1e3:.0f} kHz")
    lay.status("Sequence", f"{seq} of {nseq}")
    az, el = grid.GRIDS["sphere"]
    azc = np.append(az, 180.0)
    A, E = np.meshgrid(np.deg2rad(azc), np.deg2rad(el))
    U = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)
    keys = ORDER[:2] if test else ORDER
    for key in keys:
        f, d = data[key]
        k = int(np.argmin(abs(f - f_show)))
        lay.status("Species", sp[key]["taxon"])
        lambert.clear()
        lambert.axis("off")
        lambert.text(0.5, 0.55, sp[key]["taxon"].replace(" ", "\n"), ha="center", va="center", fontsize=24,
                     style="italic", transform=lambert.transAxes)
        lambert.text(0.5, 0.25, sp[key]["family"] + ("\nnasal emitter (BEM)" if sp[key]["emitter"] == "nasal"
                     else "\noral emitter (piston)"), ha="center", va="center", fontsize=15, transform=lambert.transAxes)
        axs3 = []
        for i, ax in enumerate(lay.axes):
            pos, ttl = ax.get_position(), ax.get_title()
            ax.remove()
            a3 = lay.fig.add_axes(pos, projection="3d")
            a3.set_title(ttl, fontsize=19, fontweight="bold", pad=0)
            pc, v, ears, em = head_polys(key)
            a3.add_collection3d(pc)
            R = 2.4 * np.abs(v).max()
            if i == 0:
                a3.scatter(*ears.T, color="tab:blue", s=50, depthshade=False)
                a3.scatter(*em.T, color="tab:red", s=50, depthshade=False)
                lim = 0.72 * np.abs(v).max()
            else:
                X = d["left" if i == 1 else "emitter"][..., k]
                db = np.maximum(20 * np.log10(X / X.max()), VMIN)
                db = np.concatenate([db, db[:, :1]], 1)
                r = R * 10 ** (db / 20)                   # radius: linear magnitude, colour: dB
                P = U * r[..., None]
                a3.plot_surface(P[..., 0], P[..., 1], P[..., 2], facecolors=lay.cmap((db - VMIN) / -VMIN),
                                rstride=2, cstride=2, linewidth=0, antialiased=False, alpha=0.55, shade=False)
                lim = 0.62 * R
            a3.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
            a3.set_box_aspect((1, 1, 1))
            a3.axis("off")
            axs3.append(a3)
        n = 12 if test else 60
        for j in range(n):
            azim = -150 + 360 * j / n
            for a3 in axs3:
                a3.view_init(elev=20, azim=azim)
            lay.status("View azimuth", f"{(azim + 180) % 360 - 180:.0f}°")
            wr.append_data(cv.fade(lay.frame(), min(1.0, (j + 1) / 4, (n - j) / 4)))
        for a3 in axs3:
            pos = a3.get_position()
            a3.remove()
            lay.axes[axs3.index(a3)] = lay.fig.add_axes(pos)
            lay.axes[axs3.index(a3)].set_title(["Head mesh", "Left ear", "Emitter"][axs3.index(a3)], fontsize=19,
                                               fontweight="bold")
    lay.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--out", default=str(paths.ROOT / "paper" / "video" / "bat_directivities_supplementary.mp4"))
    a = ap.parse_args()
    step = step_available()
    data = {k: load(k, step) for k in ORDER}
    wr = cv.writer(a.out)
    cv.hold(wr, cv.slide_title(TITLE), 1 if a.test else 4.5, fade_in=False)
    cv.hold(wr, cv.slide_text("The dataset", [
        "Seven echolocating bat species from three families: Phyllostomidae, Rhinolophidae and Vespertilionidae.",
        "For each species, a clean head mesh from a micro-CT scan of one preserved specimen, with the ear canal "
        "entrances and the nostrils or the mouth as landmarks.",
        "The complex transfer functions of both ears and of the emitter on the full sphere, from 20 to 160 kHz "
        "in 0.5 kHz steps, computed with a GPU boundary element method verified against Mesh2HRTF.",
        "The emitters of the two mouth-emitting species are analytical pistons calibrated to field measurements, "
        "and all other responses are boundary element results."]), 1 if a.test else 11)
    cv.hold(wr, cv.slide_text("What is shown", [
        "First, the seven head meshes, rotating, exactly in the frame in which they were simulated.",
        "Next, each head rotating together with the 3D beam balloons of its left ear and its emitter at 60 kHz.",
        "Then, for all species at once, frequency sweeps of the left ear, the emitter, and their product, the "
        "two-way echolocation-related transfer function.",
        "Finally, the posable models: the pinnae of two species move in antiphase, and the ERTF is recomputed for "
        "every pose.",
        "Every map shows the front and the back hemisphere in Lambert azimuthal equal-area projection: the bat's "
        "left is on the left in both, so a lobe that wraps around the side continues from one disc into the other.",
        "Each pair is normalized to its maximum over the whole sphere, with a 30 dB range."]), 1 if a.test else 11)
    nseq = 7
    turntable(wr, 1, nseq, a.test)
    balloons(wr, data, 60e3, 2, nseq, a.test)
    for s, which in enumerate(("left", "emitter", "two-way")):
        sweep(wr, data, which, s + 3, nseq, a.test)
    import make_rig_video as rv                               # posable models: antiphase pinna cycles
    ang, _, _, _, f_rig, _ = rv.load("mmicr", True)
    cv.hold(wr, cv.slide_text("Posable head models", [
        "Every head in the dataset has a skeleton in each pinna and, for the nasal species, in the noseleaf, so the "
        "meshes can be posed and simulated again.",
        "In the next two sequences, the pinnae of Micronycteris microtis and Rhinolophus formosae swing in antiphase: "
        "the base of each pinna rotates back and forth and its tip bends along, while the other pinna moves the "
        "opposite way.",
        "For every one of the 24 poses per cycle, the left ear, the right ear and the emitter were computed with the "
        f"boundary element method at {', '.join(f'{x / 1e3:.0f}' for x in f_rig)} kHz. Every frame is a computed pose, "
        "nothing is interpolated."]), 1 if a.test else 11)
    for s, key in enumerate(("mmicr", "rform")):
        rv.pinna_sequence(wr, key, cycle=True, test=a.test, title=TITLE, seq_no=(s + 6, nseq))
    cv.hold(wr, cv.slide_end(), 1 if a.test else 3.5, fade_out=False)
    wr.close()
    print(a.out)


if __name__ == "__main__":
    main()
