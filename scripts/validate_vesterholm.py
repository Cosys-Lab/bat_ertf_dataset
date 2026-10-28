"""External validation against measured in-ear HRTFs: Myotis daubentonii "Mdau2" of Vesterholm et al. 2025
(Methods Ecol. Evol., doi:10.1111/2041-210X.70197; data doi:10.5281/zenodo.15789942, CC-BY 4.0). Conventions and
provenance: docs/vesterholm_validation.md. Data: scripts/fetch_vesterholm.py -> <data>/files/...

1. Mesh: their µCT head mesh (3D Models/Mdau2_CTscan.ply, mm) remeshed with our dataset pipeline
   (scripts/remesh_ct.remesh_isolated, target edge max(L_MIN, sqrt(A / (0.433 N_MAX)))), metres, kept in their frame
   (the frame of their Mesh2HRTF simulation: its receiver lies on this surface).
2. Receiver: our disc patch (radius project.EAR_RADIUS) centred at their Mesh2HRTF receiver position
   (SOFA ReceiverPosition, left ear).
3. Directions: their 72 x 15 measurement directions (turntable azimuth sign-flipped so +az = bat's left, speaker
   elevations; zenith measured 72 times -> one direction, averaged), rotated into the mesh frame with their
   RotoTranslationMatrix (u_mesh = R u_meas, as in their ir2hrtf.m), plus their full Mesh2HRTF grid (100 el x 199 az).
4. GPU BEM (gpubem.solve_lu), 1 m, and comparison with (a) the measured magnitude (|DTFT| of their 512-tap minimum-
   phase IRs at f, = their pure-tone RMS), (b) their Mesh2HRTF simulation (magnitude on the measurement directions,
   magnitude and phase on their grid). Metrics: reference.metrics_points (per-frequency max-normalized dB patterns,
   -30 dB floor, solid-angle weights).
Usage: python scripts/validate_vesterholm.py [--coarse] [--data DIR] [--fmin 25 --fmax 100 --df 2.5]
  default (full): mesh <= 58,000 faces, 25-100 kHz in 2.5 kHz steps -> results/validation/vesterholm.json
  --coarse: mesh <= 20,000 faces, 30/50/70/90 kHz (GPU smoke test) -> results/validation/vesterholm_coarse.json
Raw transfer functions and maps: <work>/vesterholm/{solve,maps}_<tag>.npz (read by paper/figures/fig_vesterholm.py)."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import trimesh
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batertf import paths, project, reference, sources
from remesh_ct import L_MIN, N_MAX, remesh_isolated

SUB = "Data and code - Resubmission"
N_COARSE = 20000
F_COARSE = [30e3, 50e3, 70e3, 90e3]
FLOOR = -30.0
PHASE_DB = -20.0          # phase compared where both magnitudes are within this of their maximum
CHUNK = 8                 # frequencies per solve call (results saved after each chunk; reruns resume)


def unit(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], -1)


def load_measured(d):
    """Measurement grid and IRs. Returns az, el (P,) in the measurement frame (+az = bat's left), the (P, 512) IRs
    with the zenith collapsed to one direction (mean IR of its 72 recordings, magnitudes are averaged later),
    the per-point solid-angle weights, and the [15, 72] grid map (index into P) for plotting."""
    m = sio.loadmat(d / SUB / "Measured HRTF" / "Mdau2_ir.mat", squeeze_me=True)
    ir = m["files_impulse_response"]
    ttp = m["files_ttable_pos"].astype(float)
    el = m["speakers"][m["files_speaker_num"] - 1, 1].astype(float)
    keep = ttp < 360                                    # 360 duplicates 0
    az = -np.where(ttp > 180, ttp - 360, ttp)[keep]     # as ir2hrtf.m: +-180, then +az = left
    el, ir = el[keep], ir[keep]
    els = np.unique(el)                                 # -80 ... 90
    azs = np.unique(az)
    zen = el == 90
    z_ir = ir[zen]
    pts_az = np.concatenate([az[~zen], [0.0]])
    pts_el = np.concatenate([el[~zen], [90.0]])
    edges = np.concatenate([[-90], (els[1:] + els[:-1]) / 2, [90]])
    band = np.sin(np.radians(edges[1:])) - np.sin(np.radians(edges[:-1]))
    w = np.array([band[np.searchsorted(els, e)] * (5 / 360 if e < 90 else 1.0) for e in pts_el])
    gmap = np.zeros((len(els), len(azs)), int)
    for i, (a, e) in enumerate(zip(pts_az, pts_el)):
        if e < 90:
            gmap[np.searchsorted(els, e), np.searchsorted(azs, a)] = i
    gmap[-1] = len(pts_az) - 1
    return {"az": pts_az, "el": pts_el, "ir": ir[~zen], "ir_zenith": z_ir, "fs": float(m["fs"]), "w": w,
            "grid_el": els, "grid_az": azs, "gmap": gmap}


def measured_mag(meas, freqs):
    n = np.arange(meas["ir"].shape[1])
    E = np.exp(-2j * np.pi * np.outer(n, freqs) / meas["fs"])
    zen = np.abs(meas["ir_zenith"] @ E).mean(0)          # ir2hrtf.m: zenith = mean of its 72 values
    return np.vstack([np.abs(meas["ir"] @ E), zen])


def load_sofa(d):
    with h5py.File(d / SUB / "SOFA files" / "Mdau2_CTscan.sofa", "r") as h:
        H = h["Data.Real"][:, 0, :] + 1j * h["Data.Imag"][:, 0, :]
        S = h["SourcePosition"][()]
        rec = h["ReceiverPosition"][0]
        f = h["N"][()]
        meta = {k: (v.decode() if isinstance(v, bytes) else str(v)) for k, v in h.attrs.items()
                if k in ("ApplicationName", "ApplicationVersion", "Comment", "SOFAConventions", "DateCreated",
                         "License")}
    G = S.reshape(100, 199, 3)                           # [el, az]; az = -180 + k 360/199 (stored as 0..360)
    assert np.ptp(G[..., 1], 1).max() < 1e-3 and np.ptp(G[2:-2, :, 0], 0).max() < 1e-2
    az = (G[50, :, 0] + 180) % 360 - 180
    el = G[:, 0, 1]
    return {"H": H, "az": az, "el": el, "u": unit(S[:, 0], S[:, 1]), "receiver": rec, "freqs": f, "meta": meta,
            "w": np.cos(np.radians(S[:, 1]))}


def sofa_sampler(sofa, k):
    """Bilinear interpolation of |H_sofa| at frequency index k, at unit vectors u."""
    A = np.abs(sofa["H"][:, k]).reshape(100, 199)
    az = np.concatenate([sofa["az"], [sofa["az"][0] + 360]])
    f = RegularGridInterpolator((sofa["el"], az), np.concatenate([A, A[:, :1]], 1))

    def at(u):
        a = np.degrees(np.arctan2(u[:, 1], u[:, 0]))
        a = (a - az[0]) % 360 + az[0]
        return f(np.stack([np.degrees(np.arcsin(np.clip(u[:, 2], -1, 1))), a], -1))
    return at


def prepare_mesh(ply, n_max, fn):
    if fn.exists():
        return trimesh.load(fn, process=True), json.loads(fn.with_suffix(".json").read_text())
    src = trimesh.load(ply, process=True)
    L = max(L_MIN * 1e3, np.sqrt(src.area / (0.433 * n_max)))       # mm
    m, clean = remesh_isolated(ply, L)
    while len(m.faces) > n_max:                                     # the remesher overshoots the area estimate
        L *= np.sqrt(len(m.faces) / n_max) * 1.01
        m, clean = remesh_isolated(ply, L)
    assert clean, "remeshed Mdau2 mesh is not clean"
    m.apply_scale(1e-3)
    src.apply_scale(1e-3)
    assert m.is_watertight and m.is_winding_consistent
    if m.volume < 0:
        m.invert()
    d1 = trimesh.proximity.closest_point(src, m.sample(8000))[1]
    e = m.edges_unique_length
    st = {"source": str(Path(ply).name), "n_faces_source": int(len(src.faces)), "n_faces": int(len(m.faces)),
          "target_edge_mm": L, "edge_mm_mean_min_max": [e.mean() * 1e3, e.min() * 1e3, e.max() * 1e3],
          "f_max_6_per_wavelength_khz": 343 / (6 * e.mean()) / 1e3, "area_mm2": m.area * 1e6,
          "area_source_mm2": src.area * 1e6, "volume_mm3": m.volume * 1e9,
          "deviation_to_source_mm_median_p95_max": [float(np.median(d1) * 1e3), float(np.percentile(d1, 95) * 1e3),
                                                    float(d1.max() * 1e3)]}
    fn.parent.mkdir(parents=True, exist_ok=True)
    m.export(fn)
    fn.with_suffix(".json").write_text(json.dumps(st, indent=1))
    return m, st


def solve(m, patch, freqs, pts, fn):
    """gpubem.solve_lu in chunks of frequencies, saved after each chunk (resumable)."""
    from batertf import gpubem
    H = np.full((len(pts), len(freqs)), np.nan + 0j, complex)
    done = np.zeros(len(freqs), bool)
    if fn.exists():
        z = np.load(fn)
        if z["freqs"].shape == freqs.shape and np.allclose(z["freqs"], freqs) and z["H"].shape == H.shape:
            H, done = z["H"], z["done"]
    t0 = time.time()
    todo = np.where(~done)[0]
    for c in range(0, len(todo), CHUNK):
        idx = todo[c:c + CHUNK]
        H[:, idx] = gpubem.solve_lu(m, {"left": [patch]}, freqs[idx], pts, log=print)["left"]
        done[idx] = True
        np.savez(fn, H=H, freqs=freqs, done=done)
        print(f"{done.sum()}/{len(freqs)} frequencies, {(time.time() - t0) / 60:.1f} min", flush=True)
    return H


def tolist(d):
    return {k: (np.where(np.isfinite(v), np.round(v, 5), np.nan).tolist() if isinstance(v, np.ndarray) else v)
            for k, v in d.items()}


def summary(met):
    c, r = np.asarray(met["corr"], float), np.asarray(met["rms_db"], float)
    ok = np.isfinite(c)
    return {"n_freqs": int(ok.sum()), "corr_mean": float(c[ok].mean()), "corr_median": float(np.median(c[ok])),
            "corr_min": float(c[ok].min()), "rms_db_mean": float(r[ok].mean()),
            "rms_db_median": float(np.median(r[ok]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--data", type=Path, default=paths.work_dir().parent / "external" / "vesterholm2025")
    ap.add_argument("--fmin", type=float, default=25.0)
    ap.add_argument("--fmax", type=float, default=100.0)
    ap.add_argument("--df", type=float, default=2.5)
    args = ap.parse_args()
    tag = "coarse" if args.coarse else "full"
    freqs = np.array(F_COARSE) if args.coarse else np.round(np.arange(args.fmin, args.fmax + args.df / 2, args.df) * 1e3, 3)
    work = paths.work_dir() / "vesterholm"
    work.mkdir(parents=True, exist_ok=True)

    meas = load_measured(args.data / "files")
    sofa = load_sofa(args.data / "files")
    R = sio.loadmat(args.data / "files" / SUB / "Measured HRTF" / "Mdau2_RotationMatrix.mat")["RotoTranslationMatrix"]
    R3 = R[:3, :3]
    u_meas = unit(meas["az"], meas["el"])                  # measurement frame
    u_mesh = u_meas @ R3.T                                 # u_mesh = R u_meas (ir2hrtf.m)

    ply = args.data / "files" / SUB / "3D Models" / "Mdau2_CTscan.ply"
    m, mst = prepare_mesh(ply, N_COARSE if args.coarse else N_MAX, work / f"Mdau2_{tag}.stl")
    rec = sofa["receiver"]
    patch = sources.disc_patch(m, rec, project.EAR_RADIUS)
    pinfo = sources.patch_info(m, patch)
    print(f"mesh {mst['n_faces']} faces (edge {mst['edge_mm_mean_min_max'][0]:.3f} mm, f6 "
          f"{mst['f_max_6_per_wavelength_khz']:.0f} kHz); patch {pinfo['n_faces']} faces, centroid "
          f"{np.linalg.norm(pinfo['centroid'] - rec) * 1e3:.3f} mm from their receiver; {len(freqs)} frequencies",
          flush=True)

    # evaluation points (1 m): measurement directions in the mesh frame, the same unrotated (alignment check),
    # their Mesh2HRTF grid
    P = len(u_meas)
    pts = np.vstack([u_mesh, u_meas, sofa["u"]])
    H = solve(m, patch, freqs, pts, work / f"solve_{tag}.npz")
    H_rot, H_unrot, H_grid = H[:P], H[P:2 * P], H[2 * P:]

    Mm = measured_mag(meas, freqs)
    ks = np.array([int(np.argmin(abs(sofa["freqs"] - f))) for f in freqs])
    has = np.isclose(sofa["freqs"][ks], freqs)
    S_meas = np.full((P, len(freqs)), np.nan)
    for j in np.where(has)[0]:
        S_meas[:, j] = sofa_sampler(sofa, ks[j])(u_mesh)

    w, wg = meas["w"], sofa["w"]
    nan = lambda: {"rms_db": np.full(len(freqs), np.nan), "corr": np.full(len(freqs), np.nan),
                   "peak_deg": np.full(len(freqs), np.nan)}
    res = {"ours_vs_measured": reference.metrics_points(np.abs(H_rot), Mm, w, u_meas, FLOOR),
           "ours_vs_measured_without_rotation": reference.metrics_points(np.abs(H_unrot), Mm, w, u_meas, FLOOR),
           "m2h_vs_measured": nan(), "ours_vs_m2h": nan()}
    cplx = {k: np.full(len(freqs), np.nan) for k in ("level_offset_db", "complex_corr", "complex_corr_conj",
                                                      "phase_rms_deg", "phase_rms_deg_conj")}
    for j in np.where(has)[0]:
        sl = slice(j, j + 1)
        for k, v in reference.metrics_points(S_meas[:, sl], Mm[:, sl], w, u_meas, FLOOR).items():
            res["m2h_vs_measured"][k][j] = v[0]
        Hs = sofa["H"][:, ks[j]]
        for k, v in reference.metrics_points(np.abs(H_grid[:, sl]), np.abs(Hs)[:, None], wg, sofa["u"],
                                             FLOOR).items():
            res["ours_vs_m2h"][k][j] = v[0]
        a = H_grid[:, j]
        valid = (np.abs(a) > np.abs(a).max() * 10 ** (PHASE_DB / 20)) & \
                (np.abs(Hs) > np.abs(Hs).max() * 10 ** (PHASE_DB / 20))
        wv = wg * valid
        cplx["level_offset_db"][j] = np.sum(wv * 20 * np.log10(np.abs(a) / np.abs(Hs))) / wv.sum()
        for sfx, b in (("", Hs), ("_conj", np.conj(Hs))):
            cplx["complex_corr" + sfx][j] = np.abs(np.sum(wg * a * np.conj(b))) / np.sqrt(
                np.sum(wg * np.abs(a) ** 2) * np.sum(wg * np.abs(b) ** 2))
            dph = np.angle(a * np.conj(b))
            cplx["phase_rms_deg" + sfx][j] = np.degrees(np.sqrt(np.sum(wv * dph ** 2) / wv.sum()))
    res["ours_vs_m2h"].update(cplx)

    # broadband: power averaged over the frequencies (those with Mesh2HRTF data for the m2h comparisons)
    bb = lambda X, sel=slice(None): np.sqrt(np.nanmean(np.abs(X[:, sel]) ** 2, 1))[:, None]
    broadband = {"ours_vs_measured": reference.metrics_points(bb(H_rot), bb(Mm), w, u_meas, FLOOR)}
    if has.any():
        broadband["m2h_vs_measured"] = reference.metrics_points(bb(S_meas, has), bb(Mm, has), w, u_meas, FLOOR)
        broadband["ours_vs_m2h"] = reference.metrics_points(bb(H_grid, has), bb(sofa["H"][:, ks[has]]), wg,
                                                            sofa["u"], FLOOR)
    phase_conv = "conj" if np.nanmedian(cplx["phase_rms_deg_conj"]) < np.nanmedian(cplx["phase_rms_deg"]) else "direct"

    out = {"what": "Validation against Vesterholm et al. 2025, M. daubentonii Mdau2 (left ear): measured in-ear "
                   "HRTF magnitude and their Mesh2HRTF simulation; docs/vesterholm_validation.md",
           "run": "COARSE TEST (mesh <= 20k faces, 4 frequencies)" if args.coarse else "full",
           "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                 cwd=paths.ROOT).stdout.strip(),
           "source": {"zenodo": "10.5281/zenodo.15789942", "paper": "10.1111/2041-210X.70197",
                      "licence": "CC-BY-4.0", "sofa_attributes": sofa["meta"]},
           "mesh": mst, "receiver_m": rec.tolist(),
           "patch": {"radius_m": project.EAR_RADIUS, "n_faces": pinfo["n_faces"], "area_mm2": pinfo["area"] * 1e6,
                     "centroid_offset_mm": float(np.linalg.norm(pinfo["centroid"] - rec) * 1e3)},
           "rotation_meas_to_mesh": R3.tolist(), "rotation_angle_deg":
               float(np.degrees(np.arccos((np.trace(R3) - 1) / 2))),
           "n_measured_directions": int(P), "floor_db": FLOOR, "phase_mask_db": PHASE_DB,
           "freqs_khz": (freqs / 1e3).tolist(), "m2h_available": has.tolist(),
           "per_frequency": {k: tolist(v) for k, v in res.items()},
           "summary": {k: summary(v) for k, v in res.items() if np.isfinite(v["corr"]).any()},
           "broadband": {k: {kk: float(vv[0]) for kk, vv in v.items()} for k, v in broadband.items()},
           "m2h_phase_convention_match": phase_conv}
    s = out["summary"]["ours_vs_m2h"] if "ours_vs_m2h" in out["summary"] else None
    if s is not None:
        for k in cplx:
            s[k + "_median"] = float(np.nanmedian(cplx[k]))
    fn = paths.RESULTS / "validation" / ("vesterholm_coarse.json" if args.coarse else "vesterholm.json")
    fn.write_text(json.dumps(out, indent=1))

    # dB maps on the measurement grid [f, el, az] (measurement frame) for the figure
    g = meas["gmap"]
    db = lambda X: 20 * np.log10(X / np.nanmax(X, 0))
    np.savez(work / f"maps_{tag}.npz", freqs=freqs, grid_az=meas["grid_az"], grid_el=meas["grid_el"],
             measured=db(Mm)[g].transpose(2, 0, 1), ours=db(np.abs(H_rot))[g].transpose(2, 0, 1),
             m2h=db(S_meas)[g].transpose(2, 0, 1))
    print(f"-> {fn}")
    for k, v in out["summary"].items():
        print(f"{k:36s} corr mean {v['corr_mean']:.3f} median {v['corr_median']:.3f} min {v['corr_min']:.3f}  "
              f"rms {v['rms_db_mean']:.2f} dB  ({v['n_freqs']} freqs)")
    for k, v in out["broadband"].items():
        print(f"broadband {k:26s} corr {v['corr']:.3f} rms {v['rms_db']:.2f} dB peak {v['peak_deg']:.1f} deg")
    if s is not None:
        print("ours vs Mesh2HRTF complex:", {k: round(s[k + '_median'], 3) for k in cplx}, "phase match:", phase_conv)


if __name__ == "__main__":
    main()
