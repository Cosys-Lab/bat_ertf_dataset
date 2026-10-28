"""Export the dataset for Zenodo: one folder per species with
  mesh.stl                      clean dataset mesh (CT frame, metres)
  mesh_<frame>.stl              the same mesh in the coordinates of ertf_<frame>.h5 (origin = centre of the
                                evaluation sphere, X = azimuth 0, Y = azimuth +90 deg, Z = elevation +90 deg)
  ertf_<frame>.h5               all sources (nominal + placement ensemble), complex H [el, az, f], metadata
  ears_<frame>.sofa             nominal left/right ear, SOFA SimpleFreeFieldHRTF (frequency domain, 2 receivers)
  emitter_<frame>.sofa          nominal emitter, SOFA FreeFieldDirectivityTF
  landmarks.json                landmarks, placement ensemble, frame, conventions
and README.md (docs/zenodo_README.md), meshes.json and species.json at the top, and code/ (the user API
batertf/dataset.py and examples/, Python and MATLAB). Only complete production files are exported.
Usage: python scripts/export_dataset.py [out_dir]   (default <work>/zenodo)"""
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import sofar as sf
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import grid, paths
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ertf

CONVENTION = ("exp(+j w t) convention. H = p / p_ff at r = 1 m, with p_ff = j w rho Q exp(-jkr) / (4 pi r) the pressure of "
              "a free-field monopole at the origin with the source's volume velocity Q (equivalently, conj(p / p_ff) of the "
              "solver's exp(-j w t) result, as in Mesh2HRTF). Phase is relative to the origin: an inverse real FFT over "
              "frequency gives an impulse response with t = 0 at the origin, which starts slightly before t = 0 for "
              "directions on the source's side (up to |p|/c), so add a small pre-delay. Ears by reciprocity (vibrating patch at the ear canal "
              "entrance); nasal emitters: two in-phase nostril patches; oral emitters: analytical piston.")


REFERENCE = ("Full-sphere emission and reception directivities of seven echolocating bat species, "
             "Scientific Data (submitted); dataset DOI: 10.5281/zenodo.23729269")


def sofa_from(H, f, az, el, name, kind, sp, frame, ears_m=None):
    A, E = np.meshgrid(az, el)
    pos = np.stack([A.ravel(), E.ravel(), np.ones(A.size)], 1)       # [M, 3] azimuth, elevation, 1 m
    if kind == "ears":
        s = sf.Sofa("SimpleFreeFieldHRTF")
        s.Data_Real = np.stack([H[0].reshape(-1, len(f)).real, H[1].reshape(-1, len(f)).real], 1)
        s.Data_Imag = np.stack([H[0].reshape(-1, len(f)).imag, H[1].reshape(-1, len(f)).imag], 1)
        s.SourcePosition = pos
        s.SourcePosition_Type, s.SourcePosition_Units = "spherical", "degree, degree, metre"
        if ears_m is not None:                          # receivers at the ear landmarks, frame coordinates
            s.ReceiverPosition = np.asarray(ears_m)[:, :, None] if s.ReceiverPosition.ndim == 3 else np.asarray(ears_m)
    else:                                               # directivity: one measurement, directions as receivers
        s = sf.Sofa("FreeFieldDirectivityTF")
        s.Data_Real = H.reshape(-1, len(f))[None].real
        s.Data_Imag = H.reshape(-1, len(f))[None].imag
        s.ReceiverPosition = pos
        s.ReceiverPosition_Type, s.ReceiverPosition_Units = "spherical", "degree, degree, metre"
        for k in ("Description", "EmitterDescriptions"):   # optional string arrays: sofar 1.3 cannot write them
            s.delete(k)                                      # with numpy 2; the emitter type goes in the comment
        s.GLOBAL_Comment = ("Emitter: " + ("two in-phase nostril patches (BEM). " if sp["emitter"] == "nasal" else
                                           "analytical piston at the mouth. ") + CONVENTION)
    s.N = f
    s.GLOBAL_Title = f"{sp['name']} {name} ({frame} frame)"
    if kind == "ears":
        s.GLOBAL_Comment = CONVENTION
    if kind == "ears":
        s.GLOBAL_ListenerShortName = sp["name"]
    s.GLOBAL_DatabaseName = "Bat ERTF Meshes"
    s.GLOBAL_License = "CC BY 4.0"
    s.GLOBAL_References = REFERENCE
    s.GLOBAL_History = "Computed with the GPU boundary element code of the dataset (or analytical piston), see References"
    s.GLOBAL_Organization = "Cosys-Lab, University of Antwerp, and Flanders Make"
    s.GLOBAL_AuthorContact = "Jan Steckel, jan.steckel@uantwerpen.be"
    return s


def write_mat(fn, h, sp, landmarks):
    """The nominal sources in the lab's older MATLAB format (struct bat_hrtf as in the reference files of De Mey et
    al. 2008 and Vanderelst et al. 2010): hrtf_left, hrtf_right, hrtf_em [el x az x f], az_vec, el_vec, freq_vec,
    fmin, fmax. Differences: full sphere (azimuth -180..177.5) and complex values (single), so abs() gives the
    magnitudes of the older files; plus species, landmarks and the conventions."""
    import scipy.io as sio
    f = h["freqs"][:]
    lm = {k: np.atleast_2d(np.asarray(v, float)) if k == "nostrils" else np.asarray(v, float)
          for k, v in landmarks.items()}
    b = {"hrtf_left": h["left"]["H"][:], "hrtf_right": h["right"]["H"][:], "hrtf_em": h["emitter"]["H"][:],
         "az_vec": h["azimuth"][:].astype(float), "el_vec": h["elevation"][:].astype(float), "freq_vec": f.astype(float),
         "fmin": float(f[0]), "fmax": float(f[-1]), "species": sp["taxon"], "file_species": sp["name"],
         "emitter": "two in-phase nostril patches (BEM)" if sp["emitter"] == "nasal" else
                    f"analytical piston at the mouth, radius {sp['piston_radius_m'] * 1e3:.2f} mm",
         "landmarks_m": lm, "convention": CONVENTION,
         "dimensions": "hrtf_*: [elevation x azimuth x frequency], complex; az + = left, el + = up, frame X forward, "
                       "Y left, Z up (metres)"}
    sio.savemat(fn, {"bat_hrtf": b}, do_compression=False)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else paths.work_dir() / "zenodo"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((paths.ROOT / "data" / "meshes" / "dataset" / "meshes.json").read_text())
    species = {}
    for key, sp in paths.species().items():
        files = sorted((paths.work_dir() / "ertf").glob(f"{key}_*.h5"))
        files = [f for f in files if h5py.File(f)["done"][:].all()]
        if not files:
            print(f"{key}: no complete production file, skipped")
            continue
        d = out / sp["name"].replace(" ", "_")
        d.mkdir(exist_ok=True)
        shutil.copy(paths.ROOT / "data" / "meshes" / "dataset" / meta[key]["file"], d / "mesh.stl")
        info = {"species": sp["taxon"], "file_species": sp["name"], "taxon": sp["taxon"], "family": sp["family"], "emitter": sp["emitter"], "mesh": meta[key],
                "landmarks_ct_frame_m": sp["landmarks"], "frames": {}, "convention": CONVENTION}
        if sp["emitter"] == "oral":
            info["piston"] = {"radius_m": sp["piston_radius_m"], "calibration": sp["piston_calibration"],
                              "alternatives": [{"source": f"emitter_alt{i + 1}", "radius_m": a, "note": n} for i, (a, n) in
                                               enumerate(zip(sp.get("piston_alternatives_m", []),
                                                             sp.get("piston_alternatives_note", [])))]}
        dist_f = paths.RESULTS / "validation" / "alt_distances.json"          # scripts/alt_distances.py
        if dist_f.exists():
            dist = json.loads(dist_f.read_text()).get(key, {})
            info["alternative_distance_to_nominal_mm"] = {
                row["source"]: row.get("distance_mm", row.get("midpoint_distance_mm"))
                for rows in dist.values() for row in rows}
        for fn in files:
            frame = fn.stem.split("_", 1)[1]
            shutil.copy(fn, d / f"ertf_{frame}.h5")
            with h5py.File(d / f"ertf_{frame}.h5", "a") as h:     # release copy: no progress flag, JSON metadata
                del h["done"]
                lmf = {k: (np.asarray(v) @ np.asarray(h.attrs["R"])).tolist() for k, v in sp["landmarks"].items()}
                h.attrs["landmarks_frame_m"] = json.dumps(lmf)
                h.attrs["taxon"] = sp["taxon"]
                h.attrs["poles"] = "elevation +-90 deg rows repeat the single pole value for every azimuth"
            with h5py.File(fn) as h:
                R = np.asarray(h.attrs["R"])
                mf = trimesh.load(paths.ROOT / "data" / "meshes" / "dataset" / meta[key]["file"], process=False)
                mf.vertices = mf.vertices @ R                # v_frame = R^T v_ct: the mesh as the solver saw it
                mf.export(d / f"mesh_{frame}.stl")
                info["frames"].setdefault(frame, {})["landmarks_m"] = {
                    k: (np.asarray(v) @ R).tolist() for k, v in sp["landmarks"].items()}
                info["frames"][frame]["alternative_positions_m"] = {     # sources <name>_alt<i>, in order
                    k: [(np.asarray(a) @ R).tolist() for a in v] for k, v in run_ertf.ensemble(key, sp["landmarks"]).items()}
                f, az, el = h["freqs"][:], h["azimuth"][:], h["elevation"][:]
                info["frames"][frame].update({"R_frame_to_ct": R.tolist(), "mesh": f"mesh_{frame}.stl",
                                         "sources": [n for n in h if isinstance(h[n], h5py.Group)],
                                         "freq_hz": [float(f[0]), float(f[-1]), float(f[1] - f[0])]})
                sf.write_sofa(str(d / f"ears_{frame}.sofa"),
                              sofa_from(np.stack([h["left"]["H"][:], h["right"]["H"][:]]), f, az, el, "ears", "ears", sp, frame,
                                        ears_m=[info["frames"][frame]["landmarks_m"]["ear_left"],
                                                info["frames"][frame]["landmarks_m"]["ear_right"]]))
                sf.write_sofa(str(d / f"emitter_{frame}.sofa"), sofa_from(h["emitter"]["H"][:], f, az, el, "emitter",
                                                                          "emitter", sp, frame))
                if frame == "ct":
                    write_mat(d / f"HRTF_{d.name}.mat", h, sp, info["frames"][frame]["landmarks_m"])
        (d / "landmarks.json").write_text(json.dumps(info, indent=1))
        rd = d / "rig"                                   # posable model (scripts/build_rigs.py)
        rd.mkdir(exist_ok=True)
        for src, dst in ((paths.RESULTS / "rig" / f"{key}_rig.json", "rig.json"),
                         (paths.RESULTS / "rig" / f"{key}_weights.npz", "weights.npz"),
                         (paths.work_dir() / "rig" / f"{key}.glb", f"{d.name}.glb")):
            if src.exists():
                shutil.copy(src, rd / dst)
        species[key] = {"folder": d.name, **{k: info[k] for k in ("species", "taxon", "family", "emitter")}}
        print(f"{key}: exported {[f.name for f in files]}")
    shutil.copy(paths.ROOT / "docs" / "zenodo_README.md", out / "README.md")
    code = out / "code"                                  # user API + examples (same layout as the repository)
    (code / "batertf").mkdir(parents=True, exist_ok=True)
    shutil.copy(paths.ROOT / "batertf" / "dataset.py", code / "batertf" / "dataset.py")
    (code / "batertf" / "__init__.py").write_text('"""Bat ERTF dataset: user API in dataset.py."""\n')
    shutil.copytree(paths.ROOT / "examples", code / "examples", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.png"))
    for key, v in meta.items():                          # valid name, file names keep the spelling of the CT files
        if key in paths.species():
            v["species"] = paths.species()[key]["taxon"]
    (out / "meshes.json").write_text(json.dumps(meta, indent=1))
    (out / "species.json").write_text(json.dumps(species, indent=1))
    checksums(out)


def checksums(out):
    """checksums.txt: SHA-256 and size of every file of the release."""
    import hashlib
    lines = []
    for f in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "checksums.txt"):
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        lines.append(f"{h.hexdigest()}  {f.stat().st_size:>12d}  {f.relative_to(out).as_posix()}")
    (out / "checksums.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
