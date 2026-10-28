"""Pose grids: ERTFs of the rigged dataset meshes in a set of pinna and noseleaf poses, computed by any number of GPU
workers on any machine through a job queue on shared storage (optional tool for users, see examples/POSING.md; the
dataset itself contains only the rest pose). Edit FLEX, ABDUCT, NOSELEAF and STEP_KHZ below to choose the poses.

Poses per species (POSES): both pinnae flexed and abducted together about their base joints on a grid, and, for the
nasal species, the noseleaf tilted about its base, the rest pose included. Frequencies: from 20 kHz to the top of the
band of the species in STEP_KHZ steps. Sources: left ear, right ear, emitter (nominal landmarks, deformed with the
mesh; oral species: the calibrated piston on the deformed mouth landmark, as in run_ertf.write_piston).
A job is (species, pose, block of BLOCK frequencies). Workers claim a job by creating <queue>/locks/<job>.lock
exclusively, write <queue>/results/<job>.npz (via a temporary name), and skip jobs that have a result. Locks older
than STALE_H hours without a result are reclaimed, so killed workers do not block the queue.

Usage:
  python scripts/pose_atlas.py plan   [--queue DIR]          write the job list (idempotent)
  python scripts/pose_atlas.py work   [--queue DIR] [--device cuda:0] [--species key ...]
  python scripts/pose_atlas.py status [--queue DIR]
  python scripts/pose_atlas.py merge  [--queue DIR]          -> <queue>/atlas/<key>_poses.h5
Default queue: <work>/../pose_atlas, or $BATERTF_POSE_QUEUE (shared drive, e.g. /mnt/CosysAgents/BatERTF_pose_atlas)."""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batertf import grid, paths, piston, project, rig, sources

ORDER = ["pdisc", "mmicr", "mmacr", "rrouxi", "rform", "efusc", "mdaub"]
FLEX = [-20, -10, 0, 10, 20]
ABDUCT = [-10, 0, 10]
NOSELEAF = [-10, 10]
STEP_KHZ = 2.5
BLOCK = 8
STALE_H = 2.0
SRC = ("left", "right", "emitter")


def poses(key):
    """{name: pose dict} for one species."""
    out = {}
    for f in FLEX:
        for a in ABDUCT:
            name = "rest" if f == 0 and a == 0 else f"pinna_flex{f:+d}_abduct{a:+d}"
            out[name] = {} if name == "rest" else {"pinna_both.base": {"flex": float(f), "abduct": float(a)}}
    if paths.species()[key]["emitter"] == "nasal":
        for n in NOSELEAF:
            out[f"noseleaf_flex{n:+d}"] = {"noseleaf.base": {"flex": float(n)}}
    return out


def freqs(key):
    band = paths.species()[key]["band_khz"]
    return np.round(np.arange(band[0], band[1] + 1e-6, STEP_KHZ), 3) * 1e3


def queue_dir(arg):
    q = Path(arg or os.environ.get("BATERTF_POSE_QUEUE") or paths.work_dir().parent / "pose_atlas")
    for d in ("locks", "results", "atlas"):
        (q / d).mkdir(parents=True, exist_ok=True)
    return q


def plan(q):
    jobs = []
    for key in ORDER:
        f = freqs(key)
        for name, pose in poses(key).items():
            for b in range(0, len(f), BLOCK):
                jobs.append({"id": f"{key}__{name}__{b // BLOCK:02d}", "key": key, "pose_name": name, "pose": pose,
                             "freqs": f[b:b + BLOCK].tolist()})
    (q / "jobs.json").write_text(json.dumps({"step_khz": STEP_KHZ, "block": BLOCK, "jobs": jobs}, indent=0))
    print(f"{len(jobs)} jobs, {sum(len(j['freqs']) for j in jobs)} factorizations -> {q / 'jobs.json'}")


def claim(q, job):
    res, lock = q / "results" / f"{job['id']}.npz", q / "locks" / f"{job['id']}.lock"
    if res.exists():
        return False
    if lock.exists() and time.time() - lock.stat().st_mtime > STALE_H * 3600:
        try:
            lock.unlink()                               # stale: the worker died
        except OSError:
            return False
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{socket.gethostname()} {os.getpid()} {time.ctime()}\n".encode())
    os.close(fd)
    return True


class Model:
    """Deformed mesh and source patches of one (species, pose), cached across frequency blocks."""

    def __init__(self, key, pose):
        self.key, self.pose = key, pose
        self.sp = paths.species()[key]
        r = rig.load_rig(key)
        self.m = r.deform(pose)
        self.check = r.check(self.m, pose)
        lm = r.deform_landmarks(pose, self.m)
        self.lm = lm
        self.patches = {"left": [sources.disc_patch(self.m, lm["ear_left"], project.EAR_RADIUS)],
                        "right": [sources.disc_patch(self.m, lm["ear_right"], project.EAR_RADIUS)]}
        if self.sp["emitter"] == "nasal":
            self.patches["emitter"] = [sources.disc_patch(self.m, lm[k], project.NOSTRIL_RADIUS)
                                       for k in ("nostril_0", "nostril_1")]

    def solve(self, f, device):
        from batertf import gpubem
        pts, _ = grid.unique_points("sphere")
        H = gpubem.solve_lu(self.m, self.patches, f, pts, device=device, log=lambda s: None)
        out = {s: gpubem.to_grid(H[s], "sphere") for s in self.patches}
        if self.sp["emitter"] == "oral":                # piston on the deformed mouth landmark (pose moves no lips)
            mouth = np.asarray(self.sp["landmarks"]["mouth"])
            axis = piston.mouth_axis(self.m, mouth)
            out["emitter"] = piston.transfer(grid.unit_vectors(*grid.GRIDS["sphere"]), f, mouth, axis,
                                             self.sp["piston_radius_m"])
        return np.stack([out[s] for s in SRC]).astype(np.complex64)


def work(q, device, only):
    jobs = json.loads((q / "jobs.json").read_text())["jobs"]
    if only:
        jobs = [j for j in jobs if j["key"] in only]
    cache = {}
    while True:
        todo = [j for j in jobs if not (q / "results" / f"{j['id']}.npz").exists()]
        if not todo:
            print("queue empty", flush=True)
            return
        did = False
        for job in todo:
            if not claim(q, job):
                continue
            t0 = time.time()
            ck = (job["key"], job["pose_name"])
            if ck not in cache:
                cache = {ck: Model(job["key"], job["pose"])}  # one model at a time keeps memory low
            mdl = cache[ck]
            f = np.array(job["freqs"])
            H = mdl.solve(f, device) if mdl.check["valid"] else np.zeros((3, len(grid.EL), 144, len(f)), np.complex64)
            tmp = q / "results" / f"{job['id']}.tmp.npz"
            np.savez(tmp, H=H, freqs=f, valid=mdl.check["valid"], check=json.dumps(mdl.check, default=str),
                     host=socket.gethostname(), seconds=time.time() - t0)
            os.replace(tmp, q / "results" / f"{job['id']}.npz")
            (q / "locks" / f"{job['id']}.lock").unlink(missing_ok=True)
            print(f"{socket.gethostname()} {job['id']}: valid {mdl.check['valid']}, {time.time() - t0:.0f} s", flush=True)
            did = True
            break                                       # re-read the queue: other workers may have advanced
        if not did:
            time.sleep(30)                              # everything left is claimed by others


def status(q):
    jobs = json.loads((q / "jobs.json").read_text())["jobs"]
    done = {p.stem for p in (q / "results").glob("*.npz") if not p.name.endswith(".tmp.npz")}
    locks = list((q / "locks").glob("*.lock"))
    per = {}
    for j in jobs:
        per.setdefault(j["key"], [0, 0])[0] += 1
        per[j["key"]][1] += j["id"] in done
    hosts = {}
    for p in (q / "results").glob("*.npz"):
        try:
            h = str(np.load(p)["host"])
            hosts[h] = hosts.get(h, 0) + 1
        except Exception:
            pass
    print(f"{sum(v[1] for v in per.values())}/{len(jobs)} jobs done, {len(locks)} running; per species "
          + ", ".join(f"{k} {v[1]}/{v[0]}" for k, v in per.items()) + f"; by host {hosts}")


def merge(q):
    import h5py
    jobs = json.loads((q / "jobs.json").read_text())["jobs"]
    az, el = grid.GRIDS["sphere"]
    for key in ORDER:
        f = freqs(key)
        names = list(poses(key))
        r = rig.load_rig(key)
        fn = q / "atlas" / f"{key}_poses.h5"
        with h5py.File(fn, "w") as h:
            h["freqs"], h["azimuth"], h["elevation"] = f, az, el
            h.attrs.update(species=paths.species()[key]["name"], taxon=paths.species()[key]["taxon"],
                           sources=json.dumps(SRC), step_khz=STEP_KHZ,
                           convention="as ertf_ct.h5: exp(+j w t), H = p / p_monopole(origin), 1 m, CT frame")
            for name in names:
                mine = sorted((j for j in jobs if j["key"] == key and j["pose_name"] == name), key=lambda j: j["id"])
                parts = [np.load(q / "results" / f"{j['id']}.npz") for j in mine
                         if (q / "results" / f"{j['id']}.npz").exists()]
                if len(parts) < len(mine):
                    print(f"{key} {name}: {len(parts)}/{len(mine)} blocks, skipped")
                    continue
                g = h.create_group(name)
                H = np.concatenate([p["H"] for p in parts], -1)
                for k, s in enumerate(SRC):
                    g[s] = H[k]
                g.attrs["pose"] = json.dumps(mine[0]["pose"])
                g.attrs["valid"] = bool(parts[0]["valid"])
                g.attrs["check"] = str(parts[0]["check"])
                g["vertices"] = r.deform(mine[0]["pose"]).vertices.astype(np.float32)   # faces as mesh.stl
        print(fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "work", "status", "merge"])
    ap.add_argument("--queue")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--species", nargs="*")
    a = ap.parse_args()
    q = queue_dir(a.queue)
    {"plan": lambda: plan(q), "work": lambda: work(q, a.device, a.species), "status": lambda: status(q),
     "merge": lambda: merge(q)}[a.cmd]()


if __name__ == "__main__":
    main()
