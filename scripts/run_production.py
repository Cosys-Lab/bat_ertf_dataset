"""Run the whole production, coarse to fine (every 8th, 4th, 2nd frequency, then all; each pass over all species): every species in the CT frame, then the validation run of P. discolor in the frame of
its reference. Waits for running placement-rule jobs first, and skips species whose output file is complete
(run_ertf.py itself resumes partly finished files). One log per run in <work>/prod_<key>_<frame>.log, a summary in
<work>/production.log.
Usage: python scripts/run_production.py   (meant to run detached, e.g. with nohup)"""
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import paths

STRIDES = [8, 4, 2, 1]      # every 8th frequency (4 kHz) for all species first, then refine
PRE = [("results/placement/rform_rule.json", ["scripts/apply_placement_rule.py", "rform"]),
       ("results/validation/gpubem_vs_numcalc_ct_mmicr.txt", ["tests/gpubem_vs_numcalc_ct.py", "mmicr", "--gpu"])]
RUNS = [("mmicr", "ct", None), ("pdisc", "ct", None), ("mmacr", "ct", None), ("rrouxi", "ct", None),
        ("rform", "ct", None), ("efusc", "ct", None), ("mdaub", "ct", None)]   # pdisc: its ct frame is the pose of De Mey et al. (scripts/rotate_pdisc_pose.py)


def busy():
    r = subprocess.run(["pgrep", "-f", "[a]pply_placement_rule.py"], capture_output=True)
    return r.returncode == 0


def complete(key, frame):
    f = paths.work_dir() / "ertf" / f"{key}_{frame}.h5"
    if not f.exists():
        return False
    with h5py.File(f) as h:
        return bool(h["done"][:].all())


def main():
    work = paths.work_dir()
    log = open(work / "production.log", "a")
    say = lambda s: (log.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n"), log.flush())
    while busy():
        time.sleep(30)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    for out_file, cmd in PRE:                           # one-off GPU jobs before production
        if not (paths.ROOT / out_file).exists():
            say(f"pre: {' '.join(cmd)}")
            with open(work / "production_pre.log", "a") as out:
                r = subprocess.run([sys.executable] + cmd, cwd=paths.ROOT, stdout=out, stderr=subprocess.STDOUT, env=env)
            say(f"pre: exit {r.returncode}")
    for stride in STRIDES:                              # coarse-to-fine over all species
        for key, frame, band in RUNS:
            if complete(key, frame):
                continue
            say(f"{key} {frame} stride {stride}: start")
            t = time.time()
            with open(work / f"prod_{key}_{frame}.log", "a") as out:
                r = subprocess.run([sys.executable, str(Path(__file__).parent / "run_ertf.py"), key, frame] + (band or [])
                                   + ["--stride", str(stride)], stdout=out, stderr=subprocess.STDOUT, env=env)
            say(f"{key} {frame} stride {stride}: exit {r.returncode}, complete {complete(key, frame)}, "
                f"{(time.time() - t) / 60:.0f} min")
    say("all done")


if __name__ == "__main__":
    main()
