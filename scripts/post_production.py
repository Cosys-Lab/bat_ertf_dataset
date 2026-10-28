"""After the production run: the GPU validation runs, then all figures, tables, the dataset export and the paper.
Waits until scripts/run_production.py has exited, then runs the STEPS in order (a missing script is skipped, a failing
step is logged and the next one runs). Log: <work>/post_production.log.
Usage: python scripts/post_production.py   (meant to run detached, e.g. with nohup)"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import paths

ROOT = paths.ROOT
STEPS = [["scripts/convergence_study.py"],
         ["scripts/validate_vesterholm.py"],
         ["scripts/sensitivity_checks.py"],
         ["paper/figures/fig_validation.py"],
         ["paper/figures/fig_reference_metrics.py"],
         ["paper/figures/fig_ertf_species.py"],
         ["paper/figures/fig_ertf_full.py"],
         ["paper/figures/fig_directivity.py"],
         ["paper/figures/fig_convergence.py"],
         ["paper/figures/fig_sensitivity.py"],
         ["paper/figures/fig_vesterholm.py"],
         ["paper/figures/make_tables.py"],
         ["scripts/export_dataset.py"],
         ["scripts/make_video.py"]]


def running(name):
    """True while a process whose command line contains `name` (other than this one) is alive."""
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if name in cmd and "post_production" not in cmd:
            return True
    return False


def main():
    log = paths.work_dir() / "post_production.log"
    say = lambda s: log.open("a").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}\n")
    say("waiting for run_production.py")
    while running("run_production.py") or running("run_ertf.py"):
        time.sleep(60)
    for step in STEPS:
        if not (ROOT / step[0]).exists():
            say(f"{step[0]}: missing, skipped")
            continue
        t0 = time.time()
        with (paths.work_dir() / f"post_{Path(step[0]).stem}.log").open("w") as f:
            r = subprocess.run([sys.executable, *step], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                               env={**os.environ, "HDF5_USE_FILE_LOCKING": "FALSE"})
        say(f"{step[0]}: exit {r.returncode}, {(time.time() - t0) / 60:.1f} min")
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"], cwd=ROOT / "paper" / "scidata",
                       capture_output=True)
    say(f"paper: latexmk exit {r.returncode}")
    say("done")


if __name__ == "__main__":
    main()
