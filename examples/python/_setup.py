"""Shared by the examples: import path of the dataset API and the species folder from the command line.
Layout: <repo>/examples/python (GitHub) or <dataset>/code/examples/python (Zenodo); the API is <..>/batertf/dataset.py."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from batertf import dataset as bd  # noqa: E402

DEFAULT = ROOT.parent / "Phyllostomus_discolor"      # Zenodo layout: <dataset>/code/.. holds the species folders


def args(doc, **extra):
    p = argparse.ArgumentParser(description=doc)
    p.add_argument("species", nargs="?", default=str(DEFAULT), help="species folder of the dataset (default: %(default)s)")
    p.add_argument("--out", help="save the figure to this file instead of showing it")
    for k, v in extra.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    a = p.parse_args()
    if not Path(a.species).is_dir():
        p.error(f"species folder not found: {a.species}")
    return a


def finish(fig, a):
    import matplotlib.pyplot as plt
    if a.out:
        fig.savefig(a.out, dpi=120)
        print("saved", a.out)
    else:
        plt.show()
