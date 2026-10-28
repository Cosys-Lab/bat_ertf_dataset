"""Project paths. Machine-specific locations (work dir, NumCalc binary) come from config/local.toml."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MESH_DIR = ROOT / "data" / "meshes" / "original"
REF_DIR = ROOT / "data" / "reference"
RESULTS = ROOT / "results"


def local():
    p = ROOT / "config" / "local.toml"
    cfg = tomllib.loads(p.read_text()) if p.exists() else {}
    paths = cfg.get("paths", {})
    return {"work_dir": Path(paths.get("work_dir", ROOT / "work")),
            "numcalc": Path(paths.get("numcalc", "NumCalc"))}


def work_dir():
    w = local()["work_dir"]
    w.mkdir(parents=True, exist_ok=True)
    return w


def species():
    """Species config. "name" is the name used in the mesh files (traceability); "taxon" the verified scientific name
    for publication (defaults to "name")."""
    sp = tomllib.loads((ROOT / "config" / "species.toml").read_text())["species"]
    for s in sp.values():
        s.setdefault("taxon", s["name"])
    return sp
