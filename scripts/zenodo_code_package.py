"""Clean code release, for GitHub and for the Zenodo record: the committed state (git HEAD) of the code, the technical
documentation, the dataset meshes, and the rig, placement and validation results that the scripts read, with the
public README (docs/github_README.md). Internal working notes (publication plan, reviews, summaries) are left out.
Usage: python scripts/zenodo_code_package.py [version] [out_dir]
       -> <out_dir>/bat_ertf_dataset/ (tree) and <out_dir>/bat_ertf_code-<version>.zip
       (default version 1.0.0, out_dir <work>/zenodo_code_upload)"""
import io
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import paths

INCLUDE = ["batertf", "scripts", "tests", "examples", "config", "docs", "data/meshes/dataset", "results/rig",
           "results/validation", "results/placement", "requirements.txt", "CITATION.cff", ".zenodo.json", "LICENSE",
           ".gitignore"]
EXCLUDE = ["docs/publication_plan.md", "docs/reviews", "docs/summary_2026-10-28.md", "docs/related_data_survey.md",
           "docs/github_README.md", "docs/zenodo_metadata.json", "docs/zenodo_metadata.md"]


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else "1.0.0"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else paths.work_dir() / "zenodo_code_upload"
    out.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=paths.ROOT, capture_output=True, check=True).stdout
    present = [p for p in INCLUDE if (paths.ROOT / p).exists()]
    if git("status", "--porcelain", "--", *present).strip():
        print("warning: uncommitted changes in the included paths are not in the release (it is built from HEAD)")
    files = [f for f in git("ls-files", "--", *present).decode().splitlines()
             if not any(f == e or f.startswith(e + "/") for e in EXCLUDE)]
    tree = out / "bat_ertf_dataset"
    if tree.exists():
        shutil.rmtree(tree)
    zf = out / f"bat_ertf_code-{version}.zip"
    with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files + ["README.md"]:
            src = "docs/github_README.md" if f == "README.md" else f
            data = git("show", f"HEAD:{src}")
            (tree / f).parent.mkdir(parents=True, exist_ok=True)
            (tree / f).write_bytes(data)
            z.writestr(f"bat_ertf_dataset/{f}", data)
    commit = git("rev-parse", "--short", "HEAD").decode().strip()
    print(f"{tree} and {zf.name}: {len(files) + 1} files, {zf.stat().st_size / 1e6:.1f} MB, commit {commit}")


if __name__ == "__main__":
    main()
