"""Pack the exported dataset (scripts/export_dataset.py) for upload to Zenodo.

Zenodo allows at most 100 files per record, and the unpacked dataset has more, so every species folder becomes one zip
(stored, not compressed: HDF5/SOFA of complex data barely compress), code/ becomes code.zip, and the top-level files are
copied as they are (README.md, checksums.txt of the unpacked files, species.json, meshes.json). Also writes
MANIFEST.json with size and MD5 of every upload file (Zenodo reports MD5s, so the upload can be checked).
Usage: python scripts/zenodo_package.py [export_dir] [upload_dir]
       (defaults: <work>/zenodo and <work>/zenodo_upload)"""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import paths

TOP = ("README.md", "checksums.txt", "species.json", "meshes.json")


def md5(fn):
    h = hashlib.md5()
    with open(fn, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else paths.work_dir() / "zenodo"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else paths.work_dir() / "zenodo_upload"
    if not (src / "species.json").exists():
        sys.exit(f"{src} is not an export (no species.json): run scripts/export_dataset.py first")
    out.mkdir(parents=True, exist_ok=True)
    for f in out.iterdir():                              # rebuild from scratch
        f.unlink()
    for name in TOP:
        shutil.copy(src / name, out / name)
    folders = sorted(json.loads((src / "species.json").read_text()).values(), key=lambda v: v["folder"])
    for d in [src / v["folder"] for v in folders] + [src / "code"]:
        zf = out / f"{d.name}.zip"
        with zipfile.ZipFile(zf, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for f in sorted(p for p in d.rglob("*") if p.is_file()):
                z.write(f, f.relative_to(src).as_posix())       # paths inside start with the folder name
        print(f"{zf.name}: {zf.stat().st_size / 1e6:.0f} MB", flush=True)
    for zc in sorted((paths.work_dir() / "zenodo_code_upload").glob("bat_ertf_code-*.zip")):   # the complete code
        shutil.copy(zc, out / zc.name)                                                    # (scripts/zenodo_code_package.py)
    manifest = {f.name: {"bytes": f.stat().st_size, "md5": md5(f)} for f in sorted(out.iterdir())
                if f.name != "MANIFEST.json"}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    total = sum(v["bytes"] for v in manifest.values())
    print(f"{len(manifest)} files, {total / 1e9:.2f} GB -> {out}")


if __name__ == "__main__":
    main()
