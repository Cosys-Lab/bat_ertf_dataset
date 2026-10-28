"""Create a Zenodo DRAFT of the dataset record and upload the packed files (scripts/zenodo_package.py). It never
publishes: check the draft in the web interface and press Publish there.

Token: environment variable ZENODO_TOKEN (personal access token with the scopes deposit:write and deposit:actions;
create it at https://zenodo.org/account/settings/applications/, or at https://sandbox.zenodo.org/... for tests).
Default target is the Zenodo SANDBOX (a test instance, DOIs are not real). Use --production for the real record;
then the metadata must not contain any 'TODO'.
Usage: python scripts/zenodo_upload.py [--production] [--upload-dir DIR] [--metadata docs/zenodo_metadata.json]
       python scripts/zenodo_upload.py --check     (validate metadata and files, upload nothing)
Writes the draft's id and links to <upload-dir>/DRAFT.json. Uploads resume per file (files already in the draft with
the same MD5 are skipped), so the script can be rerun after an interruption."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import paths

HOSTS = {"sandbox": "https://sandbox.zenodo.org", "production": "https://zenodo.org"}


def load_metadata(fn, production):
    meta = json.loads(Path(fn).read_text())
    meta = {k: v for k, v in meta.items() if not k.startswith("_")}
    todo = "TODO" in json.dumps(meta)
    if production and todo:
        sys.exit("metadata still contains TODO: fill in docs/zenodo_metadata.json before a production upload")
    if todo:
        print("note: metadata contains TODO placeholders (allowed for a sandbox draft)")
        meta["related_identifiers"] = [r for r in meta.get("related_identifiers", []) if "TODO" not in r["identifier"]]
    return meta


def md5(fn):
    h = hashlib.md5()
    with open(fn, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--production", action="store_true")
    ap.add_argument("--upload-dir", default=str(paths.work_dir() / "zenodo_upload"))
    ap.add_argument("--metadata", default=str(paths.ROOT / "docs" / "zenodo_metadata.json"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    up = Path(a.upload_dir)
    files = sorted(f for f in up.iterdir() if f.name not in ("MANIFEST.json", "DRAFT.json"))
    manifest = json.loads((up / "MANIFEST.json").read_text())
    meta = load_metadata(a.metadata, a.production)
    assert len(files) <= 100, "Zenodo allows at most 100 files per record"
    for f in files:
        assert f.name in manifest and manifest[f.name]["bytes"] == f.stat().st_size, f"{f.name} differs from MANIFEST"
    print(f"{len(files)} files, {sum(f.stat().st_size for f in files) / 1e9:.2f} GB, metadata OK")
    if a.check:
        return
    host = HOSTS["production" if a.production else "sandbox"]
    token = os.environ.get("ZENODO_TOKEN") or sys.exit("set ZENODO_TOKEN")
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    state = up / "DRAFT.json"
    draft = json.loads(state.read_text()) if state.exists() and json.loads(state.read_text()).get("host") == host else None
    if draft is None:
        r = s.post(f"{host}/api/deposit/depositions", json={})
        r.raise_for_status()
        dep = r.json()
        draft = {"host": host, "id": dep["id"], "bucket": dep["links"]["bucket"], "html": dep["links"]["html"]}
        state.write_text(json.dumps(draft, indent=1))
    have = {f["filename"]: f["checksum"] for f in s.get(f"{host}/api/deposit/depositions/{draft['id']}/files").json()}
    for f in files:
        if have.get(f.name) == manifest[f.name]["md5"]:
            print(f"{f.name}: already uploaded")
            continue
        with open(f, "rb") as fh:
            r = s.put(f"{draft['bucket']}/{f.name}", data=fh)
        r.raise_for_status()
        ok = r.json().get("checksum", "").replace("md5:", "") == manifest[f.name]["md5"]
        print(f"{f.name}: uploaded, MD5 {'OK' if ok else 'MISMATCH'}", flush=True)
    r = s.put(f"{host}/api/deposit/depositions/{draft['id']}", json={"metadata": meta})
    r.raise_for_status()
    print(f"draft ready (not published): {draft['html']}")


if __name__ == "__main__":
    main()
