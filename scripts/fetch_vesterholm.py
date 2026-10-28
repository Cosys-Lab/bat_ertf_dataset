"""Fetch the Mdau2 files of Vesterholm et al. 2025 (Zenodo doi:10.5281/zenodo.15789942, CC-BY 4.0) without
downloading the whole 593 MB archive: the zip's central directory and the selected members are read with HTTP range
requests. Files land in <dest>/files/<member path>; a listing of the archive goes to <dest>/listing.json.
Usage: python -I scripts/fetch_vesterholm.py <dest> [--list]   (dest: e.g. /data/BatERTF/external/vesterholm2025)"""
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

RECORD = "https://zenodo.org/api/records/15789942"
# members needed for the Mdau2 validation (regular expressions on the archive path)
# (the archive has no author README; the conventions are in the MATLAB scripts next to the data)
WANT = [r"Mdau2_CTscan\.ply$", r"Measured HRTF/Mdau2[^/]*\.mat$", r"SOFA files/Mdau2_CTscan\.sofa$",
        r"(Measured HRTF|SOFA files)/[^/]*\.m$"]


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
            self.url = r.url                         # follow redirects once

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def main():
    dest = Path(sys.argv[1])
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RECORD) as r:
        rec = json.load(r)
    f = rec["files"][0]
    raw = io.BufferedReader(HTTPRangeFile(f["links"]["self"]), buffer_size=1 << 20)
    z = zipfile.ZipFile(raw)
    listing = [{"name": i.filename, "size": i.file_size, "compressed": i.compress_size, "crc": f"{i.CRC:08x}"}
               for i in z.infolist()]
    (dest / "listing.json").write_text(json.dumps({"record": RECORD, "doi": rec.get("doi"),
                                                    "licence": rec["metadata"].get("license"),
                                                    "archive": f["key"], "archive_size": f["size"],
                                                    "archive_checksum": f["checksum"], "members": listing}, indent=1))
    if "--list" in sys.argv:
        for e in listing:
            print(f"{e['size']:>12d}  {e['name']}")
        return
    for i in z.infolist():
        if i.is_dir() or not any(re.search(p, i.filename) for p in WANT):
            continue
        out = dest / "files" / i.filename
        if out.exists() and out.stat().st_size == i.file_size:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        data = z.read(i)                              # zipfile checks the CRC
        out.write_bytes(data)
        print(f"{i.file_size:>12d}  {i.filename}  sha256 {hashlib.sha256(data).hexdigest()[:16]}", flush=True)


if __name__ == "__main__":
    main()
