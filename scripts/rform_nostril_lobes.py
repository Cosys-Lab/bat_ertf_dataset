"""Lobe structure of the R. formosae emitter for the nostril pairs of the placement rule, at 36-43 kHz (the range of
its calls): main-lobe direction and the level of the strongest other lobe (local maximum more than 30 deg from the
main lobe), in dB re the main lobe. Vanderelst et al. 2013 (Fig. 3A) found a single main lobe for this noseleaf.
The nominal pair of the dataset is the rule's second choice (in the nostril openings), the rule's first choice is
emitter_alt1 (on the internarial region at the base of the sella).
Usage: python scripts/rform_nostril_lobes.py   -> results/validation/rform_nostril_lobes.json"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import paths

FREQS = (36e3, 40e3, 43e3)


def main():
    out = {"what": __doc__.split("\n")[0], "freqs_khz": [f / 1e3 for f in FREQS], "sources": {}}
    with h5py.File(paths.work_dir() / "ertf" / "rform_ct.h5", "r") as h:
        f, az, el = h["freqs"][:], h["azimuth"][:], h["elevation"][:]
        A0, E0 = np.meshgrid(np.radians(az), np.radians(el[1:-1]))
        U = np.stack([np.cos(E0) * np.cos(A0), np.cos(E0) * np.sin(A0), np.sin(E0)], -1)
        for src, label in (("emitter", "nominal (rule's second choice)"), ("emitter_alt1", "rule's first choice")):
            rows = []
            for fk in FREQS:
                k = int(np.argmin(abs(f - fk)))
                A = 20 * np.log10(np.abs(h[src]["H"][1:-1, :, k]))
                A -= A.max()
                i, j = np.unravel_index(np.argmax(A), A.shape)
                peaks = np.argwhere(A == ndimage.maximum_filter(A, size=7, mode="wrap"))
                other = [A[a, b] for a, b in peaks
                         if np.degrees(np.arccos(np.clip(U[a, b] @ U[i, j], -1, 1))) > 30]
                rows.append({"freq_khz": float(f[k] / 1e3), "main_lobe_az_el_deg": [float(az[j]), float(el[i + 1])],
                             "next_lobe_db": float(max(other)) if other else None})
            out["sources"][src] = {"label": label, "per_frequency": rows}
            print(src, [(r["freq_khz"], round(r["next_lobe_db"], 1)) for r in rows])
    fn = paths.RESULTS / "validation" / "rform_nostril_lobes.json"
    fn.write_text(json.dumps(out, indent=1))
    print(fn)


if __name__ == "__main__":
    main()
