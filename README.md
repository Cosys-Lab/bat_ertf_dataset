# Full-sphere emission and reception directivities of seven echolocating bat species

Code of the dataset [10.5281/zenodo.23729269](https://doi.org/10.5281/zenodo.23729269): simulated far-field emission
and reception directivities (complex transfer functions of both ears and of the emitter, full sphere, 20-160 kHz) of
*Phyllostomus discolor*, *Micronycteris microtis*, *Macrophyllum macrophyllum*, *Rhinolophus rouxii*, *Rhinolophus
formosae*, *Eptesicus fuscus* and *Myotis daubentonii*, computed from micro-CT head meshes with a GPU boundary element
method. The dataset is described in a Data Descriptor (Scientific Data, submitted).

This repository contains everything that produced the dataset, and the tools to use and extend it:

| Path | What |
|---|---|
| `batertf/dataset.py` | load a species folder of the dataset, interpolate to any direction and frequency, rotate the head, impulse responses, echo simulation (needs only NumPy and h5py) |
| `examples/python`, `examples/matlab` | examples in Python and MATLAB (directivity maps, impulse responses, echoes, head rotation, posing) |
| `examples/POSING.md` | how to pose the pinnae and noseleaves and compute new transfer functions |
| `batertf/gpubem.py` | GPU boundary element solver (Burton-Miller, constant elements, collocation, LU) |
| `batertf/rig.py` | posable rigs of the pinnae and noseleaves (skeleton, dual-quaternion skinning, validity check, glTF) |
| `batertf/` (other) | meshes, source patches, grids, analytical piston, NumCalc interface, reference metrics |
| `scripts/` | mesh cleaning, source placement, production, export, validation, rigs, figures of the paper |
| `tests/` | verification against the analytical sphere and NumCalc (Mesh2HRTF) |
| `config/` | species, landmarks and rig settings |
| `data/meshes/dataset/` | the seven clean head meshes (CT frame, metres) |
| `results/` | rigs, placement and validation results that the scripts read |
| `docs/` | technical notes (solver, meshes, placement rule, rigging, calibration, validation) |

A video of the dataset: https://youtu.be/Q9Kxj9oIEwg

## Quick start: use the data

Download a species folder from Zenodo (one zip per species), then:

```python
import numpy as np
from batertf import dataset as bd

d = bd.load("Myotis_daubentonii")              # species folder
H = d.interp("left", np.array([[1.0, 0, 0]]))  # left ear, straight ahead, all frequencies
t, h = d.impulse_response("left", np.array([[0.8, 0.6, 0]]))
```

In MATLAB, run `examples/matlab/exampleDirectivity.m` (set `folderSpecies` to a species folder), or load the
`HRTF_<species>.mat` file of a species folder (struct `bat_hrtf`).

## Pose the heads and compute new transfer functions

See [`examples/POSING.md`](examples/POSING.md) and `examples/python/05_pose_and_simulate.py`. This needs an NVIDIA GPU
with 10-27 GB of memory, depending on the mesh. Mesh2HRTF can be used on a CPU instead.

## Install

    python -m venv .venv
    .venv/bin/pip install -r requirements.txt     # PyTorch with CUDA for the solver, see requirements.txt
    python tests/sphere_check.py                  # analytical verification of the solver

Machine-specific paths (work folder for large outputs, NumCalc binary) go in `config/local.toml`:

    [paths]
    work_dir = "/data/BatERTF/work"
    numcalc = "/path/to/NumCalc"

## Conventions

- **Frame:** X forward, Y left, Z up, in metres.
- **Directions:** azimuth is positive to the bat's left, elevation positive up.
- **Referencing:** H = p / p_ff at 1 m, relative to a free-field monopole at the origin with the same volume velocity.
- **Time convention:** exp(+j w t).

Details are in the dataset README.

## Citation and licence

Please cite the Data Descriptor and the dataset (DOI 10.5281/zenodo.23729269). The code is released under the BSD
3-Clause licence (`LICENSE`), the data under CC BY 4.0.
