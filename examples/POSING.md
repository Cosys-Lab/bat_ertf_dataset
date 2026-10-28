# Posing the heads and computing new transfer functions

Every species folder of the dataset contains a rig (`rig/`): a skeleton for each pinna (3 bones: base, mid, distal)
and, for the five nasal species, for the noseleaf (2 bones). With it, the pinnae and the noseleaf can be posed, and
the transfer functions of the posed head can be computed with the same boundary element method (BEM) as the dataset.
The dataset itself contains only the rest pose, which is the preserved posture of the specimen; posed transfer
functions are computed by the user, as described here.

## What you need

- **The full code** from the GitHub repository (the Zenodo `code.zip` only contains the loader and the examples):
  `batertf/rig.py` (rig), `batertf/gpubem.py` (solver), `batertf/sources.py`, `batertf/piston.py`, `batertf/grid.py`.
  Python 3.12, PyTorch with CUDA, NumPy, SciPy, h5py, trimesh, PyMeshLab (for the validity check).
- **An NVIDIA GPU.** The solver builds the dense BEM matrix (complex single precision), so the memory grows with the
  square of the number of triangles:

  | Mesh | Triangles | Matrix | Time per frequency (RTX PRO 5000) |
  |---|---|---|---|
  | *M. daubentonii* | 35,500 | 10 GB | about 8 s |
  | *M. macrophyllum* | 46,300 | 17 GB | about 15 s |
  | *M. microtis*, *E. fuscus*, *R. rouxii*, *P. discolor* | 52,000-54,000 | 22-23 GB | about 20 s |
  | *R. formosae* | 57,700 | 27 GB | about 25 s |

  Add about 15 % for the rest of the solver. Every frequency of every pose needs one LU decomposition. All sources
  of that frequency (both ears, the emitter) then cost little extra.
- **No suitable GPU?** Export the posed mesh as STL (`m.export("posed.stl")`) and compute it with Mesh2HRTF/NumCalc
  on a CPU. The source patches are the faces listed by `sources.disc_patch` (see below). NumCalc solves the same
  discrete problem; our comparison on a dataset mesh agrees within 0.13 dB.

## Joints and axes

A pose is a dictionary `{joint: {axis: degrees}}`. The joints are `pinna_left.base`, `.mid` and `.distal`, the same
for `pinna_right`, and `noseleaf.base` and `.distal`. `pinna_both.<joint>` sets both pinnae at once, mirror-symmetric.

| Axis | Positive direction |
|---|---|
| `flex` | the bone tilts backward (an upright pinna) or up (a forward-pointing one) |
| `abduct` | the tip moves outward, away from the midline (noseleaf: to the bat's left) |
| `yaw` | the bone turns about the vertical axis of the head: the opening of a pinna turns forward (noseleaf: to the left) |
| `twist` | the bone rotates about its own axis (mirror-symmetric for the two sides) |

Valid ranges (no self-intersections, no flipped faces, no edge stretched more than twice) are in
`results/rig/valid_ranges.json` and in Supplementary Table S4 of the paper. They were scanned one joint and one axis at a time.
For the pinna base, flex and abduct reach about ±15 to ±40 degrees and yaw ±15 to ±35 degrees, depending on the species.
Combine joints with care and always run the check (`Rig.check`).

## Step by step (Python)

```python
from batertf import rig, sources, project, gpubem, grid

r = rig.load_rig_folder("Myotis_daubentonii")          # species folder of the dataset
pose = {"pinna_both.base": {"flex": 15, "yaw": 10},     # both pinnae 15 deg backward, turned 10 deg forward
        "pinna_both.distal": {"flex": -20}}             # and the tips bent 20 deg
m = r.deform(pose)                                      # posed mesh, same triangles as mesh.stl
assert r.check(m, pose)["valid"]
lm = r.deform_landmarks(pose, m)                        # ears, nostrils or mouth on the posed surface
patches = {"left": [sources.disc_patch(m, lm["ear_left"], project.EAR_RADIUS)],
           "right": [sources.disc_patch(m, lm["ear_right"], project.EAR_RADIUS)]}
pts, _ = grid.unique_points("sphere")                   # the full-sphere grid of the dataset
H = gpubem.solve_lu(m, patches, [40e3, 80e3, 120e3], pts)
H_left = gpubem.to_grid(H["left"], "sphere")            # [elevation, azimuth, frequency], as ertf_ct.h5
```

`examples/python/05_pose_and_simulate.py` does all of this, including the emitter, and saves the result in the
layout of `ertf_ct.h5`. It then compares the result with the rest pose:

    python examples/python/05_pose_and_simulate.py <species folder> --flex 15 --yaw 10 --freqs 40000,80000,120000

The emitter follows the species:
- **Nasal species:** two patches at the posed nostrils (`nostril_0`, `nostril_1`), radius `project.NOSTRIL_RADIUS`.
- **Oral species:** the analytical piston of the dataset. Its centre, axis and radius are in the attributes of
  `emitter` in `ertf_ct.h5`. Poses of the pinnae do not change it.

## Posing interactively (Blender)

`rig/<species>.glb` is a rigged glTF model (millimetres). Import it in Blender, pose the bones, and export it as glTF
(keep the bone names). Then convert it back into a pose:

```python
pose = rig.pose_from_gltf(r, "posed.glb")
```

and continue as above. The glTF skinning matches `Rig.deform` within 0.06 mm at 20 degrees.

## Many poses: pose grids on one or more GPUs

`scripts/pose_atlas.py` computes a grid of poses for all species through a job queue on shared storage. Any number of
GPU workers, on any machine, can join; the queue resumes after interruptions.
1. Edit the pose lists (`FLEX`, `ABDUCT`, `NOSELEAF`) and the frequency step (`STEP_KHZ`) at the top of the script.
2. Run `python scripts/pose_atlas.py plan --queue <shared folder>` once.
3. On each GPU, run `python scripts/pose_atlas.py work --queue <shared folder> --device cuda:0`.
4. Run `python scripts/pose_atlas.py merge --queue <shared folder>`. This writes one `<species>_poses.h5` per
   species, with a table of the poses, the transfer functions and the posed vertices.

The cost is about 20 s per frequency and pose for a 50,000-element mesh. For example, 15 poses × 57 frequencies
(2.5 kHz steps) × 7 species is about 30 GPU hours.

## Limitations

- **Geometric deformation:** the skinning is not a model of the tissue. The soft band at the base of a pinna
  stretches and compresses rather than folding like skin.
- **Tip folding:** bending the tip of a strongly curved pinna squeezes its concave side.
- **Source patches:** the ear canal entrances and the nostrils are essentially unaffected by a pose (weights of at
  most 0.002 and 0.01). The source patches therefore stay where they are, while the anatomy around them moves.
- **Validation:** posed responses have not been validated against measurements of live bats. Use them for studies
  of the effect of posture, with this in mind.
