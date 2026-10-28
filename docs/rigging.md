# Rigs: deformable pinnae and noseleaves

Jan, 2026-10-28: "add skeletons to the ears and noseleaves so we can deform them". Each of the seven dataset meshes
(`data/meshes/dataset/*.stl`, CT frame X forward, Y left, Z up, metres) has a rig: skinning weights and a bone chain
per pinna (3 bones) and per noseleaf (2 bones, nasal species only), parented to the fixed head. A pose (joint
rotations in degrees) gives a deformed mesh with **the same vertices order and faces**, so source patches and landmarks
map by index and the mesh goes straight into `gpubem.solve_lu`.

| File | What |
|---|---|
| `batertf/rig.py` | segmentation, skeleton, weights, dual-quaternion skinning, validity check, glTF export / pose import |
| `batertf/rigplot.py` | renders (mesh coloured by bone weight, skeleton) |
| `config/rig.toml` | manual overrides per species (only R. formosae needs one) |
| `scripts/build_rigs.py` | builds all rigs: `results/rig/<key>_rig.json`, `<key>_weights.npz`, `<key>_rig.png`, `<work>/rig/<key>.glb` |
| `scripts/rig_ranges.py` | valid angle range per joint and axis: `results/rig/valid_ranges.json` |
| `scripts/rig_poses.py` | `results/rig/poses.png` (M. microtis, P. discolor; rest and six poses) |
| `scripts/rig_demo_bem.py` | BEM demo: `results/rig/demo_mmicr.json`, `demo_mmicr.png` |

## Usage

```python
from batertf import rig, sources, project, gpubem, grid
r = rig.load_rig("mmicr")
print(r.bones)        # pinna_left.base/.mid/.distal, pinna_right.*, noseleaf.base/.distal
pose = {"pinna_left.base": {"flex": 20},           # tilt the left pinna backward / up about its base
        "pinna_left.distal": {"flex": -30},        # bend its tip
        "noseleaf.base": {"flex": 10, "abduct": 5}}
m = r.deform(pose)                                  # trimesh, same faces as r.rest
print(r.check(m, pose))                             # watertight, self-intersections, inverted faces, stretch, valid
lm = r.deform_landmarks(pose, m)                    # ear_left/right, nostril_0/1 or mouth, on the deformed surface
patches = {"left": [sources.disc_patch(m, lm["ear_left"], project.EAR_RADIUS)]}
pts, _ = grid.unique_points("sphere")
H = gpubem.solve_lu(m, patches, [80e3], pts)
```

`"pinna_both.<joint>"` sets both pinnae at once (the axes are mirror-symmetric, so equal angles give a symmetric pose).

**Blender.** `/data/BatERTF/work/rig/<key>.glb` (glTF 2.0 with skin: head root, the bones, one end joint per chain,
4 weights per vertex). Units **millimetres** (a 30 mm head in metres falls inside Blender's default clip range); Blender's
importer turns the glTF Y-up axes back into X forward, Y left, Z up. Each joint's local axes: X = flex, Y = along the
bone, Z = X x Y (= abduct on the left side and the noseleaf, -abduct on the right side; twist is +Y left, -Y right).
After posing, export as GLB (keep bone names) and read the pose back with `rig.pose_from_gltf(r, "posed.glb")`; it
uses the skinning matrices (posed global x inverse bind), so it does not depend on how the importer oriented the bones.
The GLB skin (linear blend, 4 weights) reproduces `Rig.deform` (dual quaternions) within 0.06 mm at 20 deg; the pose
round trip is exact (tested on a programmatically posed file; not yet opened in Blender itself).

## Method

1. **Head core.** The solid is voxelized (0.1 mm) and morphologically opened with a 1 mm ball (Euclidean distance
   transforms). Structures thinner than about 2 mm (pinnae, tragus, noseleaf) vanish; `dcore` = distance of each
   vertex from the opened solid. It is zero on the head and grows along each appendage from its base.
2. **Segmentation.** Vertices with dcore > 0.5 mm form connected components. Pinna of a side: the largest component
   on that side within 4 mm of the ear landmark, plus further components on that side within 4.5 mm (tragus /
   antitragus where they are separate: P. discolor left, E. fuscus and M. daubentonii both sides). Noseleaf: the
   largest component within 6 mm of `nose_centre` with its centroid within 2.5 mm of the midline, cut 0.5 mm above
   the nostril midpoint along the direction nostrils -> leaf tip (the part below stays with the head); the piece
   connected to the tip is kept. Oral species (E. fuscus, M. daubentonii) have no noseleaf; no mouth/lip region is rigged.
3. **Appendage weight h.** Harmonic interpolation (1/edge-length graph Laplacian) between 1 on the appendage where
   dcore >= 3.5 mm (pinna) / 1.5 mm (noseleaf) and 0 where dcore <= 0.1 mm, on other thin structures, on the other
   appendages and on the cut-off snout part (the noseleaf is fully rigid only 1 mm away from it); then smoothstep.
   The pinna band is wide on purpose: rotating a pinna whose base line is up to 10 mm long needs a soft base, or the
   skin folds into itself.
4. **Chain.** t = geodesic (Dijkstra) distance from the h = 0.5 iso-line at the base, normalised to 1 at its 99th
   percentile. Joints: base = centroid of the base iso-line; inner joints = centroids of the iso-t bands t = 0.35, 0.7
   (pinna) or 0.5 (noseleaf); end = centroid of t >= 0.96. For a funnel-shaped pinna the centroids lie on the funnel
   axis, i.e. the chain follows the pinna's medial axis.
5. **Bone weights.** h x smooth partition-of-unity hats in t (blend half-width 0.2 in t); head root 1 - h. Finally
   Gaussian spatial smoothing (sigma 0.4 mm, Euclidean neighbours): both skins of a 0.2-0.4 mm thick sheet get the
   same transform. Without it the two skins of the thin pinna base got different weights and crossed at 15-25 deg.
6. **Deformation.** Forward kinematics, then dual-quaternion skinning (no candy-wrapper collapse under twist).

### Joint axes and sign conventions

With d the rest direction of the bone (joint -> next joint), at every joint:

| Axis | Rotation axis | Positive |
|---|---|---|
| flex | -Y, made perpendicular to d | bone turns from +X toward +Z and from +Z toward -X: an upright pinna / noseleaf tilts **backward**, a forward-pointing one (M. microtis pinnae, phyllostomid spears of M. microtis and M. macrophyllum) tilts **up**. Same on both sides. At the distal joint this folds the tip. |
| abduct | cross(d, n), n = lateral outward (+Y left, -Y right; +Y for the noseleaf) made perpendicular to d | tip moves **outward** (away from the midline); noseleaf: to the bat's left |
| twist | +d (left, noseleaf), -d (right) | mirror-symmetric twist about the bone |

The rotation at a joint is R = R_flex R_abduct R_twist (twist applied first), about the joint, in the frame of the
parent bone; the axes are fixed in the rest pose. Joints are named `<appendage>.<base|mid|distal>`.

### Landmarks

Landmarks are attached to the surface (face + barycentric coordinates + normal offset) and follow it. The ear canal
landmarks lie at the pinna base where the weight is ~0 (largest bone weight at an ear landmark: 0.002, except
R. formosae right ear 0.15), so they essentially stay; in the M. microtis demo the left-ear landmark moved 0.001 mm
and the 0.4 mm ear patch had the same faces as at rest. The nostrils carry at most 0.01 weight (stay fixed), so the
emitter patches are unchanged by noseleaf poses. If you want the ear patch to follow the pinna, build it from the
deformed landmark (as in the demo) and raise its weight in the rig, or keep the rest patch faces by index.

### Validity check (`Rig.check`)

Valid = watertight and consistently wound (topology is unchanged, so always), no self-intersecting faces (pymeshlab
`compute_selection_by_self_intersections_per_face`, as in `scripts/remesh_ct.state`), no face whose normal flips
against its skinned rest normal, and edge length ratio deformed/rest within 0.25-2.0 (edges shorter than half the
mean are skipped: the repaired spots have 0.02 mm edges whose ratios mean nothing). Also reported: max edge length (for
the six-elements-per-wavelength limit), minimum area ratio, volume change.

## Segmentation per species

| Species | Pinna (moving vertices L/R, chain mm) | Noseleaf (moving vertices, chain mm) | Manual override |
|---|---|---|---|
| P. discolor | 4750 / 4751, 17.6 / 18.5 (left incl. separate tragus) | spear, 441, 5.8 | none |
| M. microtis | 8106 / 8043, 14.5 / 14.7 | spear (points forward), 721, 6.4 | none |
| M. macrophyllum | 5609 / 5842, 12.7 / 12.8 | spear, 1445, 8.8 | none |
| R. rouxii | 6358 / 6217, 13.0 / 13.8 | lancet + sella, 1204, 4.4 | none |
| R. formosae | 6933 / 6610, 23.7 / 23.7 | lancet + upper sella, 976, 3.3 | `max_lateral_mm = 2.5`: the thin component above the nostrils also holds two lateral flaps out to \|y\| = 6 mm; without the override they would rotate with the lancet |
| E. fuscus | 5646 / 5572, 12.9 / 13.1 (incl. tragus) | none (oral) | none |
| M. daubentonii | 3722 / 3176, 9.2 / 9.2 (incl. tragus) | none (oral) | none |

Figures: `results/rig/<key>_rig.png` (front, side, top, zoom on noseleaf or ear base).

## Valid ranges (single joint, single axis, 5 deg steps, scan up to +-60 deg)

Base joints (`results/rig/valid_ranges.json` has every joint, and the reason of the first failure, nearly always
self-intersection: the pinna or its base fold hitting the head or itself). Left / right pinna:

| Species | pinna base flex | pinna base abduct | pinna base twist | noseleaf base flex / abduct / twist |
|---|---|---|---|---|
| P. discolor | -30..+25 / -25..+20 | -20..+20 / -20..+20 | -35..+25 / -30..+20 | -60..+60 / -15..+20 / -30..+35 |
| M. microtis | -25..+35 / -30..+40 | -15..+15 / -15..+20 | -30..+25 / -40..+35 | -60..+60 / -30..+25 / -45..+50 |
| M. macrophyllum | -40..+40 / -40..+40 | -25..+30 / -20..+25 | -30..+30 / -40..+35 | -60..+60 / -20..+15 / -40..+35 |
| R. rouxii | -35..+40 / -30..+30 | -35..+30 / -35..+35 | -50..+55 / -35..+35 | -25..+35 / -35..+30 / -45..+45 |
| R. formosae | -20..+25 / -20..+25 | -20..+20 / -10..+20 | -20..+15 / -25..+20 | -15..+10 / -5..+5 / -30..+30 |
| E. fuscus | -30..+30 / -30..+30 | -30..+30 / -35..+30 | -30..+30 / -25..+30 | - |
| M. daubentonii | -50..+40 / -60..+50 | -45..+40 / -40..+45 | -50..+55 / -35..+55 | - |

Mid joints reach about +-30-50 deg, distal joints (tip bending) mostly +-50-60 deg (R. formosae distal flex
-25..+35). Combined poses are not covered by the table: check them with `Rig.check`.

## Demo: M. microtis left pinna +-20 deg (scripts/rig_demo_bem.py)

Left pinna base flex +20 deg (tilted up/back) and -20 deg (down/forward); both meshes valid (no self-intersection,
edge stretch 0.66-1.45). Left-ear patch at the deformed landmark (unchanged faces), `gpubem.solve_lu`, full-sphere
grid, 40/80/120 kHz (~19 s per frequency on the RTX PRO 5000). Comparison with the rest pose (dB patterns normalised
per frequency, -30 dB floor, cos(el)-weighted correlation):

| | 40 kHz | 80 kHz | 120 kHz |
|---|---|---|---|
| flex +20: correlation full sphere / frontal | 0.79 / 0.71 | 0.80 / 0.77 | 0.77 / 0.73 |
| flex +20: peak shift (deg) | 14 | 18 | 60 (secondary lobe takes over) |
| flex +20: max gain change (dB) | -0.5 | +3.1 | -0.7 |
| flex -20: correlation full sphere / frontal | 0.82 / 0.73 | 0.76 / 0.81 | 0.77 / 0.71 |
| flex -20: peak shift (deg) | 15 | 27 | 25 |
| flex -20: max gain change (dB) | -0.3 | -4.2 | +1.6 |

A 20 deg rotation of the pinna about its base, with the ear canal patch unchanged, lowers the pattern correlation
with the rest pose to 0.76-0.82 (rms difference 5-7 dB): pinna posture is a first-order parameter of these ERTFs.
Peak shifts of 14-27 deg follow the pinna; the 60 deg at 120 kHz is a jump between two lobes of similar level.

## Limitations

- Skinning is geometry, not a mechanical tissue model: the base band stretches and compresses (edge ratios 0.6-1.5 at
  20 deg) instead of folding like skin over cartilage; volumes are kept only approximately (<0.1 % at 20 deg).
- Large rotations near the base self-intersect (pinna into the head or its own base fold); single-axis ranges are in
  the table, combinations must be checked.
- The chain follows geodesic distance from the base; the fold line of a distal flex is an iso-geodesic line, not a
  chosen anatomical crease. On strongly curved (funnel) pinnae a tip fold compresses the concave side.
- The tragus moves with the base bone but only partially (its dcore stays below the 3.5 mm band end), so it lags
  behind the pinna; it is a separate structure anatomically and this is arguably right, but not tuned.
- Noseleaf of R. formosae: only the lancet and upper sella (3.3 mm chain) move; small ranges at its base
  (abduct +-5 deg). The horseshoe and lateral flaps are fixed head in all rhinolophids.
- Joint axes come from head axes (lateral Y), not from each pinna's sheet plane; at a distal joint "flex" is a fold
  only where the sheet contains the lateral direction (true for the pinnae here).
- Element size: rigid parts keep their elements; the band may stretch edges up to 2x before the check fails, which
  lowers the six-elements-per-wavelength frequency there (reported as `edge_max_mm`).
