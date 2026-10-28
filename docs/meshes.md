# Meshes

## Sources

| Folder | What | In git | In the dataset |
|---|---|---|---|
| `data/meshes/Old/` | Original CT meshes from Jan (mm, CT frame), plus renders; `micro High Res 200 percent.stl` is a 2x scaled copy for 3D printing | no (source material, kept in Dropbox) | no |
| `data/meshes/ct/` | Intermediate: the CT meshes repaired and moved into the earlier working frame (scripts/prepare_ct_meshes.py), used to compare with the reference mesh | no | no |
| `data/meshes/original/` | The first meshes Jan supplied: remeshed and smoothed versions of the CT meshes (ICP median distance 0.17-0.24 mm), in a different frame. Superseded. | yes (history) | no |
| `data/meshes/dataset/` | **Dataset meshes**: clean remeshes of the CT meshes (scripts/remesh_ct.py), CT frame, metres, plus `meshes.json` | yes | yes |

Only clean meshes go into the dataset (Jan, 2026-10-27): closed, 2-manifold, one body, consistently oriented
outward, no self-intersections. `remesh_ct.py` refuses to write a mesh that does not pass.

## Frame

The CT files share the frame of the original lab simulations. Calibrating our BEM on the reference ERTFs of
*M. microtis* (Vanderelst et al. 2010), fitting only to acoustic data, recovered the CT file frame within 2-5 deg with
axes relabelled: forward = CT +z, left = CT +x, up = CT +y. The dataset meshes use that frame with the axes renamed:
**X forward, Y left, Z up**, in metres. Six of the seven CT frames point forward along the head's midline. The
nasal emitters other than P. discolor (M. microtus, M. macrophyllum, both Rhinolophus) share Vanderelst's
convention (nostril plane vertical). The P. discolor reference of De Mey et al. (2008) is in a different orientation
(33 deg from its CT frame), so it is compared after an orientation fit.

## Remeshing (scripts/remesh_ct.py)

1. Repair the CT mesh (duplicates, null faces, non-manifold elements, small holes).
2. Isotropic explicit remeshing (pymeshlab) with re-projection on the CT surface and a surface-distance check;
   target edge 0.27 mm, or larger where needed to stay within 58,000 faces (the GPU solver's dense-matrix budget).
3. Local repair loop until clean: self-intersecting faces, faces on non-manifold edges and at non-manifold vertices
   are cut out with two or more rings of neighbours, the holes are closed and the patches remeshed. Each defect
   mask is read separately (pymeshlab selection filters replace the selection), and each species runs in its own
   process (pymeshlab keeps state between meshes, after which remeshing silently does nothing).
4. Two-sided deviation from the CT surface and element statistics in `data/meshes/dataset/meshes.json`.

Deviations from the CT surface are 0.02-0.06 mm median and below 0.18 mm at the 95th percentile. The maxima,
0.3-0.8 mm, are at the locally repaired spots.

## Result

All seven pass the independent check (trimesh + pymeshlab: closed, outward, one body, no non-manifold elements,
no holes, no self-intersections):

| Species | Faces | Mean edge (mm) | f at 6 elements/wavelength (kHz) |
|---|---|---|---|
| *Eptesicus fuscus* | 52,474 | 0.299 | 191 |
| *Macrophyllum macrophyllum* | 46,306 | 0.277 | 207 |
| *Micronycteris microtus* | 51,980 | 0.270 | 211 |
| *Myotis daubentonii* | 35,536 | 0.276 | 207 |
| *Phyllostomus discolor* | 53,984 | 0.392 | 146 |
| *Rhinolophus formosae* | 57,746 | 0.424 | 135 |
| *Rhinolophus rouxii* | 53,672 | 0.279 | 205 |
