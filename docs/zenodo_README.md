# Bat ERTF Meshes: simulated Echolocation-Related Transfer Functions of seven bat species

Complex-valued transfer functions of both ears and of the emitter of seven bat species, computed with a boundary
element method (Burton-Miller, validated against NumCalc/Mesh2HRTF and an analytical solution) on cleaned micro-CT
head meshes. Dataset DOI: https://doi.org/10.5281/zenodo.23729269. Paper: Scientific Data Data Descriptor (submitted; DOI to be added). Code: https://github.com/Cosys-Lab/bat_ertf_dataset, archived in this record as bat_ertf_code-1.0.0.zip (BSD 3-Clause). Contact: Jan Steckel, jan.steckel@uantwerpen.be (Cosys-Lab, University of Antwerp, and Flanders Make).

## Species

| Folder | Species | Family | Emitter | Band (kHz) |
|---|---|---|---|---|
| Phyllostomus_discolor | *Phyllostomus discolor* | Phyllostomidae | two nostrils (BEM) | 20-145 |
| Micronycteris_microtus | *Micronycteris microtis* | Phyllostomidae | two nostrils (BEM) | 20-160 |
| Macrophyllum_macrophyllum | *Macrophyllum macrophyllum* | Phyllostomidae | two nostrils (BEM) | 20-160 |
| Rhinolophus_rouxii | *Rhinolophus rouxii* | Rhinolophidae | two nostrils (BEM) | 20-160 |
| Rhinolophus_formosae | *Rhinolophus formosae* | Rhinolophidae | two nostrils (BEM) | 20-135 |
| Eptesicus_fuscus | *Eptesicus fuscus* | Vespertilionidae | mouth (analytical piston, a = 6.59 mm) | 20-160 |
| Myotis_daubentonii | *Myotis daubentonii* | Vespertilionidae | mouth (analytical piston, a = 6.28 mm) | 20-160 |

## Files per species

- `mesh.stl`: clean head mesh (closed, 2-manifold, one body, no self-intersections), metres, CT frame.
- `mesh_<frame>.stl`: the same mesh in exactly the coordinates of `ertf_<frame>.h5`: the origin is the centre of the
  evaluation sphere, X points to azimuth 0, Y to azimuth +90 deg (the bat's left), Z to elevation +90 deg. Use this
  file to see which direction is which (figure "frames" in the paper shows all heads with the axes).
- `ertf_ct.h5`: all transfer functions in the CT frame (see below). *P. discolor* also has `ertf_reference.h5`, in
  the frame of the reference simulation of De Mey et al. (2008), 25-95 kHz.
- `HRTF_<species>.mat`: the nominal left ear, right ear and emitter in the lab's older MATLAB format: `load` gives a
  struct `bat_hrtf` with `hrtf_left`, `hrtf_right`, `hrtf_em` [elevation x azimuth x frequency], `az_vec`, `el_vec`,
  `freq_vec`, as in the reference files of De Mey et al. (2008) and Vanderelst et al. (2010). Differences: full sphere
  (azimuth -180..177.5 deg) and complex values (single precision; `abs()` gives the magnitudes of the older files),
  plus `species`, `landmarks_m` and `convention`.
- `ears_reference.sofa`, `emitter_reference.sofa` (*P. discolor* only): the transfer functions of `ertf_reference.h5` as SOFA.
- `ears_ct.sofa`: nominal left (receiver 1) and right (receiver 2) ear, SOFA `SimpleFreeFieldHRTF`, frequency domain.
- `emitter_ct.sofa`: nominal emitter, SOFA `FreeFieldDirectivityTF` (directions as receivers).
- `landmarks.json`: receiver and nostril positions (or mouth and piston parameters), the alternative source positions (`frames.<frame>.alternative_positions_m`, in the order of the `*_alt*` sources), the
  frame rotations, mesh statistics and conventions.

### HDF5 layout (`ertf_<frame>.h5`)

| Path | Content |
|---|---|
| `freqs` | frequencies [Hz], 0.5 kHz steps |
| `azimuth`, `elevation` | grid [deg]: azimuth -180..177.5, elevation -90..90, 2.5 deg steps |
| `left/H`, `right/H`, `emitter/H` | complex64 transfer function [elevation, azimuth, frequency] |
| `<source>/patch_faces` | indices of the vibrating mesh faces |
| `left_alt1..4`, `right_alt1..4`, `emitter_alt1..4` | placement ensemble: alternative receiver and nostril positions |
| attributes | species, frame, R (frame to CT rotation), landmarks, solver version (git), conventions |

## Conventions

- **Frame:** CT frame of the original lab simulations, X forward, Y left, Z up, metres. Azimuth is positive to
  the bat's left, elevation positive up; u = [cos(el) cos(az), cos(el) sin(az), sin(el)].
- **Time convention:** exp(+j w t).
- **Referencing:** H = p / p_ff at r = 1 m, with p_ff = j w rho Q exp(-jkr) / (4 pi r) the free-field pressure of a
  monopole at the origin with the source's volume velocity Q (in the solver's exp(-j w t) convention this is
  conj(p / p_ff) with p_ff = -j w rho Q exp(jkr) / (4 pi r), as in Mesh2HRTF); c = 343 m/s, rho = 1.1839 kg/m^3.
- **Impulse responses:** an inverse real FFT over frequency (after tapering the band edges) gives an impulse response
  with t = 0 at the origin. For directions on the source's side it starts up to |p|/c (about 60 us) before t = 0, so
  add a small pre-delay (`code/batertf/dataset.py` uses 0.2 ms).
- **Ears:** by reciprocity, a vibrating patch (radius 0.4 mm) at the ear canal entrance; H is the patch-averaged
  pressure for a source in direction (az, el) at 1 m.
- **Two-way ERTF:** emitter H times ear H, per direction and frequency.

## Code and examples

`code/` holds a small Python API (`code/batertf/dataset.py`: load, interpolate on the sphere, rotate the head,
two-way ERTFs, impulse responses, a point-reflector echo simulator) and examples in Python and MATLAB
(`code/examples/`, see `code/examples/README.md` for the conventions in detail). The complete code that produced the dataset,
including the solver and the posing tools, is `bat_ertf_code-1.0.0.zip` in this record and on GitHub
(https://github.com/Cosys-Lab/bat_ertf_dataset).

## Source placement

The receiver and nostril positions are modelling choices. For *P. discolor* and *M. microtis* they best match the
earlier simulations (De Mey et al. 2008; Vanderelst et al. 2010). For the other species they follow a rule derived
from those two: receivers at the maximum of front/back ratio plus surface occlusion, nostrils at the mirror-symmetric
pair 1.5-2.5 mm apart with the highest front/back ratio. The `*_alt*` sources show how the ERTF depends on this
choice.

## Origin of the meshes

The head meshes are legacy meshes from the archive of the University of Antwerp, made from micro-CT scans of one
preserved specimen per species. The origin of the individual specimens can no longer be traced. They are released
to preserve them and to make the most of this resource for future research.

| Item | Status |
|---|---|
| Scanner, voxel size, scan settings | P. discolor, M. microtis: SkyScan 1076 micro-CT, 35 um, downsampled to 70 um (Vanderelst et al. 2010, procedure of De Mey et al. 2008); R. rouxii: 70 um (Vanderelst et al. 2011) |
| Segmentation software and procedure | Amira, automatic thresholding with manual corrections (Vanderelst et al. 2010) |
| Earlier publications using the meshes | P. discolor: De Mey et al. 2008; M. microtis: Vanderelst et al. 2010; R. rouxii: Vanderelst et al. 2011; R. formosae: Vanderelst et al. 2013 |
| Processing in this release (cleaning, remeshing, frame) | fully documented (paper, Supplementary Note 1, `code/`) |

## Licence

Data: Creative Commons Attribution 4.0 International (CC BY 4.0). Code: BSD 3-Clause licence.
