# External validation: measured in-ear HRTF of *Myotis daubentonii* "Mdau2" (Vesterholm et al. 2025)

## Provenance and licence

- **Paper:** Vesterholm K.K., Häfele F.T., Figeac A., Jakobsen L. (2025). 3D photogrammetry as a low-cost and
  non-invasive method for acoustic modelling of animal hearing. *Methods in Ecology and Evolution* 17(1):125-136.
  doi:10.1111/2041-210X.70197.
- **Data:** Zenodo record 15789942, doi:10.5281/zenodo.15789942, published 2025-07-02, licence **CC-BY 4.0**
  (record metadata). The SOFA file's own attribute says "No license provided, ask the author for permission". The
  record licence applies; we cite the record and do not redistribute their files.
- **Archive:** one file, `Data and code.zip`, 593,126,084 bytes, md5 `65fc47512c1e64df7fa86ffe4db7c4a0`, top folder
  `Data and code - Resubmission/`. **The archive has no author README.** The conventions below come from the MATLAB
  scripts that ship with the data (`ir2hrtf.m`, `PlotMeasuredHRTFs.m`, `PlotHRTFsFromSOFA.m`), from the file
  contents, and from consistency checks against the data.
- **Download:** `python -I scripts/fetch_vesterholm.py /data/BatERTF/external/vesterholm2025` reads the zip's central
  directory and only the selected members over HTTP range requests (CRC checked). It also writes the full archive
  listing to `listing.json`. The data stay outside Dropbox and git.

### Files used (in `Data and code - Resubmission/`)

| File | Bytes | Content |
|---|---|---|
| `3D Models/Mdau2_CTscan.ply` | 659,024 | µCT head mesh, **mm**, 12,195 vertices / 24,406 faces, watertight, one body, area 1049 mm² |
| `Measured HRTF/Mdau2_ir.mat` | 36,535,058 | measured in-ear IRs (see below) |
| `Measured HRTF/Mdau2_RotationMatrix.mat` | 306 | `RotoTranslationMatrix` (4 x 4) |
| `Measured HRTF/Mdau2_Exp.mat` | 10,393,677 | `Mdau2_Exp` (100 x 199 x 76): measured dB maps, rotated and interpolated onto the simulation grid |
| `SOFA files/Mdau2_CTscan.sofa` | 20,492,981 | their Mesh2HRTF 1.1.1 simulation on the µCT mesh (GeneralTF, complex) |
| `Measured HRTF/*.m`, `SOFA files/*.m` | < 10 kB each | their processing and plotting scripts (documentation) |

## What the data contain

**Measured (`Mdau2_ir.mat`).** The specimen is called `Mdau03` in the recording paths and `CT_Mdau03` in the SOFA
comment. Recorded 2023-09-22 in a previously deceased bat, with an in-ear microphone in **one ear (the left, see
below)**.
- `files_impulse_response` (1095 x 512, fs = 375 kHz): one IR per (speaker, turntable position). It is
  `files_impulse_response_full` (4096 taps) truncated and windowed; the tail energy is < 1e-10.
  `files_speaker_num` gives the speaker (1-15) and `files_ttable_pos` the turntable position (0:5:360°; 0 and 360
  are both present).
- `speakers`: 15 speakers at elevations 90, 70, 60, 50, 40, 30, 20, 10, 0, -10, -20, -30, -45, -60, -80°. Nothing
  is measured below -80°.
- The IRs are **reference-compensated per speaker**: |FFT(IR)| = `files_spectra` - `refs_spectra[speaker]` in dB
  (checked to 0.01 dB). The references are recordings without the bat. The IRs peak at sample 0-2, so they are
  minimum-phase. The measured phase is not physical, so **only the magnitude is compared**.
- The analysis band is set in `ir2hrtf.m`: Mdau 25-100 kHz in 1 kHz steps. The magnitude is the RMS of 50 ms pure
  tones filtered with the 512-tap IR, which equals |DTFT(IR)| at f; we use the DTFT.

**Simulated (`Mdau2_CTscan.sofa`).** Mesh2HRTF 1.1.1 (sofar 1.1.1), created 2024-06-13. `Data.Real/Imag` has shape
(19,900, 1, 76), for 25-100 kHz in 1 kHz steps. `SourcePosition` (spherical, deg, 1 m) is a **100 el x 199 az
grid** (el = linspace(-90, 90, 100), az = -180 + k·360/199, stored as 0-360). This is the same grid as
`Mdau2_Exp`. `ReceiverPosition` = (-4.938, 3.446, -2.180) mm.

## Conventions and alignment (what the validation uses)

1. **Measurement directions** (as in `ir2hrtf.m`): drop turntable position 360; wrap to ±180; then
   **az = -turntable** ("+AZ is left ... view of bat"). el is the speaker elevation. The 72 zenith recordings are one
   direction; their magnitude is averaged, as they do. This gives 1009 directions. MATLAB `sph2cart` gives
   u = (cos el cos az, cos el sin az, sin el), the same as `batertf.grid`.
2. **Mesh / simulation frame = PLY frame.** Their Mesh2HRTF receiver lies 0.012 mm from the PLY surface, with the
   outward normal (0.20, 0.96, 0.18). So the SOFA was computed on this mesh, in this frame, in metres. The frame is
   approximately X forward (narrow snout at +X, 4 mm wide at x = 5-7 mm), Y left (pinnae span ±11 mm in Y), Z up
   (pinna tips at +Z). The receiver is at +Y, so it is the **left ear**, and both the simulated and measured maps
   peak at positive azimuth.
3. **Measurement frame to mesh frame:** **u_mesh = R u_meas**, R = `RotoTranslationMatrix(1:3,1:3)` (a rotation
   of 32.9°; det = 1). This is exactly what `ir2hrtf.m` does to build `Mdau2_Exp`, which is the map that the paper
   compares with the simulation. The translation column (-6.8, 4.7, -0.5; mm) does not matter for far-field
   directions.
   - Check 1: our reading of the IRs reproduces their `Mdau2_Exp` (rotated, gridded) maps: correlation 0.999, median
     difference 0.07-0.12 dB at 30 and 70 kHz.
   - Check 2 (their sim vs measured, nearest-neighbour, 30/50/70/90 kHz): correlation with R 0.76/0.74/0.71/0.60;
     with no rotation 0.76/0.69/0.65/0.62; with R^T 0.68/0.61/0.57/0.54. R is clearly better than R^T, and better
     than no rotation at 50-70 kHz.
   - **Confidence:** high for the az sign, the frame identity of the mesh and SOFA, the receiver and the ear side.
     Medium for the measurement-to-mesh rotation. It is their alignment, and we apply it the way they do. Its
     residual error is unknown: the measured and simulated peaks still differ by about 20-30°, for their simulation
     as for ours. The validation also reports the result without the rotation as a sensitivity check.
4. **Phase convention of their SOFA:** it matches ours (exp(+jωt) after our conjugation) only after **complex
   conjugation of their data**. Complex correlation is 0.99-1.00 with conj and 0.01-0.04 without. So Mesh2HRTF
   1.1.1's SOFA stores the NumCalc exp(-jωt) values, and our files store the conjugate. The referencing is the same
   (monopole at the origin with the same volume velocity): levels agree to about 1 dB.

## Our pipeline on their mesh (`scripts/validate_vesterholm.py`)

- **Mesh:** `remesh_ct.remesh_isolated` on the PLY (repair, isotropic remeshing, local repair until clean), target
  edge max(L_MIN = 0.27 mm, sqrt(A / (0.433 N_MAX))). Then metres, kept in the PLY frame (no CT-frame relabelling).
  Full run: N_MAX = 58,000. The L_MIN floor then gives about 33k faces (f6 about 210 kHz). Coarse test:
  N_MAX = 20,000. Cached in `<work>/vesterholm/Mdau2_<tag>.stl/.json`.
- **Receiver:** a disc patch of radius `project.EAR_RADIUS` (0.4 mm), centred at **their Mesh2HRTF
  ReceiverPosition**. The placement rule (`apply_placement_rule.py`) is *not* used: their microphone position (in
  the ear canal) is not given in mesh coordinates, so their simulation receiver is the best available proxy.
- **Solver:** `gpubem.solve_lu`, evaluated at 1 m at the 1009 measurement directions mapped into the mesh frame,
  the same directions unrotated (sensitivity check), and their 19,900 SOFA directions. Frequencies are saved in
  chunks to `<work>/vesterholm/solve_<tag>.npz`, so the run can resume.
- **Metrics:** `reference.metrics_points`, the same definitions as `reference.metrics`: each pattern is normalised
  to its own maximum, floored at -30 dB, solid-angle weighted; we compute the correlation of the dB patterns, the rms
  dB difference, and the angle between the peaks. The measurement-grid weights are elevation bands between midpoints
  times 5°, and a cap for the zenith; the SOFA grid uses cos(el). For "Mesh2HRTF vs measured", their SOFA magnitude
  is interpolated bilinearly at R u_meas. Only frequencies that are on their 1 kHz grid are used (every 5 kHz in a
  2.5 kHz run). For "ours vs Mesh2HRTF" we compare on their own grid, with no interpolation. We also report the level
  offset, the complex correlation, and the rms phase difference where both magnitudes are within 20 dB of the
  maximum. The broadband value is the pattern of the power averaged over frequency.
- **Outputs:** `results/validation/vesterholm.json` (full run) or `vesterholm_coarse.json` (coarse test);
  `<work>/vesterholm/maps_<tag>.npz`; the figure `paper/figures/fig_vesterholm.py` writes `vesterholm.pdf/png`
  (with `--coarse`, it writes `results/validation/vesterholm_coarse.png`).

## Coarse test (GPU smoke test: 19,854 faces, 30/50/70/90 kHz; NOT the final result)

Mesh: mean edge 0.347 mm, deviation from the PLY median/p95/max = 0.05/0.13/0.28 mm. Patch: 18 faces, 0.51 mm²,
centroid 0.14 mm from their receiver. Solve time 4 s per frequency, 4.7 GB peak.

| Comparison | corr 30 / 50 / 70 / 90 kHz | rms dB 30 / 50 / 70 / 90 | broadband corr / rms |
|---|---|---|---|
| Ours vs measured | 0.74 / 0.73 / 0.69 / 0.56 | 4.3 / 5.4 / 6.6 / 7.2 | 0.87 / 3.1 dB |
| Their Mesh2HRTF vs measured | 0.75 / 0.72 / 0.68 / 0.56 | 4.3 / 5.5 / 6.7 / 7.1 | 0.87 / 3.1 dB |
| Ours vs their Mesh2HRTF | 0.997 / 0.997 / 0.989 / 0.982 | 0.4 / 0.6 / 1.2 / 1.4 | 0.999 / 0.3 dB |
| Ours vs measured, without the rotation | 0.74 / 0.68 / 0.65 / 0.59 | 4.3 / 5.8 / 7.0 / 7.0 | |

Ours vs Mesh2HRTF, complex: complex correlation 0.999/0.999/0.997/0.993. The rms phase difference is 3/5/5/7°
(valid region, their data conjugated). Our level is 1.0 dB below theirs (median). The figure is in
`results/validation/vesterholm_coarse.png`.

**Reading.** Our solver reproduces their independent BEM (NumCalc ML-FMM, a different mesh discretisation) to
within about 1 dB and a few degrees of phase. Our agreement with the measurement equals theirs (correlation 0.56-0.75,
in line with their reported "> 0.6"). The remaining model-to-measurement gap is therefore common to both
simulations. It comes from the specimen and measurement (post-mortem pinna posture, microphone in the canal,
alignment), not from the solver.

## Full run (after production; not run yet)

    ~/venvs/batertf/bin/python scripts/validate_vesterholm.py
    ~/venvs/batertf/bin/python paper/figures/fig_vesterholm.py

---

# In-flight emission beam of *M. daubentonii* (Häfele et al. 2026) vs our mouth piston

## Source, licence, and how we may use it

- Häfele F.T., Wisniewska D.M., Vesterholm K.K., Jakobsen L. (2026). Echolocating bats sacrifice binaural
  localization cues for target-focused hearing during high-speed foraging. *PNAS*, doi:10.1073/pnas.2605701123.
  Data: Zenodo doi:10.5281/zenodo.20927789. Code: github.com/fhaefele/target-focused-hearing-of-echolocating-bats-pnas
  (commit cf819a9, read only).
- **Licence CC BY-NC-ND 4.0.** We compare and cite only. The beam, and maps or numbers derived from it, are **never
  put into our dataset or the Zenodo export**. The comparison figure in the paper (`paper/figures/hafele_beam.pdf`)
  is our own analysis, cited as such. We keep their beam maps small and attributed there. If the journal or the
  authors object, drop the map row and keep only the half-angle curve.
- Only `BEAM_INFLIGHT_daubentons_1kHz_steps.mat` was downloaded (24,398,873 bytes, md5
  `2fc13fa391fec9e835c999343377893a`, matches the record), to `/data/BatERTF/external/hafele2026/download/` (not
  Dropbox, not git).

## Conventions

- Variable `daub`, 181 x 361 x 122: el = -90:1:90 (rows), az = -180:1:180 (columns; +180 duplicates -180, so their
  `figure_04.m` drops the last column, and we do too). Page 1 is broadband, pages 2..122 are 30..150 kHz in 1 kHz
  steps (`K.Freq = 30e3:1e3:150e3` in `figure_04.m`). Values are dB re max (amplitude dB: `figure_02.m` uses
  `(10.^(beam/20)).^2` as power), floored at -40 dB.
- **Frame:** the flight direction in the wind tunnel, not a head frame. The peak is at az = 0 and el = -15 to -25°
  (row 0 taken as el = -90). The sign of az/el does not matter here: the piston is axisymmetric, and we align its
  axis with the beam's peak direction at each frequency.
- It is a smooth fitted or interpolated model, not raw microphone levels. Above about 100 kHz it has a step (half-angle
  25 -> 44° at 102 kHz) and a lobe stretched vertically. The pole rows are not constant. Treat > 100 kHz with care.
- **Independence:** our piston radius (6.28 mm) was calibrated to field recordings (Surlykke, Pedersen & Jakobsen
  2009, Table 1, 45-75 kHz). This wind-tunnel beam is an independent data set (other bats, other setup).

## Method (`scripts/compare_hafele_beam.py` -> `results/validation/hafele_beam.json`, `paper/figures/hafele_beam.pdf`)

Per frequency (30-150 kHz, 1 kHz):
- piston = `piston.directivity` (baffled piston with obliquity factor), a = 6.28 mm, axis = the beam's peak direction;
- the -6 dB half-angle (solid-angle-equivalent cap, as in `sensitivity_checks.half_angle`);
- DI = 10 log10(max power / mean power over the sphere);
- the pattern correlation and rms dB (`reference.metrics_points`: max-normalised, -30 dB floor, cos(el) weights);
- best-fit radius (a) from the half-angle and (b) from the maximum correlation over 1-15 mm.

## Result

| Band | half-angle piston - beam | DI piston - beam | corr (rms dB) at a = 6.28 mm | best-fit radius: half-angle / corr |
|---|---|---|---|---|
| 45-75 kHz | -15.0° | +4.8 dB | 0.84 (5.4 dB) | 3.5 / 3.6 mm |
| 30-100 kHz | -15.9° | +5.3 dB | 0.84 (5.0 dB) | 3.5 / 3.4 mm |
| 30-150 kHz | -17.3° | +6.9 dB | 0.79 (5.7 dB) | 3.0 / 2.8 mm |

Examples: at 50 kHz the half-angle is 39.7° in flight vs 22.0° for the piston (Surlykke field: 22°). At 80 kHz it
is 25.6° vs 13.7°. The **shape** of the main lobe is piston-like: with a fitted radius the correlation is 0.94-0.98
at 50-80 kHz. The **width** corresponds to an equivalent radius of about 3.5 mm, about half the field-calibrated
6.28 mm. This matches the known beam-width plasticity of vespertilionids: bats in confined spaces (wind tunnel,
lab) widen their beam by reducing gape and/or lowering frequency (Jakobsen & Surlykke 2010, PNAS 107:13930;
Surlykke et al. 2009 vs lab data). Our emitter represents the open-space search beam. The in-flight wind-tunnel beam
is a lower bound on directivity. Both should be stated in the paper, as the range of the oral-emitter
directivity.
