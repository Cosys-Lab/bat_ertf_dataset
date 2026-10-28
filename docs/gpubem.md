# GPU BEM solver (`batertf/gpubem.py`)

A dense boundary element solver for the 3D exterior Helmholtz problem on the GPU. It is a drop-in alternative to
NumCalc for the bat ERTFs. Its output is referenced and conjugated exactly like `numcalc.read_source`.

```python
from batertf import gpubem, grid
pts, _ = grid.unique_points("hemisphere")
H = gpubem.solve(m, {"left": [faces_l], "emitter": [nostril1, nostril2]}, freqs, pts)   # {source: (P, F)}
H_grid = gpubem.to_grid(H["left"], "hemisphere")                                         # (el, az, F)
```

All sources of one frequency share one matrix assembly. Production uses `solve_lu` (in-place LU, then one forward/backward substitution per source); the batched GMRES of `solve` (below) is kept for reference, as it did not converge reliably on the CT meshes.

## Formulation (signs verified on the sphere)

- Time convention exp(-iωt), G = exp(ikr)/(4πr), R = x − y, r = |R|. The normal n points out of the body into the
  fluid.
- Rigid body (Neumann) with vibrating patches: q = ∂p/∂n = iωρ v_n. This follows from Euler for exp(-iωt):
  ∇p = iωρ v. With v_n = 0.1 m/s on the patches and 0 elsewhere, the ρ, c and v_n values come from `numcalc.py`.
- Field: p(x) = Σ_j [p_j ∂G/∂n_y(x, y_j) − G(x, y_j) q_j] A_j. On the surface this gives the CBIE
  (½I − K) p = −S q. The normal derivative at x gives the HBIE W p = (½I + K') q.
- Burton–Miller: (½I − K − αW) p = −S q − α(½I + K') q, with α = +i/k. The right-hand side only needs the columns of S
  and K' on the patch faces, because q is zero elsewhere.
- Kernels: ∂G/∂r = G(ik − 1/r), ∂G/∂n_y = −G'(R·n_y)/r, ∂G/∂n_x = G'(R·n_x)/r, and
  ∂²G/∂n_x∂n_y = −[G''(R·n_x)(R·n_y)/r² + G'(n_x·n_y/r − (R·n_x)(R·n_y)/r³)], with G'' = G[(ik − 1/r)² + 1/r²].
- Discretisation: piecewise-constant flat triangles with collocation at the centroids, the same as NumCalc.
- The sign of α: the plain CBIE (α = 0) fails at the discrete irregular frequency. Both α = +i/k and α = −i/k remove
  it, and give identical accuracy and smallest singular values (table B). This matches Burton–Miller theory:
  uniqueness only requires Im α ≠ 0. **The claim "the wrong sign fails" was not reproduced.** +i/k is the usual
  choice for exp(-iωt) and is the default (`ALPHA_SIGN`).

## Quadrature

- **Self terms**, for a flat triangle with collocation at its centroid, use polar coordinates around the centroid.
  R(θ) is the distance from the centroid to the boundary. Each edge is integrated with R = d_e/cos φ and 16-point
  Gauss–Legendre in φ.
  - K_ii = K'_ii = 0 (R·n = 0 in the plane).
  - S_ii = (1/4π)∮(e^{ikR} − 1)/(ik) dθ. The radial integral is exact.
  - W_ii: in the plane the kernel is e^{ikr}(1/r³ − ik/r²)/4π = d/dr(−e^{ikr}/r)/4π. Its Hadamard finite part over
    [0, R] is ik − e^{ikR}/R, so W_ii = ik/2 − (1/4π)∮e^{ikR}/R dθ. The k → 0 limit is FP∫1/r³ dA = −∮dθ/R.
  - Checked numerically against brute-force quadrature: Im S_ii matches to 1e-12, and W_ii(k) − W_ii(0), which is a
    regular integral, matches to 0.3 %.
- **Far pairs**: a 3-point (degree 2) rule on the source triangle.
- **Near pairs** are found with cKDTree on the centroids. Distance d is measured in source-element sizes
  s = √(4A/√3):
  - d < 1.5 s: the 3-point rule on 16² = 256 sub-triangles (768 points).
  - 1.5 s ≤ d < 4 s: the 3-point rule on 6² = 36 sub-triangles.
  - On the 46k-face head these are 0.61 M and 5.9 M pairs.
- **Evaluation points** at 1 m use the 3-point rule.

## Linear algebra

- The matrix is assembled densely on the GPU in complex64, 256 rows at a time, with the near-pair entries
  overwritten afterwards. For 46k faces it takes 16.8 GB.
- Rows are Jacobi-scaled by 1/(½ − αW_ii).
- The solver is a batched restarted GMRES written in torch: one Krylov space per source, all sources multiplied in
  one GEMM, CGS2 orthogonalisation, Givens rotations in complex128, restart 60, tolerance 1e-5 on the relative
  residual.
- `direct=True` uses LU instead. This is used for the irregular-frequency demonstration, because GMRES started from
  zero barely excites the near-null space of the CBIE.

## Validation A: sphere vs the series solution (`tests/gpubem_sphere.py`)

- The setup is the same as `tests/sphere_check.py`: a = 5 mm, a cap patch of radius 0.8 mm (`disc_patch`), the
  analytic cap with equal area, points at 1 m on the circle z = 0, referenced and conjugated like
  `numcalc.read_source`.
- Angles are measured from the area-weighted axis of the faceted cap, which is 1.2–2.2° off +X. Measuring from +X
  produced a spurious, non-converging error of 1.7 dB and 21° near the shadow-side minimum at ka = 11.
- The table gives the maximum/rms error. "Front" means θ ≤ 90°; "full" means 0–180°.

| f (kHz) | ka | sub 4 (5120 f) front | sub 4 full | sub 5 (20480 f) front | sub 5 full |
|---|---|---|---|---|---|
| 20 | 1.83 | 0.04/0.03 dB, 0.1/0.1° | 0.04/0.03 dB, 0.5/0.3° | 0.02/0.02 dB, 0.0/0.0° | 0.03/0.01 dB, 0.3/0.2° |
| 34.3 | 3.14 | 0.04/0.03 dB, 0.2/0.1° | 0.06/0.03 dB, 0.7/0.3° | 0.02/0.02 dB, 0.0/0.0° | 0.04/0.02 dB, 0.5/0.2° |
| 40 | 3.66 | 0.04/0.03 dB, 0.2/0.1° | 0.06/0.03 dB, 0.8/0.3° | 0.02/0.01 dB, 0.1/0.0° | 0.04/0.02 dB, 0.5/0.2° |
| 80 | 7.33 | 0.03/0.02 dB, 0.3/0.2° | 0.27/0.05 dB, 2.0/0.5° | 0.05/0.02 dB, 0.1/0.1° | 0.09/0.04 dB, 0.6/0.2° |
| 120 | 10.99 | 0.06/0.03 dB, 1.0/0.5° | 0.91/0.16 dB, 5.7/1.3° | 0.10/0.05 dB, 0.3/0.1° | 0.53/0.11 dB, 1.9/0.5° |

- Target: < 0.3 dB and < 10° up to ka ≈ 11 at subdivision 5. It is met over the front half. Over the full circle it
  is met except at the 120 kHz shadow-side minimum, where sub 5 reaches 0.53 dB (at about −18 dB).
- Refinement from sub 4 to sub 5 roughly halves the full-circle error at 80 and 120 kHz.
- For comparison, NumCalc in `sphere_check.py` had < 0.21 dB and < 7.6° on the front half.

### Validation B: the irregular frequency (sub 4, LU solve)

- f* is the discrete CBIE eigenfrequency near ka = π, found as the minimum of the CBIE matrix's smallest singular
  value. The faceted sphere shifts it from 34.300 to 34.333 kHz.
- Columns: far-field rms/max dB error | error of the surface-mean pressure (the n = 0 term, where the spurious
  uniform mode lives) | σ_min of the Jacobi-scaled matrix.

| f (Hz) | CBIE | BM +i/k | BM −i/k |
|---|---|---|---|
| 34000 | 0.01/0.03 dB, 0.007, 6.4e-2 | 0.03/0.06 dB, 0.010, 0.11 | 0.03/0.06 dB, 0.010, 0.11 |
| 34300 | 0.16/0.36 dB, 0.079, 6.3e-3 | 0.03/0.06 dB, 0.010, 0.11 | 0.03/0.06 dB, 0.010, 0.11 |
| **34333 (f\*)** | **14.9/21.2 dB, 20.3, 3.0e-5** | 0.03/0.06 dB, 0.010, 0.11 | 0.03/0.06 dB, 0.010, 0.11 |
| 34600 | 0.02/0.05 dB, 0.011, 5.1e-2 | 0.03/0.06 dB, 0.011, 0.11 | 0.03/0.06 dB, 0.011, 0.11 |

At f\*, the CBIE alone fails by 15 dB rms. Both Burton–Miller variants are unaffected, with flat σ_min of about 0.11
(figure: `results/validation/gpubem_sphere.png`, right column).

## Validation C: bat heads vs NumCalc (`tests/gpubem_vs_numcalc.py`)

- Each comparison uses the same mesh (`mesh.load`), the same patch faces (`patch_faces_original` from the run's
  `project.json`) and the same evaluation points as the NumCalc run.
- Metrics are computed over the directions where NumCalc is above (its maximum − 40 dB):
  - rms, 99th-percentile and maximum dB difference;
  - magnitude-weighted rms phase difference;
  - Pearson correlation of the dB patterns.
- Full tables are in `results/validation/gpubem_vs_numcalc_<run>.txt`.

### `pilot_pdisc`: P. discolor, hemisphere grid

| source | 30 kHz | 45 | 60 | 75 | 90 |
|---|---|---|---|---|---|
| left (rms/max dB, phase°) | 0.03/0.44, 0.2 | 0.06/0.47, 0.2 | 0.03/0.25, 0.2 | 0.05/0.37, 0.1 | 0.04/0.26, 0.1 |
| right | 0.02/0.22, 0.1 | 0.06/0.48, 0.2 | 0.04/0.27, 0.2 | 0.07/0.57, 0.2 | 0.03/0.22, 0.1 |
| emitter | 0.01/0.04, 0.0 | 0.01/0.17, 0.0 | 0.01/0.06, 0.0 | 0.01/0.09, 0.0 | 0.01/0.17, 0.0 |

Pattern correlation is 1.0000 in every case.

### `pilot2_pdisc` (37 sources, full sphere grid) and `pilot2_mmicr` (M. microtus, 46100 faces)

- `pilot2_pdisc`: the median rms is 0.01 dB and the maximum rms is 0.11 dB over all sources and frequencies. The
  99th percentile is ≤ 0.44 dB, the phase rms is ≤ 0.4° and the correlation is ≥ 0.9999. The largest single-point
  differences are 3.3 dB (earL3, 30 kHz) and 2.1 dB (earR3, 60 kHz); both are in deep pattern nulls.
- `pilot2_mmicr`: only 125 and 150 kHz had results, because that run was still in progress. At 150 kHz, over 28
  sources, the rms is ≤ 0.04 dB, the maximum is ≤ 0.35 dB and the phase rms is 0.1°.

Ear sources differ about 3–10 times more than nose sources (0.03–0.1 dB versus 0.01 dB rms). They sit deep in the
ear canal, where near-singular integration matters most; the two codes treat it differently.

NumCalc uses the same discretisation (constant elements, collocation, Burton–Miller), so this agreement shows that the
two implementations are equivalent. It does not measure the absolute error. The absolute accuracy comes from the
sphere and from mesh resolution: the P. discolor mean edge is 0.42 mm, about 9.1 elements per wavelength at
90 kHz; the M. microtus mean edge is 0.28 mm, about 8.1 at 150 kHz.

## Timing (RTX PRO 5000 Blackwell, 48 GB; P. discolor, 45764 faces)

| | per frequency |
|---|---|
| setup (near-pair search, once per mesh) | 0.9 s |
| matrix assembly (all 2.1e9 entries + 6.5 M near pairs) | 3.8–4.0 s |
| GMRES, 1 source | 2.2 s (≈ 110–175 iterations) |
| GMRES, 3 / 8 / 16 / 32 / 37 sources | 2.5 / 2.8 / 3.3 / 5.9 / 8.5 s |
| field at 5.3k (hemisphere) – 10.4k (sphere) points | 0.2–0.5 s |
| peak GPU memory | 19.8 GB (46k faces), 24.6 GB (46.1k faces, 39 sources) |

- Cost per additional source: about 0.15–0.2 s up to 37 sources.
- One frequency with 3 sources (left, right, emitter) takes about 6.5 s. With 37 sources it takes about 13 s.
- NumCalc takes about 60–120 s per frequency and source on one core.
- Throughput comparison:
  - 3 sources: the GPU is about 30–50 times faster than one NumCalc process.
  - 37 sources: it is about 200–300 times faster than one core, and still about 15–20 times faster than NumCalc
    running 14 processes in parallel.
- Assembly time does not depend on frequency. GMRES iterations stay at about 100–175 from 30 to 150 kHz.

## Limitations and notes

- **Memory**: N² × 8 bytes. 46k faces need 17 GB, and the 48 GB card can hold up to about 70k faces. Larger meshes
  would need fast multipole/H-matrix compression or an out-of-core matrix.
- **Precision**: float32/complex64 throughout. This is adequate at GMRES tolerance 1e-5, given the 0.01 dB
  agreement with NumCalc (double precision).
- **Quadrature**: the 3-point far-field rule is sufficient beyond 4 element sizes at these mesh resolutions (5–6+
  elements per wavelength). For coarser meshes or higher ka, use `asm_kw={"far_rule": gpubem._R7}`.
- **Near-singular integration** uses only subdivision, with no analytic or Duffy treatment. Adjacent pairs use 768
  points. The closest quadrature point is about 2.3 sub-element sizes from the collocation point. This was not
  verified separately; the evidence is indirect, from the sphere and NumCalc agreement.
- **Surface pressure** near the cap edge has O(h) element-to-element noise: 16 / 7.6 / 4.3 % at sub 3 / 4 / 5. This
  comes from the jagged faceted patch boundary, and the CBIE shows it too. The far field is not affected.
- **Irregular frequencies**: tested only at the first (n = 0) eigenfrequency of the sphere.
- **Sharing the GPU**: the solver uses the whole GPU while it runs. Other jobs on the GPU need about 25 GB free.
