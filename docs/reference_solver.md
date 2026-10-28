# The solver behind the reference ERTFs

Both references were computed with the BEM of Otani & Ise ("BEM3D"; Vanderelst et al. 2010 cite Otani & Ise 2006
[J. Acoust. Soc. Am. 119, 2589-2598, doi:10.1121/1.2191608] and 2003 [Acoust. Sci. Tech. 24, 259-266]; De Mey et al.
2008 has M. Otani as co-author). Full text: `data/1.2191608.pdf` (from Jan, 2026-10-27).

## Formulation (Otani & Ise 2006, Sec. II, Eq. 1)

    (1/2 I + G_n + j w rho G Y) P = g A

- **Conventional boundary integral equation (CBIE):** Helmholtz integral equation with constant elements and
  collocation at the element centres. There is **no Burton-Miller or CHIEF term**, and the paper does not discuss
  fictitious (irregular) frequencies.
- **Dense solution** by LU decomposition or the inverse of the coefficient matrix ("source-independent method",
  postprocessing with the stored inverse). This is why the bat models were limited to about 32,000 triangles
  (Vanderelst et al. 2010).
- **Receivers** are field points near the (blocked) ear canal entrances; the bat studies averaged four of them per
  ear. The emitter used one receiver per nostril, by reciprocity, with the complex fields of both nostrils summed.
- **Mesh criterion:** element length below 1/4-1/6 of the wavelength; the bat meshes have a 0.5 mm maximum edge
  (1/4 wavelength at 150 kHz).
- The paper's own reference results (the source-independent method, identical to the conventional BEM) show many
  steep peaks and dips at the highest frequencies (Figs. 6-8).

## Consequences for the comparison

- A CBIE without Burton-Miller or CHIEF fails near the interior Dirichlet eigenfrequencies of the closed body. In our
  sphere test (tests/gpubem_sphere.py) the CBIE alone gives 15-21 dB errors there, while Burton-Miller stays at
  0.03 dB. For a bat head these frequencies are dense in the ultrasonic band.
- This is consistent with the P. discolor reference: its patterns jump between frequencies 0.5 kHz apart (median
  neighbour correlation 0.93, dropping to 0.74 at 37, 54.5 and 90.5 kHz, against 0.996-0.999 for our Burton-Miller
  BEM), and its maps are speckled above about 70 kHz. The M. microtis reference is smooth (0.998-0.999). We cannot
  tell from the papers why one reference is affected and the other is not; possible causes are the frequencies that
  happened to be computed, the mesh, or the version of the solver.
- Wording for the paper: present this as an explanation consistent with the evidence, not as a proven cause.
