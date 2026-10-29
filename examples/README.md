# Using the Bat ERTF dataset

A small Python API (`batertf/dataset.py`, needs numpy and h5py; matplotlib and scipy for the examples) and equivalent
MATLAB functions that work on a released species folder (`ertf_ct.h5`, `landmarks.json`, ...). In the Zenodo
package they are in `code/`; on GitHub in `batertf/` and `examples/`.

## Conventions

| | |
|---|---|
| Frame | X forward (snout), Y left, Z up, metres; origin = centre of the evaluation sphere. `mesh.stl` is the head in exactly these coordinates. |
| Directions | azimuth positive to the bat's **left**, elevation positive up; u = [cos(el) cos(az), cos(el) sin(az), sin(el)]. Grid: az -180..177.5, el -90..90, 2.5 deg; arrays `[el, az, freq]` (MATLAB after `loadBatERTF`: same). |
| Referencing | H = p / p_ff at 1 m, p_ff = free-field pressure of a monopole **at the origin** with the source's volume velocity. The 1 m propagation delay and 1/r spreading are therefore not in H; only the head is. |
| Time convention | exp(+j w t): a delay tau multiplies H by exp(-j w tau). The inverse real FFT (`numpy.fft.irfft`, MATLAB `ifft(..., 'symmetric')`) gives the impulse response. (The solver works in exp(-j w t); the stored data are conjugated. In the exp(+j w t) convention the reference monopole is p_ff = j w rho Q exp(-j k r) / (4 pi r).) |
| Ears | by reciprocity: H_ear(u) = response of the ear canal entrance to a far-field source in direction u. |
| Emitter | H_emitter(u) = emitted far field in direction u (nasal: two in-phase nostril patches; oral species: analytical piston, |H| = 2 on axis). |
| Two-way | echo of a point target in direction u: H_emitter(u) * H_ear(u) (both referenced to the origin, so the product is the round trip relative to the origin). |
| Causality | sources and ears sit 10-20 mm from the origin, so for directions on their side the wave arrives up to ~60 us **before** it reaches the origin: an impulse response from H is not causal around t = 0. `impulse_response` adds a pre-delay `t0` (default 0.2 ms) and returns t with t = 0 at the origin. |
| Band | 20 kHz to 135-160 kHz (species dependent), 0.5 kHz steps. Outside the band H is unknown: impulse responses and echoes are band-limited with a 5 kHz raised-cosine taper at both edges. |

### Interpolation

`interp` is bilinear in direction and linear in frequency on the real and imaginary parts of
G = H exp(-j k u.p), where p is the source's landmark (ear canal entrance, mean of the nostrils, or mouth). The raw
phase changes by 1-2 rad between neighbouring 2.5 deg nodes at the top of the band, so interpolating raw real/imag
loses magnitude between nodes, and unwrapped phase is ill-defined on the sphere; G is smooth and the removed delay is
restored exactly. With this, predicting the odd 2.5 deg nodes from a 5 deg subgrid gives a relative rms error at the
top of the band of 0.07-0.14 for the ears and < 0.1 for the emitters (0.4-0.6 without), and about 4x less at the
native 2.5 deg spacing. Use the native grid (`d.H[source]`) when you do not need off-grid directions.

### Head rotation

`d.rotate(R)` returns the dataset with the head turned by R (3x3, columns = the head's X, Y, Z axes in the new
frame): H_new(u) = H(R^T u). `rotate(rotation_matrix(yaw=30))` turns the head 30 deg to the left, so its old
azimuth 0 is at azimuth +30. Positive pitch is nose down (right-handed about +Y), positive roll lifts the left ear.

### Echo simulation

`d.simulate_echoes(reflectors, call, fs)` (Python) and `simulateEchoes.m` with `exampleEchoSimulation.m` (MATLAB): point reflectors with
rows (x, y, z, a) in the head frame [m]; a = echo pressure at 1 m from the reflector relative to the incident pressure
(target strength 20 log10 a). Each echo at an ear is call * H_emitter(u) * H_ear(u) * a / r^2 * exp(-j w 2r/c) *
10^(-alpha(f) 2r / 20), with the call given as free-field monopole pressure at 1 m and r, u from the origin. This is
a far-field model (r >> head size, no near-field or multiple reflections). alpha is the ISO 9613-1 atmospheric
absorption (stated for up to 10 kHz; above that the usual extrapolation, about 10-20 % uncertain, e.g. 1.3 dB/m at
40 kHz and 3.3 dB/m at 100 kHz for 20 C, 50 % RH).

## Python

    python examples/python/01_plot_directivity.py <species folder> [--freq 60000] [--out fig.png]
    python examples/python/02_impulse_responses.py <species folder> [--az 30 --el -10]
    python examples/python/03_echo_simulation.py <species folder>
    python examples/python/04_head_rotation.py <species folder>

In the Zenodo package the species folder defaults to `../Phyllostomus_discolor` (relative to `code/`).

```python
from batertf import dataset as bd
d = bd.load("Phyllostomus_discolor")
d.H["left"]                                          # complex [el, az, f] on the grid
u = bd.unit_vector([0, 30], [0, -10])               # two directions
H = d.interp("left", u, freqs=[40e3, 80e3])          # [2, 2]
tw = d.two_way(u)                                    # {'left': ..., 'right': ...}
t, h = d.impulse_response("right", u, fs=500e3, n=1024)
t, left, right = d.simulate_echoes([[0.5, 0.1, 0.0, 0.05]], call, fs=500e3)
```

`d.sources` also lists the placement ensemble (`left_alt1..4`, ...): the same quantities for alternative ear and
nostril positions, to judge how much a result depends on that modelling choice.

## MATLAB

Functions: `loadBatERTF` (load a species folder), `interpolateERTF` (any direction and frequency),
`calculateImpulseResponse`, `simulateEchoes` (point reflectors, both ears), `calculateUnitVectors`,
`calculateAirAbsorption`, `calculateBandTaper` and `findSpeciesFolder`. Example scripts: `exampleDirectivity.m`,
`exampleImpulseResponses.m`, `exampleEchoSimulation.m` and `exampleHeadRotation.m`.

Where the data are: set `folderSpecies` in the workspace before running an example, for instance
`folderSpecies = 'D:\data\BatERTF\Myotis_daubentonii';`. Otherwise `findSpeciesFolder` looks in the environment
variable `BAT_ERTF_DATA` (a species folder or the dataset root) and then relative to the script, as in the Zenodo
layout (`<root>/code/examples/matlab`).

Base MATLAB R2017b or newer; the spectrograms of `exampleEchoSimulation.m` need the Signal Processing Toolbox. h5py
stores complex numbers as an HDF5 compound {r, i}, which `h5read` returns as a struct; `loadBatERTF` converts it and
permutes MATLAB's reversed dimension order back to `[el, az, f]`. The SOFA files can be read with the SOFA toolbox
instead (`SOFAload`, see `exampleDirectivity.m`).
