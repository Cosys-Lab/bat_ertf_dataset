"""User API for the released dataset (one species folder: ertf_<frame>.h5 + landmarks.json). Needs numpy and h5py.

    import dataset as bd                   # or: from batertf import dataset as bd
    d = bd.load("Phyllostomus_discolor")
    H = d.interp("left", bd.unit_vector(30, 0))            # [1, n_freqs] complex, at d.freqs
    t, h = d.impulse_response("left", bd.unit_vector(30, 0))
    t, left, right = d.simulate_echoes([[0.5, 0.1, 0, 0.01]], call, fs)

Conventions (as in the data files):
- Frame: X forward, Y left, Z up, metres, origin at the centre of the evaluation sphere. Azimuth positive to the bat's
  left, elevation positive up, u = [cos(el) cos(az), cos(el) sin(az), sin(el)]. Arrays are [el, az, freq].
- H = p / p_ff: pressure at 1 m relative to the free-field pressure of a monopole at the origin with the same volume
  velocity. The propagation delay of 1 m is therefore removed: H describes the head, not the path.
- Time convention exp(+j w t): a delay tau is exp(-j w tau); numpy's irfft gives the impulse response.
- Ears by reciprocity: H_ear(u) is the ear's response to a far-field source in direction u. H_emitter(u) is the
  emitted far field in direction u. Two-way (echo of a point target in direction u): H_emitter(u) * H_ear(u).

Interpolation (direction and frequency) is linear in the complex value (real and imaginary parts), after removing
the delay of the source position p (ear canal entrance, nostrils or mouth; `acoustic_centre`): G = H exp(-j k u.p).
Sources sit 10-20 mm from the origin, so the phase of H changes by up to k |p| * 2.5 deg ~ 1-2 rad between grid
nodes at 160 kHz; linear interpolation of the raw real/imag parts then loses magnitude between nodes, and unwrapped
phase is ill-defined on the sphere. G varies slowly: it is interpolated bilinearly (real/imag) on the 2.5 deg grid
and the delay is restored exactly. Tested by predicting the odd 2.5 deg nodes from a 5 deg subgrid: the error at
fmax drops from ~0.4-0.6 to ~0.07-0.14 (relative rms) for the ears and to <0.1 for the emitters; on the full 2.5 deg
grid it is roughly 4x smaller again (bilinear error ~ spacing^2).
"""
import json
from pathlib import Path

import h5py
import numpy as np

C = 343.0                      # speed of sound used in the simulations [m/s]


def unit_vector(az, el):
    """Unit vectors (..., 3) from azimuth and elevation in degrees."""
    a, e = np.deg2rad(np.asarray(az, float)), np.deg2rad(np.asarray(el, float))
    return np.stack(np.broadcast_arrays(np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)), -1)


def az_el(u):
    """Azimuth and elevation in degrees of (non-zero) vectors (..., 3)."""
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    return np.degrees(np.arctan2(u[..., 1], u[..., 0])), np.degrees(np.arcsin(np.clip(u[..., 2], -1, 1)))


def rotation_matrix(yaw=0.0, pitch=0.0, roll=0.0):
    """R = Rz(yaw) Ry(pitch) Rx(roll), degrees, right-handed: positive yaw turns +X toward +Y (left), positive pitch
    turns +X toward -Z (nose down), positive roll turns +Y toward +Z (left ear up)."""
    a, b, c = np.deg2rad([yaw, pitch, roll])
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    return rz @ ry @ rx


def absorption_db_per_m(f, temperature_c=20.0, humidity_pct=50.0, pressure_kpa=101.325):
    """Atmospheric absorption [dB/m] after ISO 9613-1 (pure-tone formula, oxygen and nitrogen relaxation).
    ISO 9613-1 states its accuracy for 50 Hz-10 kHz; above that it is the usual extrapolation in bioacoustics
    (1.3 dB/m at 40 kHz, 3.3 dB/m at 100 kHz for 20 C, 50 % RH); expect some 10-20 % uncertainty there."""
    f = np.asarray(f, float)
    T, T0, T01 = temperature_c + 273.15, 293.15, 273.16
    pa = pressure_kpa / 101.325
    h = humidity_pct * 10 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151) / pa      # molar concentration of water vapour [%]
    fro = pa * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frn = pa * (T / T0) ** -0.5 * (9 + 280 * h * np.exp(-4.170 * ((T / T0) ** (-1 / 3) - 1)))
    return 8.686 * f ** 2 * (1.84e-11 / pa * (T / T0) ** 0.5 + (T / T0) ** -2.5 * (
        0.01275 * np.exp(-2239.1 / T) / (fro + f ** 2 / fro) + 0.1068 * np.exp(-3352.0 / T) / (frn + f ** 2 / frn)))


def band_taper(f, fmin, fmax, width=5e3):
    """Raised-cosine window: 0 outside [fmin, fmax], rising over [fmin, fmin + width], falling over [fmax - width, fmax]."""
    f = np.asarray(f, float)
    up = np.clip((f - fmin) / width, 0, 1)
    down = np.clip((fmax - f) / width, 0, 1)
    return np.sin(np.pi / 2 * up) ** 2 * np.sin(np.pi / 2 * down) ** 2


def load(folder, frame="ct"):
    """Load a species folder of the released dataset (reads ertf_<frame>.h5 and landmarks.json)."""
    return ERTF(folder, frame)


class _LazyH(dict):
    """source name -> complex H [el, az, f], read from the HDF5 file on first access."""

    def __init__(self, path, names):
        super().__init__()
        self._path, self._names = path, list(names)

    def __missing__(self, name):
        if name not in self._names:
            raise KeyError(f"{name}: not in {self._names}")
        with h5py.File(self._path, "r") as h:
            self[name] = h[name]["H"][()]
        return self[name]

    def keys(self):
        return list(self._names)

    def __iter__(self):
        return iter(self._names)

    def __len__(self):
        return len(self._names)


class ERTF:
    """Transfer functions of one species. Attributes:
    freqs [Hz], az, el [deg] (grid), H (dict-like: source -> complex [el, az, f] on the grid, dataset frame),
    sources (names: left, right, emitter, left_alt1..4, ...), landmarks [m] (in the current head orientation),
    meta (landmarks.json), attrs (HDF5 attributes), orientation (3x3, see rotate)."""

    def __init__(self, folder, frame="ct"):
        self.folder = Path(folder)
        self.path = self.folder / f"ertf_{frame}.h5"
        self.frame = frame
        with h5py.File(self.path, "r") as h:
            self.freqs = h["freqs"][()].astype(float)
            self.az = h["azimuth"][()].astype(float)
            self.el = h["elevation"][()].astype(float)
            self.sources = [n for n in h if isinstance(h[n], h5py.Group) and "H" in h[n]]
            self.attrs = {k: h.attrs[k] for k in h.attrs}
            self.source_attrs = {n: {k: h[n].attrs[k] for k in h[n].attrs} for n in self.sources}
        js = self.folder / "landmarks.json"
        self.meta = json.loads(js.read_text()) if js.exists() else {}
        lm = self.meta.get("frames", {}).get(frame, {}).get("landmarks_m", {})
        self._landmarks = {k: np.asarray(v, float) for k, v in lm.items()}
        self.H = _LazyH(self.path, self.sources)
        self.orientation = np.eye(3)
        self._centres, self._G = {}, {}
        self.species = self.meta.get("species", str(self.attrs.get("species", self.folder.name)))
        self.daz, self.del_ = self.az[1] - self.az[0], self.el[1] - self.el[0]

    def __repr__(self):
        return (f"ERTF({self.species!r}, frame={self.frame!r}, {self.freqs[0] / 1e3:g}-{self.freqs[-1] / 1e3:g} kHz "
                f"({len(self.freqs)}), grid {len(self.el)}x{len(self.az)}, sources={self.sources})")

    @property
    def landmarks(self):
        return {k: v @ self.orientation.T for k, v in self._landmarks.items()}

    def rotate(self, R):
        """Copy with the head turned by R (3x3; columns = the head's X, Y, Z axes in the new frame). Directions
        passed to interp etc. are then in the new frame: H_new(u) = H(R^T u). rotate(rotation_matrix(yaw=30))
        turns the head 30 deg to the left, so the old azimuth 0 is now at azimuth +30. Rotations compose."""
        new = object.__new__(ERTF)
        new.__dict__.update(self.__dict__)
        new.orientation = np.asarray(R, float) @ self.orientation
        return new

    def acoustic_centre(self, source):
        """Point p (dataset frame, m) whose delay is removed before interpolation: the landmark of the source
        (ear_left, ear_right, mean of the nostrils, or mouth; the *_alt* sources use their nominal source's
        landmark, a few mm away). Without landmarks: least-squares fit of the |H|^2-weighted group delay
        tau(u) = -u.p / c over all directions (less effective for ears: diffraction behind the head biases it)."""
        if source not in self._centres:
            key = {"left": "ear_left", "right": "ear_right"}.get(source.split("_alt")[0])
            lm = self._landmarks
            if key is None:
                key = "nostrils" if "nostrils" in lm else "mouth"
            if key in lm:
                self._centres[source] = np.atleast_2d(lm[key]).mean(0)
            else:
                H = self.H[source].reshape(-1, len(self.freqs))
                u = unit_vector(*np.meshgrid(self.az, self.el)).reshape(-1, 3)
                df = np.diff(self.freqs)
                step = np.isclose(df, df.min())                          # pairs at the finest frequency step
                cross = (H[:, 1:] * np.conj(H[:, :-1]))[:, step].sum(1)   # phase step ~ dk u.p
                sw = np.sqrt(np.abs(cross) / np.abs(cross).max())
                up = np.angle(cross) / (2 * np.pi * df.min() / C)
                self._centres[source] = np.linalg.lstsq(u * sw[:, None], up * sw, rcond=None)[0]
        return self._centres[source]

    def _centred(self, source):
        """H with the delay of the acoustic centre removed, G = H exp(-j k u.p), on the grid (cached)."""
        if source not in self._G:
            u = unit_vector(*np.meshgrid(self.az, self.el))
            self._G[source] = self.H[source] * np.exp(
                -1j * 2 * np.pi * self.freqs / C * (u @ self.acoustic_centre(source))[..., None]).astype(np.complex64)
        return self._G[source]

    def _grid_sample(self, data, u):
        """Bilinear interpolation of data [el, az, ...] at unit vectors u (..., 3) in the dataset frame."""
        az, el = az_el(u)
        x = ((az - self.az[0]) % 360) / self.daz
        y = np.clip((el - self.el[0]) / self.del_, 0, len(self.el) - 1 - 1e-9)
        i0, j0 = np.floor(y).astype(int), np.floor(x).astype(int) % len(self.az)
        i1, j1 = np.minimum(i0 + 1, len(self.el) - 1), (j0 + 1) % len(self.az)
        wy, wx = (y - np.floor(y))[..., None], (x - np.floor(x))[..., None]
        return ((1 - wy) * ((1 - wx) * data[i0, j0] + wx * data[i0, j1])
                + wy * ((1 - wx) * data[i1, j0] + wx * data[i1, j1]))

    def interp(self, source, directions, freqs=None):
        """H of `source` at unit vectors `directions` (..., 3) (current head orientation), at `freqs` [Hz]
        (default: the data frequencies). Returns complex (..., n_freqs); 0 outside the simulated band."""
        u = np.asarray(directions, float) @ self.orientation           # to the dataset frame: R^T u
        u = u / np.linalg.norm(u, axis=-1, keepdims=True)
        p = self.acoustic_centre(source)
        G = self._grid_sample(self._centred(source), u)
        f = self.freqs if freqs is None else np.asarray(freqs, float)
        if freqs is not None:
            g2 = G.reshape(-1, len(self.freqs))
            G = np.stack([np.interp(f, self.freqs, g.real, 0, 0) + 1j * np.interp(f, self.freqs, g.imag, 0, 0)
                          for g in g2]).reshape(G.shape[:-1] + (len(f),))
        return G * np.exp(1j * 2 * np.pi * f / C * (u @ p)[..., None])

    def two_way(self, directions, freqs=None, emitter="emitter", ears=("left", "right")):
        """Two-way ERTF emitter x ear for each ear: dict ear -> complex (..., n_freqs)."""
        e = self.interp(emitter, directions, freqs)
        return {ear: e * self.interp(ear, directions, freqs) for ear in ears}

    def impulse_response(self, source, directions, fs=500e3, n=1024, t0=0.2e-3, taper=5e3):
        """Impulse response(s) (..., n) of `source` towards `directions`, sampled at fs, by irfft of H on the
        FFT grid with a raised-cosine taper of width `taper` [Hz] at both band edges (no data below fmin =
        20 kHz or above fmax). Delayed by t0 so that arrivals before the origin (sources ahead of the origin
        towards u) stay causal. Returns t [s] (t = 0: arrival of the free-field wave at the origin) and h."""
        f = np.fft.rfftfreq(n, 1 / fs)
        H = self.interp(source, directions, f) * band_taper(f, self.freqs[0], min(self.freqs[-1], fs / 2), taper)
        h = np.fft.irfft(H * np.exp(-1j * 2 * np.pi * f * t0), n)
        return np.arange(n) / fs - t0, h

    def simulate_echoes(self, reflectors, call, fs, c=C, temperature_c=20.0, humidity_pct=50.0, absorption=True,
                        emitter="emitter", ears=("left", "right"), taper=5e3, n=None):
        """Echoes of point reflectors at the ears, frequency domain, far field (range >> head size).
        reflectors: [M, 4] rows (x, y, z, a): position in the head frame (current orientation) [m] and echo
        amplitude a = echo pressure at 1 m from the reflector / incident pressure (target strength 20 log10 a dB).
        call: emitted signal at fs, as free-field monopole pressure at 1 m (the emitter's own directivity and
        |H| ~ 2 on axis come from H_emitter). Each echo: call * H_emitter(u) * H_ear(u) * a / r^2 * exp(-j w 2r/c)
        * 10^(-alpha(f) 2r / 20), u = x/r. Returns t [s] (t = 0: call onset at the origin) and one signal per ear."""
        refl = np.atleast_2d(np.asarray(reflectors, float))
        pos, amp = refl[:, :3], refl[:, 3]
        r = np.linalg.norm(pos, axis=1)
        u = pos / r[:, None]
        if n is None:
            n = int(2 ** np.ceil(np.log2(len(call) + 2 * r.max() / c * fs + 0.002 * fs)))
        f = np.fft.rfftfreq(n, 1 / fs)
        S = np.fft.rfft(call, n) * band_taper(f, self.freqs[0], min(self.freqs[-1], fs / 2), taper)
        gain = amp[:, None] / r[:, None] ** 2 * np.exp(-1j * 2 * np.pi * f * 2 * r[:, None] / c)
        if absorption:
            gain = gain * 10 ** (-absorption_db_per_m(f, temperature_c, humidity_pct) * 2 * r[:, None] / 20)
        tw = self.two_way(u, f, emitter, ears)
        return (np.arange(n) / fs,) + tuple(np.fft.irfft((tw[e] * gain).sum(0) * S, n) for e in ears)
