"""Tests of the user API batertf/dataset.py on a synthetic species folder in the release layout (piston emitter,
ears = point receivers with a smooth directivity). Run with pytest or as a script:
    python tests/test_dataset_api.py [species_folder]
A real species folder (optional) is also checked: interpolation at the grid nodes returns the stored values."""
import json
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batertf import dataset as bd
from batertf import piston

CENTRE = np.array([0.012, 0.0, -0.004])            # mouth
AXIS = np.array([np.cos(np.deg2rad(-20)), 0, np.sin(np.deg2rad(-20))])
EARS = {"left": np.array([-0.006, 0.008, -0.006]), "right": np.array([-0.006, -0.008, -0.006])}


def ear(u, f, p):
    """Point receiver at p with a cardioid-like gain toward its side: smooth in direction, exact delay -u.p/c."""
    g = 1 + 0.5 * (u @ (p / np.linalg.norm(p)))
    return g[..., None] * np.exp(1j * 2 * np.pi * f / bd.C * (u @ p)[..., None])


def make_folder(root):
    az, el = np.arange(-180, 179.99, 2.5), np.arange(-90, 90.01, 2.5)
    f = np.arange(20e3, 160e3 + 1, 2e3)
    u = bd.unit_vector(*np.meshgrid(az, el))
    d = Path(root) / "Synthetic_bat"
    d.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / "ertf_ct.h5", "w") as h:
        h["freqs"], h["azimuth"], h["elevation"] = f, az, el
        h["emitter/H"] = piston.transfer(u, f, CENTRE, AXIS, 0.006).astype(np.complex64)
        for k, p in EARS.items():
            h[f"{k}/H"] = ear(u, f, p).astype(np.complex64)
        h.attrs["species"] = "Synthetic bat"
    lm = {"ear_left": EARS["left"].tolist(), "ear_right": EARS["right"].tolist(), "mouth": CENTRE.tolist()}
    (d / "landmarks.json").write_text(json.dumps({"species": "Synthetic bat", "frames": {"ct": {"landmarks_m": lm}}}))
    return d


_TMP = tempfile.TemporaryDirectory()
D = bd.load(make_folder(_TMP.name))


def test_grid_nodes():
    U = bd.unit_vector(*np.meshgrid(D.az, D.el))
    for s in ("left", "emitter"):
        assert np.allclose(D.interp(s, U[::7, ::5]), D.H[s][::7, ::5], atol=1e-5)


def test_between_nodes():
    """Off-grid directions and frequencies against the analytical model."""
    rng = np.random.default_rng(1)
    u = rng.normal(size=(200, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    f = np.array([31.3e3, 99.1e3, 157.7e3])
    err = np.abs(D.interp("left", u, f) - ear(u, f, EARS["left"])).max()
    ref = piston.transfer(u, f, CENTRE, AXIS, 0.006)
    err_e = np.abs(D.interp("emitter", u, f) - ref).max() / np.abs(ref).max()
    assert err < 2e-3 and err_e < 0.05, (err, err_e)          # piston ka ~ 17 at 158 kHz: narrow lobes


def test_rotation():
    u = bd.unit_vector([10, -40, 120], [5, 30, -60])
    assert np.allclose(D.rotate(np.eye(3)).interp("emitter", u), D.interp("emitter", u))
    R = bd.rotation_matrix(yaw=30, pitch=-10, roll=5)
    assert np.allclose(D.rotate(R).interp("left", u @ R.T), D.interp("left", u), atol=1e-6)   # u_new = R u
    assert np.allclose(D.rotate(R).rotate(R.T).interp("left", u), D.interp("left", u), atol=1e-6)
    Dy = D.rotate(bd.rotation_matrix(yaw=30))                       # head turned left: its axis moves to az +30
    assert np.allclose(Dy.landmarks["mouth"], bd.rotation_matrix(yaw=30) @ CENTRE)
    assert np.isclose(np.abs(Dy.interp("left", bd.unit_vector(30, 0))), np.abs(D.interp("left", bd.unit_vector(0, 0)))).all()


def test_piston_impulse_response():
    fs, t0 = 1e6, 0.2e-3
    t, h = D.impulse_response("emitter", AXIS, fs=fs, n=2048, t0=t0)
    expected = -AXIS @ CENTRE / bd.C                                  # mouth ahead of the origin: arrives early
    assert abs(t[np.argmax(h)] - expected) <= 1 / fs, (t[np.argmax(h)], expected)


def test_echo_delay():
    fs, r = 500e3, 1.0
    call = np.zeros(64)
    call[0] = 1.0
    pos = r * AXIS
    t, left, right = D.simulate_echoes([[*pos, 1.0]], call, fs, absorption=False)
    expected = 2 * r / bd.C - AXIS @ (CENTRE + EARS["left"]) / bd.C
    assert abs(t[np.argmax(np.abs(left))] - expected) <= 1 / fs, (t[np.argmax(np.abs(left))], expected)
    t, left_abs, _ = D.simulate_echoes([[*pos, 1.0]], call, fs, absorption=True)
    assert np.abs(left_abs).max() < np.abs(left).max()


def test_absorption():
    a = bd.absorption_db_per_m([1e3, 40e3, 100e3], 20, 50)
    assert abs(a[0] / 4.66e-3 - 1) < 0.03, a                        # ISO 9613-1 table: 4.66 dB/km
    assert 0.8 < a[1] < 2.0 and 2.0 < a[2] < 4.5, a


def check_real(folder):
    d = bd.load(folder)
    print(d)
    U = bd.unit_vector(*np.meshgrid(d.az, d.el))
    for s in ("left", "right", "emitter"):
        e = np.abs(d.interp(s, U[1:-1:9, ::11]) - d.H[s][1:-1:9, ::11]).max()
        print(f"  {s}: max |interp - H| at grid nodes = {e:.2e}")
        assert e < 1e-4 * np.abs(d.H[s]).max()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
    if len(sys.argv) > 1:
        check_real(sys.argv[1])
