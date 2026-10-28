"""Impulse responses and magnitude spectra of both ears for one direction (bilinear interpolation on the sphere,
band-limited 20 kHz..fmax with raised-cosine tapers). t = 0 is the arrival of the free-field wave at the origin.
Usage: python 02_impulse_responses.py [species_folder] [--az 30] [--el -10] [--out f.png]"""
import matplotlib.pyplot as plt
import numpy as np

from _setup import args, bd, finish

a = args(__doc__, az=30.0, el=-10.0)
d = bd.load(a.species)
u = bd.unit_vector(a.az, a.el)
fs = 500e3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for ear in ("left", "right"):
    t, h = d.impulse_response(ear, u, fs=fs, n=1024, t0=0.2e-3)
    ax1.plot(t * 1e3, h, label=ear)
    H = d.interp(ear, u)                                     # at the simulated frequencies
    ax2.plot(d.freqs / 1e3, 20 * np.log10(np.abs(H)), label=ear)
    print(f"{ear}: peak at {t[np.argmax(np.abs(h))] * 1e6:.1f} us, |H| mean {20 * np.log10(np.abs(H).mean()):.1f} dB")
ax1.set(xlim=(-0.1, 0.4), xlabel="time [ms] (0 = arrival at the origin)", ylabel="h", title="impulse response")
ax2.set(xlabel="frequency [kHz]", ylabel="|H| [dB re free field at origin]", title="transfer function")
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
    ax.legend()
fig.suptitle(f"{d.species}: az {a.az:g} deg, el {a.el:g} deg")
finish(fig, a)
