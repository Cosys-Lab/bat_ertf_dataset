"""Directivity maps (azimuth/elevation) of the left ear, the emitter and the two-way (emitter x left ear) at one
frequency, in dB re the maximum. Usage: python 01_plot_directivity.py [species_folder] [--freq 60000] [--out f.png]"""
import matplotlib.pyplot as plt
import numpy as np

from _setup import args, bd, finish

a = args(__doc__, freq=60e3)
d = bd.load(a.species)
k = np.argmin(np.abs(d.freqs - a.freq))              # nearest simulated frequency (or d.interp(..., freqs=[f]))
maps = {"left ear": d.H["left"][..., k], "emitter": d.H["emitter"][..., k]}
maps["two-way (emitter x left)"] = maps["left ear"] * maps["emitter"]

fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (name, H) in zip(axs, maps.items()):
    L = 20 * np.log10(np.abs(H) / np.abs(H).max())
    im = ax.pcolormesh(d.az, d.el, L, vmin=-30, vmax=0, cmap="viridis", shading="nearest")
    ax.contour(d.az, d.el, L, [-6, -3], colors="w", linewidths=0.6)
    ax.set(title=name, xlabel="azimuth [deg] (+ = left)", xlim=(180, -180), aspect="equal")
axs[0].set_ylabel("elevation [deg]")
fig.colorbar(im, ax=axs, label="dB re max", shrink=0.8)
fig.suptitle(f"{d.species}, {d.freqs[k] / 1e3:g} kHz (contours -3, -6 dB)")
finish(fig, a)
