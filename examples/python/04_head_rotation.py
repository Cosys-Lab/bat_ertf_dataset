"""Head orientation: the two-way directivity (emitter x left ear) before and after turning the head 30 deg to the left
and 15 deg nose down, and the interaural level difference along the horizon for both orientations.
Usage: python 04_head_rotation.py [species_folder] [--freq 60000] [--out f.png]"""
import matplotlib.pyplot as plt
import numpy as np

from _setup import args, bd, finish

a = args(__doc__, freq=60e3)
d = bd.load(a.species)
R = bd.rotation_matrix(yaw=30, pitch=15)              # columns: head axes in the new (world) frame
dr = d.rotate(R)
az, el = np.arange(-180, 180, 2.0), np.arange(-90, 90.1, 2.0)
U = bd.unit_vector(*np.meshgrid(az, el))              # world directions [el, az, 3]
f = [a.freq]

fig, axs = plt.subplots(1, 3, figsize=(16, 4))
for ax, (name, x) in zip(axs, (("head frame", d), ("yaw 30, pitch 15 (nose down)", dr))):
    tw = x.two_way(U, f)["left"][..., 0]
    L = 20 * np.log10(np.abs(tw) / np.abs(tw).max())
    ax.pcolormesh(az, el, L, vmin=-30, vmax=0, shading="nearest")
    ax.contour(az, el, L, [-6], colors="w", linewidths=0.6)
    ax.set(title=f"two-way, left ear: {name}", xlabel="azimuth [deg]", ylabel="elevation [deg]", xlim=(180, -180),
           aspect="equal")
    h = bd.unit_vector(az, 0)
    ild = 20 * np.log10(np.abs(x.interp("left", h, f)[:, 0]) / np.abs(x.interp("right", h, f)[:, 0]))
    axs[2].plot(az, ild, label=name)
axs[2].set(xlabel="azimuth [deg] (world, el = 0)", ylabel="ILD left - right [dB]", title="interaural level difference",
           xlim=(180, -180))
axs[2].legend()
axs[2].grid(alpha=0.3)
print("landmarks after rotation [mm]:", {k: np.round(v * 1e3, 1).tolist() for k, v in dr.landmarks.items()})
fig.suptitle(f"{d.species}, {a.freq / 1e3:g} kHz")
finish(fig, a)
