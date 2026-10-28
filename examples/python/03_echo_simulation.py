"""Biosonar echo simulation: an FM call, a few point reflectors, two-way transfer (emitter x ear), spherical
spreading, delay 2r/c and atmospheric absorption (ISO 9613-1). Plots the echo signals and spectrograms of both ears,
and a simple 'cochleogram' (log-spaced band-pass filterbank, half-wave rectification, 1 kHz low-pass).
Usage: python 03_echo_simulation.py [species_folder] [--out f.png]"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from _setup import args, bd, finish

a = args(__doc__)
d = bd.load(a.species)
fs = 500e3
tc = np.arange(int(2e-3 * fs)) / fs                    # 2 ms linear FM sweep 90 -> 35 kHz, Hann envelope
call = signal.chirp(tc, 90e3, tc[-1], 35e3) * signal.windows.hann(len(tc))

# reflectors: x, y, z [m] in the head frame (X forward, Y left, Z up), echo amplitude re 1 m (TS = 20 log10 a)
refl = np.array([[0.60, 0.15, 0.00, 0.05],            # ahead, slightly left
                 [0.90, -0.30, 0.10, 0.10],           # ahead right, a bit up
                 [1.40, 0.00, -0.20, 0.30]])          # further, below
t, left, right = d.simulate_echoes(refl, call, fs, temperature_c=25, humidity_pct=60)
for x, y, z, amp in refl:
    r = np.sqrt(x * x + y * y + z * z)
    print(f"reflector r = {r:.2f} m, az {np.degrees(np.arctan2(y, x)):+.0f} deg: delay {2 * r / bd.C * 1e3:.2f} ms")

fig, axs = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
centres = np.geomspace(25e3, 120e3, 48)
for j, (ear, x) in enumerate((("left", left), ("right", right))):
    axs[0, j].plot(t * 1e3, x, lw=0.5)
    axs[0, j].set(title=f"{ear} ear", ylabel="pressure re call at 1 m")
    f, ts, S = signal.spectrogram(x, fs, nperseg=256, noverlap=224)
    axs[1, j].pcolormesh(ts * 1e3, f / 1e3, 10 * np.log10(S + 1e-30), vmin=10 * np.log10(S.max()) - 60, shading="auto")
    axs[1, j].set(ylim=(20, 120), ylabel="frequency [kHz]")
    env = []
    for fc in centres:                                 # crude cochleogram
        sos = signal.butter(2, [fc / 1.08, fc * 1.08], "bandpass", fs=fs, output="sos")
        e = signal.sosfiltfilt(signal.butter(2, 1e3, fs=fs, output="sos"), np.maximum(signal.sosfilt(sos, x), 0))
        env.append(e)
    env = 20 * np.log10(np.maximum(np.array(env), 1e-12))
    axs[2, j].pcolormesh(t * 1e3, centres / 1e3, env, vmin=env.max() - 50, shading="auto")
    axs[2, j].set(yscale="log", xlabel="time [ms] (0 = call onset at the origin)", ylabel="channel CF [kHz]")
axs[0, 0].set_xlim(0, t[-1] * 1e3)
fig.suptitle(f"{d.species}: echoes of 3 point reflectors")
finish(fig, a)
