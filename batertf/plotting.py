"""Plotting helpers: Lambert azimuthal equal-area projection (LAEA), of the frontal hemisphere or the full sphere.

Centre: straight ahead (az = el = 0). A direction at angle c from the centre lands at radius 2 sin(c / 2), so the
frontal hemisphere fills a disc of radius sqrt(2) and equal solid angles get equal areas. Azimuth +90 deg (the bat's
left) is drawn on the left, elevation +90 deg at the top, as in the earlier az/el maps.
"""
import numpy as np


def laea(az_deg, el_deg):
    """Projected (x, y) of directions given in degrees (arrays broadcast)."""
    a, e = np.deg2rad(az_deg), np.deg2rad(el_deg)
    cos_c = np.cos(e) * np.cos(a)
    k = np.sqrt(2.0 / np.maximum(1.0 + cos_c, 1e-12))
    return -k * np.cos(e) * np.sin(a), k * np.sin(e)


def laea_map(ax, az, el, data_db, vmin=-30, vmax=0, cmap="inferno", levels=(-12, -6, -3), graticule=30):
    """Draw data_db [el, az] (frontal hemisphere grid) in LAEA with contours and an az/el graticule."""
    A, E = np.meshgrid(az, el)
    X, Y = laea(A, E)
    im = ax.pcolormesh(X, Y, data_db, vmin=vmin, vmax=vmax, cmap=cmap, shading="gouraud", rasterized=True)
    if levels:
        ax.contour(X, Y, data_db, levels=sorted(levels), colors="w", linewidths=0.6)
    t = np.linspace(-90, 90, 181)
    for g in range(-90 + graticule, 90, graticule):
        ax.plot(*laea(np.full_like(t, g), t), color="0.75", lw=0.4, alpha=0.6)    # meridians (constant azimuth)
        ax.plot(*laea(t, np.full_like(t, g)), color="0.75", lw=0.4, alpha=0.6)    # parallels (constant elevation)
    ax.plot(*laea(np.full_like(t, 90), t), color="0.4", lw=0.8)
    ax.plot(*laea(np.full_like(t, -90), t), color="0.4", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.axis("off")
    return im


def laea_hemi_map(ax, az, el, data_db, side="front", vmin=-30, vmax=0, cmap="inferno", levels=(-12, -6, -3),
                  graticule=30, n=241):
    """One hemisphere of full-sphere data [el, az] (azimuth -180..<180) in LAEA, rendered by inverse projection.
    side="front": centred straight ahead; side="back": centred straight behind, seen from behind the bat. In both,
    the bat's left (azimuth +90 deg) is on the left and up is up, so the left/right rims of the two discs are the same
    directions (azimuth +90 / -90)."""
    from .rotate import SphereSampler
    sgn = 1.0 if side == "front" else -1.0
    x = np.linspace(-np.sqrt(2), np.sqrt(2), n)
    X, Y = np.meshgrid(x, x)
    rho2 = X ** 2 + Y ** 2
    inside = rho2 <= 2.0
    c = 1 - np.minimum(rho2, 2.0) / 2                             # cosine of the angle from the disc centre
    k = np.sqrt(2 / (1 + c))
    u = np.stack([sgn * c, -X / k, Y / k], -1)
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    img = np.ma.masked_where(~inside, SphereSampler(np.asarray(data_db, float))(u))
    im = ax.imshow(img, origin="lower", extent=[-np.sqrt(2), np.sqrt(2)] * 2, vmin=vmin, vmax=vmax, cmap=cmap,
                   interpolation="bilinear", rasterized=True)
    if levels:
        ax.contour(X, Y, img, levels=sorted(levels), colors="w", linewidths=0.5)

    def proj(a, e):                                               # forward projection of directions on this side
        a, e = np.deg2rad(a), np.deg2rad(e)
        ux, uy, uz = np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)
        kk = np.sqrt(2 / np.maximum(1 + sgn * ux, 1e-12))
        ok = sgn * ux >= -1e-9
        return np.where(ok, -kk * uy, np.nan), np.where(ok, kk * uz, np.nan)

    t = np.linspace(-90, 90, 181)
    for g in range(-180, 180, graticule):
        ax.plot(*proj(np.full_like(t, g), t), color="0.75", lw=0.35, alpha=0.5)
    tt = np.linspace(-180, 180, 721)
    for g in range(-90 + graticule, 90, graticule):
        ax.plot(*proj(tt, np.full_like(tt, g)), color="0.75", lw=0.35, alpha=0.5)
    cc = np.linspace(0, 2 * np.pi, 361)
    ax.plot(np.sqrt(2) * np.cos(cc), np.sqrt(2) * np.sin(cc), color="0.4", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.axis("off")
    return im


def laea_full_map(ax, az, el, data_db, vmin=-30, vmax=0, cmap="inferno", levels=(-12, -6, -3), graticule=30):
    """Full sphere in LAEA centred straight ahead: the frontal hemisphere is the inner disc of radius sqrt(2) (drawn
    as a grey circle), the rear hemisphere the ring around it, straight back the outer circle (radius 2).
    data_db [el, az] on the full-sphere grid (azimuth -180..<180)."""
    # Rendered by inverse projection (pixel -> direction -> bilinear sample on the sphere), because straight back maps
    # onto the whole outer circle and grid cells next to it degenerate into long slivers.
    from .rotate import SphereSampler
    n = 301
    x = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(x, x)
    rho2 = X ** 2 + Y ** 2
    inside = rho2 <= 4.0
    ux = 1 - np.minimum(rho2, 4.0) / 2
    k = np.sqrt(2 / np.maximum(1 + ux, 1e-12))
    u = np.stack([ux, -X / k, Y / k], -1)
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    img = SphereSampler(np.asarray(data_db, float))(u)
    img = np.ma.masked_where(~inside, img)
    im = ax.imshow(img, origin="lower", extent=[-2, 2, -2, 2], vmin=vmin, vmax=vmax, cmap=cmap,
                   interpolation="bilinear", rasterized=True)
    if levels:
        ax.contour(X, Y, img, levels=sorted(levels), colors="w", linewidths=0.5)
    t = np.linspace(-90, 90, 181)
    tt = np.linspace(-179.9, 179.9, 361)
    for g in range(-180 + graticule, 180, graticule):
        ax.plot(*laea(np.full_like(t, g), t), color="0.75", lw=0.35, alpha=0.5)
    for g in range(-90 + graticule, 90, graticule):
        ax.plot(*laea(tt, np.full_like(tt, g)), color="0.75", lw=0.35, alpha=0.5)
    for edge in (90, -90):                                          # boundary of the frontal hemisphere
        ax.plot(*laea(np.full_like(t, edge), t), color="0.85", lw=0.9)
    c = np.linspace(0, 2 * np.pi, 361)
    ax.plot(2 * np.cos(c), 2 * np.sin(c), color="0.4", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-2.05, 2.05)
    ax.axis("off")
    return im
