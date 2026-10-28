"""Figures of rigs (batertf/rig.py): painter's-algorithm mesh renders coloured by dominant bone, with the skeleton."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# view matrices: rows = screen x, screen y, towards the viewer (CT frame)
_c, _s = np.cos(np.radians(40)), np.sin(np.radians(40))
VIEWS = {"front": np.array([[0, 1.0, 0], [0, 0, 1], [1, 0, 0]]),                 # bat's left on the right
         "left side": np.array([[-1.0, 0, 0], [0, 0, 1], [0, 1, 0]]),           # snout to the left
         "top": np.array([[0, -1.0, 0], [1, 0, 0], [0, 0, 1]]),                 # snout up
         "front-left 40 deg": np.array([[_s, _c, 0], [0, 0, 1], [_c, -_s, 0]]),
         "back": np.array([[0, -1.0, 0], [0, 0, 1], [-1, 0, 0]])}
HEAD = np.array([0.78, 0.78, 0.78])
BONE_COLOURS = {"pinna_left": ["#1f77b4", "#ff7f0e", "#2ca02c"], "pinna_right": ["#d62728", "#9467bd", "#8c564b"],
                "noseleaf": ["#e377c2", "#bcbd22"]}


def bone_colours(rig):
    out = []
    for b in rig.bones:
        app = b.split(".")[0]
        chain = [x for x in rig.bones if x.split(".")[0] == app]
        out.append(plt.matplotlib.colors.to_rgb(BONE_COLOURS[app][chain.index(b) % len(BONE_COLOURS[app])]))
    return np.array(out)


def weight_colours(rig):
    """Per-face colour: weight-blended bone colours over grey head."""
    Wf = np.concatenate([np.clip(1 - rig.W.sum(1, keepdims=True), 0, 1), rig.W], 1)
    cols = np.vstack([HEAD, bone_colours(rig)])
    return (Wf @ cols)[rig.rest.faces].mean(1)


def render(ax, m, R, face_rgb=None, scale=1e3, light=(0.3, 0.4, 0.87), edge_alpha=0.0):
    """Back-face-culled, depth-sorted flat shading of mesh m seen through view matrix R (rows: screen x, y, to viewer)."""
    w = m.vertices @ R.T * scale
    tri = w[m.faces]
    n = m.face_normals @ R.T
    sel = n[:, 2] > 0
    o = np.argsort(tri[sel][:, :, 2].mean(1))
    rgb = np.tile(HEAD, (len(m.faces), 1)) if face_rgb is None else np.asarray(face_rgb)[:, :3]
    sh = 0.3 + 0.7 * np.clip(n[sel][o] @ np.asarray(light) / np.linalg.norm(light), 0, 1)
    c = np.clip(rgb[sel][o] * sh[:, None], 0, 1)
    ax.add_collection(PolyCollection(tri[sel][o][:, :, :2], facecolors=c,
                                     edgecolors=(0, 0, 0, edge_alpha) if edge_alpha else "none", linewidths=0.1))
    ax.set_aspect("equal")
    return w


def draw_skeleton(ax, rig, R, pose=None, scale=1e3, colour="k"):
    jp = rig.joint_positions(pose or {})
    for b in rig.bones:
        child = [k for k, j in rig.joints.items() if j["parent"] == b][0]
        p, q = jp[b] @ R.T * scale, jp[child] @ R.T * scale
        ax.plot([p[0], q[0]], [p[1], q[1]], "-", color=colour, lw=1.8, zorder=10)
        ax.plot(p[0], p[1], "o", color=colour, ms=4, mfc="w" if b.endswith(".base") else colour, zorder=11)


def frame_axes(ax, ms, R, pad=1.0, scale=1e3):
    pts = np.vstack([m.vertices for m in ms]) @ R.T * scale
    lo, hi = pts[:, :2].min(0) - pad, pts[:, :2].max(0) + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
