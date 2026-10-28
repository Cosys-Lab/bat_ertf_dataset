"""Dense GPU boundary element solver for the exterior Helmholtz problem (rigid body with vibrating patches).

Same physics and conventions as NumCalc (numcalc.py): exp(-i w t), G = exp(ikr)/(4 pi r), constant triangles with
collocation at the centroids, Burton-Miller coupling. n points out of the body into the fluid, q = dp/dn = i w rho v_n.
    CBIE  (1/2 I - K) p = -S q            HBIE  W p = (1/2 I + K') q
    BM    (1/2 I - K - alpha W) p = -S q - alpha (1/2 I + K') q,   alpha = ALPHA_SIGN * i / k
    field p(x) = sum_j [p_j dG/dn_y(x, y_j) - G(x, y_j) q_j] A_j
Signs checked against the rigid sphere series solution (tests/gpubem_sphere.py); details in docs/gpubem.md.
The BM matrix is assembled densely (complex64, N^2 * 8 bytes: 17 GB for 46k faces) and solved with batched
restarted GMRES, all sources of one frequency at once.
"""
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from . import grid, numcalc

ALPHA_SIGN = 1.0     # alpha = +i/k (verified at the first interior Dirichlet eigenfrequency of the sphere, ka = pi)

# triangle rules in barycentric coordinates (weights sum to 1)
_R3 = (np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]]), np.full(3, 1/3))
_a, _b, _c, _d = 0.059715871789770, 0.470142064105115, 0.797426985353087, 0.101286507323456
_R7 = (np.array([[1/3, 1/3, 1/3], [_a, _b, _b], [_b, _a, _b], [_b, _b, _a], [_c, _d, _d], [_d, _c, _d], [_d, _d, _c]]),
       np.array([0.225] + [0.132394152788506] * 3 + [0.125939180544827] * 3))


def subdivided_rule(s, rule=_R3):
    """Rule `rule` on each of the s^2 congruent sub-triangles of the reference triangle, as one barycentric rule."""
    pts, wts = [], []
    for i in range(s):
        for j in range(s - i):
            tris = [[(i, j), (i + 1, j), (i, j + 1)]]
            if i + j < s - 1:
                tris.append([(i + 1, j), (i + 1, j + 1), (i, j + 1)])
            for t in tris:
                v = np.array([[1 - (a + b) / s, a / s, b / s] for a, b in t])     # sub-triangle corners (bary)
                pts.append(rule[0] @ v)
                wts.append(rule[1] / s**2)
    return np.concatenate(pts), np.concatenate(wts)


class Mesh:
    """Geometry of a triangle mesh on the device (float32 for the kernels, float64 kept for the self terms)."""

    def __init__(self, m, device="cuda"):
        self.tri = np.asarray(m.vertices, float)[np.asarray(m.faces)]          # (N, 3, 3)
        self.c = self.tri.mean(1)
        cr = np.cross(self.tri[:, 1] - self.tri[:, 0], self.tri[:, 2] - self.tri[:, 0])
        self.area = 0.5 * np.linalg.norm(cr, axis=1)
        self.n = cr / (2 * self.area[:, None])
        self.size = np.sqrt(4 * self.area / np.sqrt(3))                       # equilateral-equivalent edge
        self.N, self.dev = len(self.tri), device
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        self.tc, self.tn, self.ttri, self.tarea = t(self.c), t(self.n), t(self.tri), t(self.area)

    def quad(self, rule, idx=None):
        """Quadrature points (n, Q, 3) and weights (n, Q) incl. the area, on elements idx."""
        tri, area = (self.ttri, self.tarea) if idx is None else (self.ttri[idx], self.tarea[idx])
        b = torch.as_tensor(rule[0], dtype=torch.float32, device=self.dev)
        w = torch.as_tensor(rule[1], dtype=torch.float32, device=self.dev)
        return torch.einsum("qa,nai->nqi", b, tri), area[:, None] * w


def _kernel(x, nx, y, ny, w, k, alpha, kind):
    """sum_q w * kernel over the last axis. x, nx: (..., 1, 3); y: (..., Q, 3); ny broadcastable to y.
    kind "A": -dG/dn_y - alpha d2G/dn_x dn_y (BM matrix);  "B": -G - alpha dG/dn_x (BM right-hand side);
    "F": (dG/dn_y, G) for the field."""
    R = x - y
    r = torch.linalg.vector_norm(R, dim=-1)
    ir = 1 / r
    E = torch.polar(ir / (4 * np.pi), k * r)                    # G
    a = torch.complex(-ir, torch.full_like(ir, k))              # ik - 1/r ; dG/dr = G a
    Rny = (R * ny).sum(-1) * ir                                 # (R . n_y) / r
    if kind == "F":
        return ((E * a * (-Rny)) * w).sum(-1), (E * w).sum(-1)
    Rnx = (R * nx).sum(-1) * ir
    if kind == "B":
        return (-E * (1 + alpha * a * Rnx) * w).sum(-1)
    nxny = (nx * ny).sum(-1)
    W = -E * ((a * a + ir * ir) * Rnx * Rny + a * ir * (nxny - Rnx * Rny))
    return ((E * a * Rny - alpha * W) * w).sum(-1)


def self_terms(mesh, k, n_gauss=16):
    """Analytic/1-D self terms of flat constant triangles at the centroid (float64):
    S_ii = 1/(4 pi) int (exp(ikR) - 1)/(ik) dtheta,  W_ii = ik/2 - 1/(4 pi) int exp(ikR)/R dtheta,  K_ii = K'_ii = 0,
    R(theta) = centroid-to-boundary distance, integrated per edge with R = d_e / cos(phi)."""
    g, gw = np.polynomial.legendre.leggauss(n_gauss)
    S = np.zeros(mesh.N, complex)
    W = np.full(mesh.N, 0.5j * k)
    for e in range(3):
        p1, p2 = mesh.tri[:, e] - mesh.c, mesh.tri[:, (e + 1) % 3] - mesh.c
        t = p2 - p1
        L = np.linalg.norm(t, axis=1)
        t /= L[:, None]
        s1, s2 = (p1 * t).sum(1), (p2 * t).sum(1)                       # foot of the perpendicular at s = 0
        d = np.linalg.norm(p1 - s1[:, None] * t, axis=1)
        f1, f2 = np.arctan2(s1, d), np.arctan2(s2, d)
        phi = 0.5 * (f2 - f1)[:, None] * g + 0.5 * (f1 + f2)[:, None]
        wq = 0.5 * (f2 - f1)[:, None] * gw
        R = d[:, None] / np.cos(phi)
        S += (wq * (np.exp(1j * k * R) - 1) / (1j * k)).sum(1) / (4 * np.pi)
        W -= (wq * np.exp(1j * k * R) / R).sum(1) / (4 * np.pi)
    return S, W


class Assembler:
    """Near-pair lists and quadrature rules of one mesh (frequency independent)."""

    def __init__(self, mesh, near=4.0, close=1.5, s_near=6, s_close=16, far_rule=_R3, pair_batch=20000):
        self.mesh, self.far_rule, self.pair_batch = mesh, far_rule, pair_batch
        tree = cKDTree(mesh.c)
        pairs = tree.query_pairs(near * mesh.size.max(), output_type="ndarray")
        pairs = np.concatenate([pairs, pairs[:, ::-1]])
        d = np.linalg.norm(mesh.c[pairs[:, 0]] - mesh.c[pairs[:, 1]], axis=1)
        ratio = d / mesh.size[pairs[:, 1]]                              # distance in source-element sizes
        self.tiers = [(torch.as_tensor(pairs[sel], device=mesh.dev), subdivided_rule(s))
                      for sel, s in [(ratio < close, s_close), ((ratio >= close) & (ratio < near), s_near)]]
        self.n_pairs = [len(t[0]) for t in self.tiers]

    def _near(self, k, alpha, kind, pairs, rule, out, col_map=None):
        m = self.mesh
        for s in range(0, len(pairs), self.pair_batch):
            i, j = pairs[s:s + self.pair_batch].T
            y, w = m.quad(rule, j)
            v = _kernel(m.tc[i, None], m.tn[i, None], y, m.tn[j, None], w, k, alpha, kind)
            out[i, j if col_map is None else col_map[j]] = v.to(out.dtype)

    def matrix(self, k, alpha, out=None, block=256):
        """BM system matrix (N, N) complex64 on the device, rows Jacobi-scaled; returns (A, row scale)."""
        m = self.mesh
        A = out if out is not None else torch.empty(m.N, m.N, dtype=torch.complex64, device=m.dev)
        y, w = m.quad(self.far_rule)
        ny = m.tn[:, None]
        for r0 in range(0, m.N, block):
            r1 = min(r0 + block, m.N)
            A[r0:r1] = _kernel(m.tc[r0:r1, None, None], m.tn[r0:r1, None, None], y, ny, w, k, alpha, "A")
        for pairs, rule in self.tiers:
            self._near(k, alpha, "A", pairs, rule, A)
        _, Wii = self_terms(m, k)
        diag = torch.as_tensor(0.5 - alpha * Wii, dtype=torch.complex64, device=m.dev)
        A.diagonal().copy_(diag)
        scale = 1 / diag
        A *= scale[:, None]
        return A, scale

    def rhs_matrix(self, k, alpha, cols):
        """Operator (-S - alpha K') restricted to columns `cols` (N, len(cols)); the -alpha/2 q term is added later."""
        m = self.mesh
        cols_t = torch.as_tensor(cols, device=m.dev)
        y, w = m.quad(self.far_rule, cols_t)
        B = _kernel(m.tc[:, None, None], m.tn[:, None, None], y[None], m.tn[cols_t][None, :, None], w[None],
                    k, alpha, "B").to(torch.complex64)
        col_map = torch.full((m.N,), -1, dtype=torch.long, device=m.dev)
        col_map[cols_t] = torch.arange(len(cols), device=m.dev)
        for pairs, rule in self.tiers:
            p = pairs[col_map[pairs[:, 1]] >= 0]
            self._near(k, alpha, "B", p, rule, B, col_map)
        Sii, _ = self_terms(m, k)
        B[cols_t, torch.arange(len(cols), device=m.dev)] = torch.as_tensor(-Sii[cols], dtype=torch.complex64,
                                                                           device=m.dev)
        return B


def gmres(A, B, tol=1e-5, restart=60, maxiter=600):
    """Batched restarted GMRES (one independent Krylov space per column of B). Returns X, iterations, rel. residual."""
    X = torch.zeros_like(B)
    bn = torch.linalg.vector_norm(B, dim=0)
    it = 0
    while True:
        Rr = B - A @ X
        beta = torch.linalg.vector_norm(Rr, dim=0)
        if (beta / bn).max() < tol or it >= maxiter:
            return X, it, (beta / bn).max().item()
        m = B.shape[1]
        V = torch.zeros(restart + 1, *B.shape, dtype=B.dtype, device=B.device)
        H = torch.zeros(restart + 1, restart, m, dtype=torch.complex128, device=B.device)
        cs = torch.zeros(restart, m, dtype=torch.complex128, device=B.device)
        sn = torch.zeros(restart, m, dtype=torch.complex128, device=B.device)
        g = torch.zeros(restart + 1, m, dtype=torch.complex128, device=B.device)
        g[0] = beta
        V[0] = Rr / beta
        for j in range(restart):
            wv = A @ V[j]
            for _ in range(2):                                          # CGS2
                h = torch.einsum("inm,nm->im", V[:j + 1].conj(), wv)
                wv = wv - torch.einsum("inm,im->nm", V[:j + 1], h)
                H[:j + 1, j] += h.to(torch.complex128)
            hn = torch.linalg.vector_norm(wv, dim=0)
            H[j + 1, j] = hn.to(torch.complex128)
            V[j + 1] = wv / hn.clamp_min(1e-30)
            for i in range(j):                                          # previous Givens rotations
                t = cs[i].conj() * H[i, j] + sn[i].conj() * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = t
            nu = torch.sqrt(H[j, j].abs() ** 2 + H[j + 1, j].abs() ** 2).clamp_min(1e-300)
            cs[j], sn[j] = H[j, j] / nu, H[j + 1, j] / nu
            H[j, j], H[j + 1, j] = nu, 0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j].conj() * g[j]
            it += 1
            if (g[j + 1].abs() / bn).max() < tol or it >= maxiter:
                j += 1
                break
        else:
            j = restart
        Hj = H[:j, :j].permute(2, 0, 1)                                 # (m, j, j) upper triangular
        yv = torch.linalg.solve_triangular(Hj, g[:j].T[..., None], upper=True)[..., 0]   # (m, j)
        X = X + torch.einsum("jnm,mj->nm", V[:j], yv.to(B.dtype))


def solve(m, patches, freqs, points, device="cuda", alpha_sign=ALPHA_SIGN, tol=1e-5, log=print, asm_kw=None,
          stats=None, surface=None, direct=False):
    """Transfer functions of every source, referenced and conjugated exactly like numcalc.read_source.
    m: trimesh (closed, outward); patches: {source: [face index arrays]} (several arrays vibrate in phase);
    points: (P, 3) evaluation points [m]. Returns {source: complex array (P, n_freqs)}.
    alpha_sign = 0 gives the plain CBIE (for the irregular-frequency test). If `surface` is a dict, the surface
    pressure (N, n_sources) of each frequency is stored in it (key: frequency). direct=True: LU instead of GMRES."""
    mesh = Mesh(m, device)
    t0 = time.time()
    asm = Assembler(mesh, **(asm_kw or {}))
    names = list(patches)
    faces = {n: np.concatenate([np.asarray(f) for f in patches[n]]) for n in names}
    cols = np.unique(np.concatenate(list(faces.values())))
    vel = np.zeros((mesh.N, len(names)))
    for s, n in enumerate(names):
        vel[faces[n], s] = numcalc.VELOCITY
    vel_c = torch.as_tensor(vel[cols], dtype=torch.complex64, device=device)
    pts = torch.as_tensor(np.asarray(points), dtype=torch.float32, device=device)
    yq, wq = mesh.quad(_R3)
    out = {n: np.zeros((len(points), len(freqs)), complex) for n in names}
    A_buf = torch.empty(mesh.N, mesh.N, dtype=torch.complex64, device=device)
    if stats is not None:
        stats.update(n_faces=mesh.N, near_pairs=asm.n_pairs, setup_s=time.time() - t0, per_freq=[])
    for kf, f in enumerate(freqs):
        w = 2 * np.pi * f
        k = w / numcalc.C
        alpha = alpha_sign * 1j / k
        torch.cuda.synchronize(); ta = time.time()
        A, scale = asm.matrix(k, alpha, out=A_buf)
        Bop = asm.rhs_matrix(k, alpha, cols)
        qc = 1j * w * numcalc.RHO * vel_c                               # q on the patch columns
        b = Bop @ qc
        b[torch.as_tensor(cols, device=device)] -= alpha / 2 * qc
        b *= scale[:, None]
        torch.cuda.synchronize(); tb = time.time()
        if direct:
            p = torch.linalg.solve(A, b)
            n_it, res = 0, (torch.linalg.vector_norm(A @ p - b) / torch.linalg.vector_norm(b)).item()
        else:
            p, n_it, res = gmres(A, b, tol=tol)
        torch.cuda.synchronize(); tc = time.time()
        if surface is not None:
            surface[f] = p.cpu().numpy()
        q = torch.zeros(mesh.N, len(names), dtype=torch.complex64, device=device)
        q[torch.as_tensor(cols, device=device)] = qc
        for s0 in range(0, len(pts), 64):
            dGdn, G = _kernel(pts[s0:s0 + 64, None, None], None, yq[None], mesh.tn[None, :, None], wq[None], k, 0, "F")
            pf = dGdn.to(torch.complex64) @ p - G.to(torch.complex64) @ q
            for s, n in enumerate(names):
                r = np.linalg.norm(points[s0:s0 + 64], axis=1)
                Q = numcalc.VELOCITY * mesh.area[faces[n]].sum()
                p_ff = -1j * w * numcalc.RHO * Q * np.exp(1j * k * r) / (4 * np.pi * r)
                out[n][s0:s0 + 64, kf] = np.conj(pf[:, s].cpu().numpy() / p_ff)
        torch.cuda.synchronize(); td = time.time()
        msg = dict(f=f, assembly_s=tb - ta, solve_s=tc - tb, field_s=td - tc, iters=n_it, residual=res,
                   peak_gb=torch.cuda.max_memory_allocated() / 1e9)
        if stats is not None:
            stats["per_freq"].append(msg)
        log(f"gpubem {f/1e3:.1f} kHz: N={mesh.N}, assembly {tb-ta:.1f} s, GMRES {tc-tb:.1f} s ({n_it} it, "
            f"res {res:.1e}, {len(names)} rhs), field {td-tc:.1f} s, peak {msg['peak_gb']:.1f} GB")
    del A_buf
    torch.cuda.empty_cache()
    return out


def solve_lu(m, patches, freqs, points, device="cuda", alpha_sign=ALPHA_SIGN, log=print, asm_kw=None, chunk=64):
    """Same result as solve(), with a direct solver for many sources: per frequency the BM matrix is LU-factored once,
    in place (no second N x N buffer), and the sources are solved in chunks of `chunk` right-hand sides (their
    right-hand-side operator is assembled per chunk). No iteration limit, so no convergence issue on fine CT meshes;
    cost per frequency is about one factorisation plus a few seconds per chunk."""
    mesh = Mesh(m, device)
    asm = Assembler(mesh, **(asm_kw or {}))
    names = list(patches)
    faces = {n: np.concatenate([np.asarray(f) for f in patches[n]]) for n in names}
    pts = torch.as_tensor(np.asarray(points), dtype=torch.float32, device=device)
    r = np.linalg.norm(points, axis=1)
    yq, wq = mesh.quad(_R3)
    out = {n: np.zeros((len(points), len(freqs)), complex) for n in names}
    A_buf = torch.empty(mesh.N, mesh.N, dtype=torch.complex64, device=device)
    piv = torch.empty(mesh.N, dtype=torch.int32, device=device)
    for kf, f in enumerate(freqs):
        w = 2 * np.pi * f
        k = w / numcalc.C
        alpha = alpha_sign * 1j / k
        torch.cuda.synchronize(); ta = time.time()
        A, scale = asm.matrix(k, alpha, out=A_buf)
        torch.cuda.synchronize(); tb = time.time()
        At = A.mT                                       # column-major view: cuSOLVER factors it without a copy
        torch.linalg.lu_factor(At, out=(At, piv))       # in place: A's buffer now holds the LU factors of A^T
        torch.cuda.synchronize(); tc = time.time()
        for c0 in range(0, len(names), chunk):
            nm = names[c0:c0 + chunk]
            cols = np.unique(np.concatenate([faces[n] for n in nm]))
            cols_t = torch.as_tensor(cols, device=device)
            vel = np.zeros((len(cols), len(nm)))
            for s, n in enumerate(nm):
                vel[np.searchsorted(cols, faces[n]), s] = numcalc.VELOCITY
            qc = 1j * w * numcalc.RHO * torch.as_tensor(vel, dtype=torch.complex64, device=device)
            b = asm.rhs_matrix(k, alpha, cols) @ qc
            b[cols_t] -= alpha / 2 * qc
            b *= scale[:, None]
            p = torch.linalg.lu_solve(At, piv, b.conj(), adjoint=True).conj()   # A x = b from the factors of A^T
            q = torch.zeros(mesh.N, len(nm), dtype=torch.complex64, device=device)
            q[cols_t] = qc
            for s0 in range(0, len(pts), 64):
                dGdn, G = _kernel(pts[s0:s0 + 64, None, None], None, yq[None], mesh.tn[None, :, None], wq[None], k, 0, "F")
                pf = (dGdn.to(torch.complex64) @ p - G.to(torch.complex64) @ q).cpu().numpy()
                rr = r[s0:s0 + 64]
                for s, n in enumerate(nm):
                    Q = numcalc.VELOCITY * mesh.area[faces[n]].sum()
                    p_ff = -1j * w * numcalc.RHO * Q * np.exp(1j * k * rr) / (4 * np.pi * rr)
                    out[n][s0:s0 + 64, kf] = np.conj(pf[:, s] / p_ff)
        torch.cuda.synchronize(); td = time.time()
        log(f"gpubem-lu {f/1e3:.1f} kHz: N={mesh.N}, assembly {tb-ta:.1f} s, LU {tc-tb:.1f} s, "
            f"{len(names)} sources {td-tc:.1f} s, peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    del A_buf
    torch.cuda.empty_cache()
    return out


def to_grid(H, grid_name="hemisphere"):
    """(P, F) on grid.unique_points(grid_name) -> (el, az, F) like numcalc.read_source."""
    return H[grid.unique_points(grid_name)[1]]
