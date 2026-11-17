"""Claim 1's variational program for E(rho) (main.tex section 3.1, Theorem 1 of docs/green_list_watermarking_proof.tex)
and its log-barrier Newton solver, moved unchanged from claim1_program.py, whose docstring describes the method."""
import time

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.special import rel_entr, xlogy

ID_TOL = 1e-12  # a start must satisfy the equalities to this
GAP, DEC_TOL, T_FACTOR, MAX_NEWTON = 1e-9, 1e-10, 10.0, 200  # barrier: gap target, centring tolerance


def objective(x, keep, logPGY, PGgY):
    """f(Q) = sum Q ln[Q Q_Y / (P_GY Q_YZ)], 0 ln 0 = 0, and its gradient ln Q - ln Q_YZ + ln Q_Y - ln P_GY + 1."""
    Q = np.zeros(keep.shape)
    Q[keep] = np.maximum(x, 0)
    QYZ = Q.sum(0)
    QY = QYZ.sum(1)
    f = xlogy(Q, Q).sum() - xlogy(QYZ, QYZ).sum() + xlogy(QY, QY).sum() - (Q.sum(2) * logPGY).sum()
    # ln Q - ln Q_YZ = ln Q_{G|y,z}; an empty (y, z) column takes P_{G|y}, its cheapest direction (a subgradient)
    cond = np.where(QYZ > 0, Q / np.where(QYZ > 0, QYZ, 1), PGgY[:, :, None])
    g = np.log(np.maximum(cond, 1e-300)) + np.log(np.maximum(QY, 1e-300))[:, None] - logPGY[:, :, None] + 1
    return float(f), g[keep]


def forms(Q, PGY):
    """Claim 1's objective three ways: D(Q_GYZ || P_GY Q_Z|Y), D(Q_GY || P_GY) + I_Q(G; Z | Y),
    D(Q_Y || P_Y) + sum_yz Q_YZ D(Q_G|yz || P_G|y)."""
    QGY, QYZ = Q.sum(2), Q.sum(0)
    QY, PY = QYZ.sum(1), PGY.sum(0)
    QZgY = QYZ / QY[:, None]
    return (rel_entr(Q, PGY[:, :, None] * QZgY).sum(),
            rel_entr(QGY, PGY).sum() + rel_entr(Q, QGY[:, :, None] * QZgY).sum(),
            rel_entr(QY, PY).sum() + rel_entr(Q, (PGY / PY)[:, :, None] * QYZ).sum())


def program(L, rho):
    """Variables Q(G, y, z) where keep; equalities Aeq x = beq, edit cost x <= rho; viol(x) = largest violation."""
    mu, m = L["mu"], L["m"]
    Gi, yi, zi = np.indices((len(mu), m, m))
    keep = yi == zi if rho <= 0 else np.ones(Gi.shape, bool)
    zz, GG = np.divmod(np.arange(m * (len(mu) - 1)), len(mu) - 1)  # every z, every list but the last
    prod = (zi.ravel() == zz[:, None]) * ((Gi.ravel() == GG[:, None]) - mu[GG][:, None])
    Aeq, beq = np.vstack([np.ones(Gi.size), prod])[:, keep.ravel()], np.r_[1.0, np.zeros(len(zz))]
    cost = (yi != zi)[keep].astype(float)
    viol = lambda x: float(max(np.abs(Aeq @ x - beq).max(), cost @ x - rho, -x.min(), 0.0))
    return keep, Aeq, beq, cost, viol


def barrier(fun, hess, x, Aeq, beq, cost, rho):
    """Log-barrier Newton method from a strictly feasible x. Centres f + barrier/t under Aeq x = beq by Newton steps on
    the KKT system, t = 1, 10, 100, ... until (number of inequalities)/t <= GAP. Returns x, the gap bound, the last
    Newton decrement, whether the last centring converged, the Newton steps, and how many earlier centrings did not."""
    ineq, n, p = rho > 0, x.size, len(beq)

    def psi(x, t):  # the centring objective in the units of f
        if x.min() <= 0 or (ineq and cost @ x >= rho):
            return np.inf
        return fun(x)[0] - (np.log(x).sum() + (np.log(rho - cost @ x) if ineq else 0.0)) / t

    t, steps, missed = 1.0, 0, 0
    while True:
        conv = False
        for _ in range(MAX_NEWTON):
            s = rho - cost @ x if ineq else 1.0
            g = t * fun(x)[1] - 1 / x + (cost / s if ineq else 0.0)
            H = t * hess(x) + np.diag(1 / x ** 2) + (np.outer(cost, cost) / s ** 2 if ineq else 0.0)
            K = np.block([[H, Aeq.T], [Aeq, np.zeros((p, p))]])
            rhs = np.r_[-g, beq - Aeq @ x]
            lu = lu_factor(K)
            sol = lu_solve(lu, rhs)
            sol += lu_solve(lu, rhs - K @ sol)  # one refinement step: without it Aeq x drifted to 1e-8 at large t
            dx = sol[:n]
            dec2 = max(float(dx @ H @ dx), 0.0) / t  # squared Newton decrement of f + barrier/t
            steps += 1
            if np.sqrt(dec2) <= DEC_TOL:
                conv = True
                break
            a, p0 = 1.0, psi(x, t)
            # backtracking; the slack of 1e-13 |psi| absorbs psi's rounding so the last Newton steps go through
            while psi(x + a * dx, t) > p0 - 0.25 * a * dec2 + 1e-13 * max(1.0, abs(p0)):
                a /= 2
                if a < 1e-14:
                    break
            if a < 1e-14:
                break
            x = x + a * dx
        if (n + ineq) / t <= GAP:
            return x, (n + ineq) / t, float(np.sqrt(dec2)), conv, steps, missed
        missed += not conv
        t *= T_FACTOR


def solve(L, rho, starts):
    """One barrier run per start."""
    keep, Aeq, beq, cost, viol = program(L, rho)
    logPGY = np.log(L["PGY"])
    fun = lambda x: objective(x, keep, logPGY, L["PGY"] / L["PY"])
    m = L["m"]
    Gi, yi, zi = np.indices(keep.shape)
    yk, zk = yi[keep], zi[keep]
    SYZ, SY = (yk[:, None] == yk) & (zk[:, None] == zk), yk[:, None] == yk

    def hess(x):
        """Exact Hessian of f: diag(1/Q) - 1/Q_YZ on entries sharing (y, z) + 1/Q_Y on entries sharing y."""
        Q = np.zeros(keep.shape)
        Q[keep] = x
        QYZ = Q.sum(0)
        return np.diag(1 / x) - SYZ / QYZ[yk, zk][:, None] + SY / QYZ.sum(1)[yk][:, None]

    runs = []
    for s, eps in starts:
        W = (1 - eps) * np.eye(m) + eps / (m - 1) * (1 - np.eye(m))
        x0 = (L["mu"][Gi] * s[yi] * W[yi, zi])[keep]
        assert viol(x0) <= ID_TOL and x0.min() > 0 and (rho == 0 or cost @ x0 < rho), "start not strictly feasible"
        t = time.perf_counter()
        x, gap, dec, conv, steps, missed = barrier(fun, hess, x0, Aeq, beq, cost, rho)
        runs.append({"value": fun(x)[0], "violation": viol(x), "gap_bound": gap, "last_decrement": dec,
                     "last_centring_converged": conv, "newton_steps": steps, "earlier_centrings_not_converged": missed,
                     "seconds": time.perf_counter() - t})
    return runs
