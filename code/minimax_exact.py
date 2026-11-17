#!/usr/bin/env python
"""Exact finite-n minimax miss rate beta*_n(alpha, rho) on two 2-token instances, against Theorem 1's exponent
(docs/green_list_watermarking_proof.tex). Plan: plans/2026-11-12-minimax-exact.md.

Both instances: m = 2, gamma = 1/2, e^delta = 3, uniform p, every substitution allowed (an edit flips the token).
  A: lists of one token. A position is green or red: (1/2, 1/2) under the null, (3/4, 1/4) at an unedited watermarked
     position, (1/4, 3/4) at an edited one. Statistic S, the green count.
  D: Bernoulli lists. A position is inactive (list empty or full), singleton-green or singleton-red: (1/2, 1/4, 1/4)
     under the null, (1/2, 3/8, 1/8) unedited, (1/2, 1/8, 3/8) edited. Statistic (N1, S): singleton positions and the
     green ones among them.
Reduced game: the detector is a function of the statistic, a blind attack a mixture over j <= J = floor(rho n) edited
positions. On D, P(N1 = n1, a of the j edited positions are singletons) = Bin(n - j, 1/2)(n1 - a) Bin(j, 1/2)(a), and
then S ~ Bin(n1 - a, 3/4) + Bin(a, 1/4), built one position at a time by two-point direct convolutions in linear space.
K9a D at n = 3, 4 and A at n = 3..6, alpha = 0.1, J = 0, 1, 2: the LP over every detector phi(k^n, z^n) and every edit
    map within J edits has the same value as the reduced game's LP, to 1e-8.
K9b both instances, every n and rho, level alpha: L = miss of the randomised Neyman-Pearson test at level exactly alpha
    against the attacker's mixture (first j = J alone), U = the largest miss of that test over j <= J. While
    U/L > 1.001 the best-reply j joins the mixture and the weights are re-optimised (they maximise L). Pass when
    U/L <= 1.001; beta*_n is then L. Every pmf sums to 1 within 1e-12 and the test's level is alpha to 1e-12.
K9c E_n = -ln beta*_n / n; -ln beta*_n = E n + c sqrt(n) + b ln n + d through the lengths gives E. A: against E_0(rho)
    (wmsim.exponent.rate); the calibration error is the largest |E - E_0| / E_0 over the verdict cells rho <= 0.3.
    D: against V_D(rho), Theorem 1's program value (wmsim.claim1); Theorem 1 is contradicted at a verdict cell where
    |E - V_D| / V_D > max(2 x calibration error, 5 %). rho = 0.4 is reported only.
"""
import argparse
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize, minimize_scalar

from wmsim.claim1 import solve
from wmsim.exponent import kl_bern, rate
from wmsim.singlepos import LABELS, law
from wmsim.util import pmf, write_csv

SLUG = "2026-11-12-minimax-exact"
QU, QE = 0.75, 0.25  # chance that a singleton list holds the token at an unedited / edited position (A: every list)
LP_ALPHA, LP_TOL, BRACKET, PMF_TOL = 0.1, 1e-8, 1e-3, 1e-12
FEAS_TOL = 1e-9  # as claim1_program.py: V is the lowest value among the starts within this violation
VERDICT_RHO, GAP_FLOOR, MAX_ROUNDS = 0.3, 0.05, 10
K9A = {"A": (3, 4, 5, 6), "D": (3, 4)}
LAWS = {"A": law(2, 0.5, np.log(3.0), np.full(2, 0.5), k=1), "D": law(2, 0.5, np.log(3.0), np.full(2, 0.5))}
TITLES = {"A": "A: 2 tokens ($m = 2$)\nevery green list has exactly one token ($|G| = 1$)",
          "D": "D: 2 tokens ($m = 2$)\nBernoulli lists: each token is green by its own coin flip (probability 1/2)"}


def cond(n, A):
    """Yield n1 = 0..n and the laws of S over n1 singleton positions of which a = 0..min(n1, A) are edited
    (row a: Bin(n1 - a, 3/4) + Bin(a, 1/4)), adding one position at a time."""
    C = np.ones((1, 1))
    yield 0, C
    for n1 in range(1, n + 1):
        new, k = np.zeros((min(n1, A) + 1, n1 + 1)), len(C)
        new[:k, 1:] += QU * C  # one more unedited singleton
        new[:k, :-1] += (1 - QU) * C
        if n1 <= A:  # row a = n1: every singleton edited
            new[n1, 1:] += QE * C[-1]
            new[n1, :-1] += (1 - QE) * C[-1]
        C = new
        yield n1, C


def wts(bh, n, j, n1, k):
    """P(N1 = n1, a of the j edited positions are singletons) for a = 0..k-1."""
    a = np.arange(k)
    ok = (a <= j) & (n1 - a <= n - j)
    w = np.zeros(k)
    w[ok] = bh[n - j][n1 - a[ok]] * bh[j][a[ok]]
    return w


def d_sweep(bh, n, J, js=(), psi=None):
    """One pass over n1 on D: the laws P_j(n1, s) for j in js; with psi, M(n1, a) = sum_s P(s | n1, a) psi(n1, s);
    R(n1, a) = the row sums of the conditional laws."""
    P = np.zeros((len(js), n + 1, n + 1))
    M, R = np.zeros((n + 1, J + 1)), np.zeros((n + 1, J + 1))
    for n1, C in cond(n, J):
        k = len(C)
        R[n1, :k] = C.sum(1)
        for i, j in enumerate(js):
            P[i, n1, :n1 + 1] = wts(bh, n, j, n1, k) @ C
        if psi is not None:
            M[n1, :k] = C @ psi[n1, :n1 + 1]
    return P, M, R


def d_by_j(bh, n, J, T):
    """sum over (n1, a) of P(N1 = n1, a | j) T(n1, a), for j = 0..J."""
    out, n1 = np.zeros(J + 1), np.arange(n + 1)[:, None]
    for j in range(J + 1):
        a = np.arange(j + 1)
        u = n1 - a
        W = np.where((u >= 0) & (u <= n - j), bh[n - j][np.clip(u, 0, n - j)], 0.0) * bh[j][a]
        out[j] = (W * T[:, :j + 1]).sum()
    return out


def game(inst, n, J, bh):
    """The null law of the statistic; laws(js), its laws under j edits; upper(psi), the miss rates and total masses
    of the test psi = 1 - phi under j = 0..J and the largest |row sum - 1| of the conditional laws."""
    if inst == "A":
        for _, C in cond(n, J):
            pass
        err = float(np.abs(C.sum(1) - 1).max())
        return bh[n], lambda js: C[list(js)], lambda psi: (C @ psi, C.sum(1), err)
    P0 = np.zeros((n + 1, n + 1))
    for n1 in range(n + 1):
        P0[n1, :n1 + 1] = bh[n][n1] * bh[n1]

    def upper(psi):
        _, M, R = d_sweep(bh, n, J, (), psi)
        return d_by_j(bh, n, J, M), d_by_j(bh, n, J, R), float(np.abs(R[R > 0] - 1).max())
    return P0, lambda js: d_sweep(bh, n, J, list(js))[0], upper


def np_test(P0, P1, alpha):
    """Randomised Neyman-Pearson test of P0 against P1 at level exactly alpha. Returns psi = 1 - phi: 0 on the cells
    of largest P1/P0, 1 - g on the boundary cell, 1 elsewhere."""
    p0, p1 = P0.ravel(), P1.ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where(p0 > 0, p1 / p0, np.where(p1 > 0, np.inf, -1.0))  # -1: outside both supports
    order = np.argsort(-lr, kind="stable")
    cum = np.cumsum(p0[order])
    b = int(np.searchsorted(cum, alpha, side="right"))  # order[:b] rejected outright
    psi = np.ones(len(p0))
    psi[order[:b]] = 0.0
    psi[order[b]] = 1 - (alpha - (cum[b - 1] if b else 0.0)) / p0[order[b]]
    return psi.reshape(P0.shape)


def best_weights(P0, Ps, alpha):
    """Mixture weights on the support that maximise the NP miss L (concave in the weights; -ln L is convex)."""
    f = lambda w: -np.log((np.tensordot(w, Ps, 1) * np_test(P0, np.tensordot(w, Ps, 1), alpha)).sum())
    k = len(Ps)
    if k == 2:
        t = minimize_scalar(lambda t: f(np.array([1 - t, t])), bounds=(0, 1), method="bounded",
                            options={"xatol": 1e-10}).x
        cands = [np.array([1 - t, t])]
    else:  # shortcut: Nelder-Mead on softmax weights, exact only to its tolerance; upgrade if a cell needs 3+ points
        soft = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum()
        cands = [soft(minimize(lambda v: f(soft(v)), np.zeros(k), method="Nelder-Mead",
                               options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 4000}).x)]
    return min(cands + list(np.eye(k)), key=f)


def bracket(P0, laws, upper, J, alpha):
    """K9b on one cell: L, U and the attacker's mixture after the rounds."""
    sup, w, Ps, worst, rounds = [J], np.array([1.0]), laws([J]), [], 0
    while True:
        P1 = np.tensordot(w, Ps, 1)
        psi = np_test(P0, P1, alpha)
        L = float((P1 * psi).sum())
        miss, mass, err = upper(psi)
        jb = int(np.argmax(miss))
        worst.append(jb)
        if miss[jb] <= (1 + BRACKET) * L or jb in sup or rounds == MAX_ROUNDS:
            break
        rounds += 1
        sup.append(jb)
        Ps = np.concatenate([Ps, laws([jb])])
        w = best_weights(P0, Ps, alpha)
    sums = [P0.sum(), *Ps.reshape(len(Ps), -1).sum(1), *mass]
    return {"L": L, "U": float(miss[jb]), "U_over_L": float(miss[jb] / L), "closed": bool(miss[jb] <= (1 + BRACKET) * L),
            "worst_j_first": worst[0], "worst_j_final": jb, "worst_j_each_round": worst, "support": sup,
            "weights": w.tolist(), "rounds": rounds, "miss_J_over_L": float(miss[J] / L),
            "mix_miss_over_L": float(w @ miss[sup] / L),
            "level_rel_err": float(abs((P0 * (1 - psi)).sum() - alpha) / alpha),
            "pmf_max_err": float(max(max(abs(x - 1) for x in sums), err))}


def full_lp(L, n, J, alpha):
    """The game over every detector phi(k^n, z^n) and every edit map within J edits, as one LP: minimise
    sum_y P(y) t_y, t_y >= 1 - sum_k P(k|y) phi(k, z) for each z within J edits of y, sum_k mu(k) phi(k, z) <= alpha."""
    nG, m = len(L["mu"]), L["m"]
    ks = np.array(list(itertools.product(range(nG), repeat=n)))
    ys = np.array(list(itertools.product(range(m), repeat=n)))
    Pky = (L["PGY"] / L["PY"])[ks[:, None, :], ys[None, :, :]].prod(-1)  # P(k^n | y^n)
    mu, nk, nz = L["mu"][ks].prod(-1), len(ks), len(ys)
    nphi, cols = nk * nz, lambda z: np.arange(nk) * nz + z
    rows, b = [], []
    for y in range(nz):
        for z in range(nz):
            if (ys[y] != ys[z]).sum() <= J:
                r = np.zeros(nphi + nz)
                r[cols(z)], r[nphi + y] = -Pky[:, y], -1.0
                rows.append(r)
                b.append(-1.0)
    for z in range(nz):
        r = np.zeros(nphi + nz)
        r[cols(z)] = mu
        rows.append(r)
        b.append(alpha)
    res = linprog(np.r_[np.zeros(nphi), L["PY"][ys].prod(-1)], A_ub=np.array(rows), b_ub=b,
                  bounds=[(0, 1)] * nphi + [(None, None)] * nz, method="highs")
    assert res.status == 0, res.message
    return float(res.fun)


def reduced_lp(P0, Ps, alpha):
    """The reduced game: minimise t, t >= 1 - sum P_j phi for each j, sum P0 phi <= alpha, phi in [0, 1]."""
    Pj, p0 = Ps.reshape(len(Ps), -1), P0.ravel()
    A = np.vstack([np.c_[-Pj, -np.ones(len(Pj))], np.r_[p0, 0.0]])
    res = linprog(np.r_[np.zeros(len(p0)), 1.0], A_ub=A, b_ub=np.r_[-np.ones(len(Pj)), alpha],
                  bounds=[(0, 1)] * len(p0) + [(None, None)], method="highs")
    assert res.status == 0, res.message
    return float(res.fun)


def plot(rhos, line, En, Efit, n_list, path):
    """Two panels (A, D): the formula against rho as a line, E_n per length and the extrapolated E as markers; under
    them one legend and a note on what the two instances share."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, gridc, axc, surf = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")  # categorical slots 1-5, fixed order
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.0), facecolor=surf)
    for ax, inst in zip(axes, ("A", "D")):
        ax.set_facecolor(surf)
        ax.plot(*line[inst], color=ink, lw=2, label="Claim 1's formula")
        for n, col, mk in zip(n_list, colors, "osD^v"):
            ax.plot(rhos, En[inst][n], ls="none", color=col, marker=mk, ms=5.5, mec=surf, mew=1.0,
                    label=rf"$-\ln\beta^*_n\,/\,n$ at $n = {n}$")
        ax.plot(rhos, Efit[inst], ls="none", marker="o", ms=11, mfc="none", mec=ink, mew=1.3,
                label="the same, extrapolated to infinite length")
        ax.set_title(TITLES[inst], color=ink, fontsize=11)
        ax.set_xlim(-0.02, line[inst][0].max() + 0.02)
        ax.grid(color=gridc, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(axc)
        ax.tick_params(colors=ink2, labelsize=10)
        ax.set_xlabel("edit fraction ρ", color=ink2, fontsize=10)
    axes[0].set_ylabel("exponent, nat/token", color=ink2, fontsize=10)
    fig.tight_layout(rect=(0, 0.24, 0.985, 1))  # room below for the legend and note, right for D's wide title
    xc = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2  # centred under the panels
    h, lab = axes[1].get_legend_handles_labels()
    order = [i for c in range(3) for i in range(c, len(h), 3)]  # legends fill by column; this order reads by row
    fig.legend([h[i] for i in order], [lab[i] for i in order], ncol=3, loc="upper center", bbox_to_anchor=(xc, 0.255),
               frameon=False, fontsize=11, labelcolor=ink, alignment="left", title_fontsize=11,
               title=r"$\beta^*_n$: the best detector's miss rate against its worst attack, computed exactly")
    fig.text(xc, 0.02, "A and D are the two test instances. In both, the model picks each token with equal probability "
             "(uniform $p$),\nand the watermark makes green tokens 3 times as likely ($e^\\delta = 3$).",
             ha="center", va="bottom", color=ink2, fontsize=11)
    fig.savefig(path, dpi=150, facecolor=surf)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, default=1e-3, help="level of the main cells (K9a uses 0.1)")
    ap.add_argument("--n-list", type=int, nargs="+", default=[100, 200, 400, 800])
    ap.add_argument("--rho-grid", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--seed", type=int, default=0, help="the Dirichlet starts of the program solver (V_D)")
    ap.add_argument("--out", default="data", help="writes <out>/results/<slug>/ and <out>/figures/<slug>/")
    args = ap.parse_args()
    t_start = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    n_list, rhos = sorted(args.n_list), np.array(sorted(args.rho_grid))
    res_dir, fig_dir = Path(args.out) / "results" / SLUG, Path(args.out) / "figures" / SLUG
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows, timing = [], {}

    def add(check, inst, statement, value, ref, tol, ok=None, note=""):
        value, ref = float(value), float(ref)
        rows.append({"check": check, "instance": inst, "statement": statement, "value": value, "reference": ref,
                     "diff": value - ref, "tol": tol, "pass": bool(abs(value - ref) <= tol if ok is None else ok),
                     "note": note})

    # K9a: the full game against the reduced one at small n
    t = time.perf_counter()
    for inst, ns in K9A.items():
        for n in ns:
            bh = [pmf(N, 0.5) for N in range(n + 1)]
            for J in (0, 1, 2):
                P0, laws, _ = game(inst, n, J, bh)
                full, red = full_lp(LAWS[inst], n, J, LP_ALPHA), reduced_lp(P0, laws(range(J + 1)), LP_ALPHA)
                add("K9a", inst, f"full-game LP = reduced-game LP at n {n}, J {J}, alpha {LP_ALPHA:g}", full, red,
                    LP_TOL)
    timing["K9a_s"] = time.perf_counter() - t

    # K9b: bracket beta*_n at every cell
    cells = []
    for inst in ("A", "D"):
        for n in n_list:
            t = time.perf_counter()
            bh = [pmf(N, 0.5) for N in range(n + 1)]
            for rho in rhos:
                tc = time.perf_counter()
                J = int(np.floor(rho * n + 1e-9))
                r = bracket(*game(inst, n, J, bh), J, args.alpha)
                cells.append({"instance": inst, "n": n, "rho": float(rho), "J": J, "beta_star": r["L"],
                              "neg_ln_beta": -np.log(r["L"]), "E_n": -np.log(r["L"]) / n, **r,
                              "seconds": time.perf_counter() - tc})
            timing[f"K9b_{inst}_n{n}_s"] = time.perf_counter() - t
    for c in cells:
        mix = "" if len(c["support"]) == 1 else f"; mixture over j {c['support']} with weights " \
                                                 f"{[round(x, 6) for x in c['weights']]}"
        add("K9b", c["instance"], f"U/L <= 1.001 at n {c['n']}, rho {c['rho']:g} (J {c['J']})", c["U_over_L"], 1,
            BRACKET, ok=c["closed"], note=f"worst j {c['worst_j_each_round']} (J {c['J']}){mix}")
    for inst in ("A", "D"):
        for n in n_list:
            cs = [c for c in cells if c["instance"] == inst and c["n"] == n]
            e = max(c["pmf_max_err"] for c in cs)
            add("K9b", inst, f"every pmf sums to 1 within 1e-12 at n {n} (largest |sum - 1| over rho)", e, 0, PMF_TOL)
            e = max(c["level_rel_err"] for c in cs)
            add("K9b", inst, f"NP level = alpha at n {n} (largest relative error over rho)", e, 0, PMF_TOL)

    # K9c: extrapolation against the references
    t = time.perf_counter()
    line_rho = np.unique(np.r_[np.round(np.arange(0, 0.5001, 0.01), 2), rhos])
    V = {}
    for rho in line_rho:  # Theorem 1's value on D, as claim1_program.py: lowest of three starts within FEAS_TOL
        L = LAWS["D"]
        s1, s2 = rng.dirichlet(np.ones(L["m"]), 2)
        runs = solve(L, rho, [(L["p"], 0.999 * rho), (s1, rho / 2), (s2, rho / 4)])
        ok = [x["value"] for x in runs if x["violation"] <= FEAS_TOL]
        V[float(rho)] = (min(ok) if ok else np.nan, float(np.ptp(ok)) if ok else np.nan, len(ok))
    timing["V_D_s"] = time.perf_counter() - t
    E0 = lambda rho: rate(np.array([0, 0, 1 - rho, rho]), 0.5, 0.5, QU, QE)
    X = np.array([[n, np.sqrt(n), np.log(n), 1.0] for n in n_list])
    fits = []
    for inst in ("A", "D"):
        for rho in rhos:
            cs = {c["n"]: c for c in cells if c["instance"] == inst and c["rho"] == rho}
            E, c_, b, d = np.linalg.lstsq(X, np.array([cs[n]["neg_ln_beta"] for n in n_list]), rcond=None)[0]
            ref = E0(rho) if inst == "A" else V[float(rho)][0]
            fits.append({"instance": inst, "rho": float(rho), **{f"E_{n}": cs[n]["E_n"] for n in n_list},
                         "E_fit": E, "c_sqrt_n": c_, "b_ln_n": b, "d": d,
                         "reference": "E_0 (sec. 3.3)" if inst == "A" else "V_D (Theorem 1)", "ref_value": ref,
                         "rel_gap": (E - ref) / ref, "verdict_cell": bool(rho <= VERDICT_RHO + 1e-12),
                         "half_E0_A": E0(rho) / 2 if inst == "D" else np.nan,
                         "green_count_bound": kl_bern(0.5 + rho, LAWS["D"]["q"]) if inst == "D" else np.nan,
                         "all_brackets_closed": all(cs[n]["closed"] for n in n_list)})
    cal = {f["rho"]: abs(f["rel_gap"]) for f in fits if f["instance"] == "A" and f["verdict_cell"]}
    cal_err = max(cal.values())
    tol = max(2 * cal_err, GAP_FLOOR)
    for f in fits:
        f["tolerance"] = tol if f["instance"] == "D" and f["verdict_cell"] else np.nan
        f["contradicted"] = bool(abs(f["rel_gap"]) > tol) if f["instance"] == "D" and f["verdict_cell"] else ""
        if f["instance"] == "D" and f["verdict_cell"]:
            add("K9c", "D", f"extrapolated E within max(2 x calibration error, 5%) = {tol:.4g} of V_D, relative, at "
                            f"rho {f['rho']:g}", f["E_fit"], f["ref_value"], tol * f["ref_value"],
                ok=not f["contradicted"], note=f"relative gap {f['rel_gap']:.4g}")
    bad = [f["rho"] for f in fits if f["contradicted"] is True]
    verdict = ("consistent with Theorem 1 at every verdict cell" if not bad else
               "Theorem 1 contradicted at rho " + ", ".join(f"{r:g}" for r in bad))

    write_csv(res_dir / "checks.csv", rows)
    write_csv(res_dir / "minimax.csv", [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in c.items()}
                                        for c in cells])
    write_csv(res_dir / "fit.csv", fits)
    line = {"A": (line_rho, np.array([E0(r) for r in line_rho])),
            "D": (line_rho, np.array([V[float(r)][0] for r in line_rho]))}
    En = {inst: {n: [next(c["E_n"] for c in cells if (c["instance"], c["n"], c["rho"]) == (inst, n, r))
                     for r in rhos] for n in n_list} for inst in ("A", "D")}
    Efit = {inst: [f["E_fit"] for f in fits if f["instance"] == inst] for inst in ("A", "D")}
    fig_path = fig_dir / "minimax_exact.png"
    plot(rhos, line, En, Efit, n_list, fig_path)
    copies = [Path("results/figures") / f"{SLUG}-minimax_exact.png", Path("figures/minimax_exact.png")]
    for dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fig_path, dst)

    counts = {c: [sum(x["pass"] for x in rows if x["check"] == c), sum(x["check"] == c for x in rows)]
              for c in sorted({x["check"] for x in rows})}
    LD = LAWS["D"]
    wall = time.perf_counter() - t_start
    diag = {"verdict": verdict, "calibration_error_A": cal_err, "calibration_error_A_by_rho": cal,
            "tolerance": tol, "check_counts_pass_total": counts, "failing_rows": [x for x in rows if not x["pass"]],
            "V_D": {f"{r:g}": {"V": v[0], "start_spread": v[1], "starts_feasible": v[2]} for r, v in V.items()},
            "V_D0_minus_prop1_E0": V[0.0][0] - float(LD["mu"] @ np.log(LD["N"]) - LD["delta"] * LD["gamma"])
            if 0.0 in V else None,
            "cells_needing_mixture": [(c["instance"], c["n"], c["rho"]) for c in cells if len(c["support"]) > 1],
            "cells_worst_j_not_J": [(c["instance"], c["n"], c["rho"], c["worst_j_first"]) for c in cells
                                    if c["worst_j_first"] != c["J"]],
            "largest_miss_J_vs_L_rel": max(abs(c["miss_J_over_L"] - 1) for c in cells if len(c["support"]) == 1),
            "run": {"seed": args.seed, "alpha": args.alpha, "n_list": n_list, "rho_grid": rhos.tolist(),
                    "timing_s": timing, "wall_time_s": wall}}
    with open(res_dir / "diagnostic.json", "w") as f:
        json.dump(diag, f, indent=2, default=float)

    for x in rows:
        print(f"{x['check']} {x['instance']} {'PASS' if x['pass'] else 'FAIL'}  {x['statement']}: value "
              f"{x['value']:.12g}, reference {x['reference']:.12g}, diff {x['diff']:.3g}"
              + (f"  [{x['note']}]" if x["note"] else ""))
    print("pass/total per check:", json.dumps(counts))
    for c in cells:
        print(f"  {c['instance']} n {c['n']:4d} rho {c['rho']:.2f} J {c['J']:3d}: beta* {c['L']:.6e}, U/L "
              f"{c['U_over_L']:.8f}, E_n {c['E_n']:.6f}, support {c['support']}, {c['seconds']:.1f} s")
    print("fit: instance rho | E_n per n | E_fit | reference | relative gap")
    for f in fits:
        print(f"  {f['instance']} {f['rho']:.2f} | " + " ".join(f"{f[f'E_{n}']:.5f}" for n in n_list)
              + f" | {f['E_fit']:.5f} | {f['ref_value']:.5f} | {f['rel_gap']:+.4f}"
              + (f" | half E0_A {f['half_E0_A']:.5f}, green-count {f['green_count_bound']:.5f}"
                 if f["instance"] == "D" else ""))
    print(f"calibration error on A {cal_err:.4g} -> tolerance {tol:.4g}; verdict on D: {verdict}")
    print(f"wrote {res_dir}/{{checks,minimax,fit}}.csv, diagnostic.json; {fig_path} -> " + ", ".join(map(str, copies)))
    print(f"wall time {wall:.1f} s")


if __name__ == "__main__":
    main()
