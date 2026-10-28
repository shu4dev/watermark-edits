#!/usr/bin/env python
"""Exact false-negative exponent of green-list detection under budgeted key-blind substitutions.

Plan: plans/2026-10-25-greenlist-substitution-exponent.md (two-level i.i.d. token model).
Positions: uniform level (V tokens) w.p. pi, spike level (top token 0.9, rest uniform) otherwise; counts
fixed at n_H = floor(pi n), n_L = n - n_H, leaked L_l = floor(lambda n_l). Watermark: independent green
lists (each token green w.p. gamma), soft bias delta; q_l = P(emitted token green) at level l.
Attack, budget B = floor(rho n): flip leaked greens (uniform level first; leaked reds cost nothing), then
edit unleaked uniform, then unleaked spike positions (replacement green w.p. gamma). The leaked greens
K_l ~ Bin(L_l, q_l) are random and every law below is the exact mixture over (K_H, K_L).
Green count: exact Bin(n, gamma) threshold tau_n, alternative in log space; E_n = -ln beta_n / n at each n,
plus n = N_BIG at lambda = 0 (green count only). Oracle NP: exact non-randomised likelihood-ratio test on the
(S_H, S_L) grid of per-level green counts, linear space. At lambda = 0 (K = 0, class counts exact): Cramer rates
I_n(tau_n / n), I_n(gamma) and D(Bern(gamma) || Bern(qbar)). rho_hat*_n: beta_n crosses 1/2 (interpolated on
the grid); rho*_n, rho*: roots in rho of the K-at-mean green rate minus tau_n / n, resp. gamma.
"""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import binom

SLUG = "2026-10-25-greenlist-substitution-exponent"
TOP = 0.9           # spike level: probability of the top token
N_BIG = 10_000      # extra n, lambda = 0 only: green count, no NP grid (plan amendment 2026-10-28)


def green_prob(x, delta):
    """P(emitted token is green | p_t puts mass x on G_t) under the soft watermark."""
    e = np.exp(delta)
    return e * x / (1.0 + (e - 1.0) * x)


def pmf(N, p):
    return binom.pmf(np.arange(N + 1), N, p)


def pad(v, size):
    out = np.zeros(size)
    out[: len(v)] = v
    return out


def logconv(a, b):
    """Log of the convolution of two pmfs given in log space (direct sum, no floor)."""
    if len(a) < len(b):
        a, b = b, a
    out = np.full(len(a) + len(b) - 1, -np.inf)
    for j, bj in enumerate(b):
        if bj > -np.inf:
            seg = out[j: j + len(a)]
            np.logaddexp(seg, a + bj, out=seg)
    return out


def level_counts(n, lam, pi):
    nH = int(np.floor(pi * n + 1e-9))
    nL = n - nH
    LH, LL = int(np.floor(lam * nH + 1e-9)), int(np.floor(lam * nL + 1e-9))
    return nH, nL, LH, LL, nH - LH, nL - LL


def threshold(n, gamma, alpha):
    """Smallest tau with P(Bin(n, gamma) >= tau) <= alpha, and that exact size."""
    t = np.arange(n + 2)
    sf = binom.sf(t - 1, n, gamma)
    tau = int(t[np.argmax(sf <= alpha)])
    return tau, float(sf[tau])


# ---------------------------------------------------------------- per-level green probabilities
def q_levels(gamma, delta, V, trials, rng):
    qH = float(green_prob(gamma, delta))  # closed form (|G_t|/V -> gamma)
    top_green = rng.random(trials) < gamma
    k = rng.binomial(V - 1, gamma, size=trials)
    q = green_prob(TOP * top_green + (1 - TOP) * k / (V - 1), delta)  # P(green | key), per key draw
    qL_formula = gamma * green_prob(TOP + (1 - TOP) * gamma, delta) + (1 - gamma) * green_prob((1 - TOP) * gamma, delta)
    return qH, float(q.mean()), float(q.std(ddof=1) / np.sqrt(trials)), float(qL_formula)


def spike_diagnostic(gamma, delta, V, trials, rng):
    """P(top token in G_t | a tail token was emitted) at the spike level: closed form and Monte Carlo."""
    e = np.exp(delta)
    c = gamma * e + 1 - gamma                                 # tail reweighting at tail green fraction gamma
    Zg, Zr = TOP * e + (1 - TOP) * c, TOP + (1 - TOP) * c    # Z_t when the top token is green / red
    p_closed = gamma * Zr / (gamma * Zr + (1 - gamma) * Zg)
    top_green = rng.random(trials) < gamma
    k = rng.binomial(V - 1, gamma, size=trials)
    tail = (1 - TOP) * (e * k + (V - 1 - k)) / (V - 1)      # unnormalised watermarked tail mass
    topm = TOP * np.where(top_green, e, 1.0)
    tail_emitted = rng.random(trials) < tail / (tail + topm)  # one emitted token per key draw
    N = int(tail_emitted.sum())
    p_mc = float(top_green[tail_emitted].mean())
    return dict(p_top_green_given_tail_closed=float(p_closed), p_top_green_given_tail_mc=p_mc,
                p_top_green_given_tail_mc_se=float(np.sqrt(p_mc * (1 - p_mc) / N)), mc_keys=trials,
                mc_tail_emissions=N, red_prob_bayes_swap_closed=float(1 - p_closed),
                red_prob_bayes_swap_mc=1 - p_mc, red_prob_keyblind_replacement=1 - gamma,
                Z_top_green=float(Zg), Z_top_red=float(Zr),
                p_tail_emitted_closed=float(gamma * (1 - TOP) * c / Zg + (1 - gamma) * (1 - TOP) * c / Zr))


# ---------------------------------------------------------------- green count (log space)
def gc_tables(UH, UL, qH, qL, gamma, tau):
    """Unleaked positions in attack order (UH uniform, then UL spike); with b of them edited the green count
    there is R(b) = Bin(b, gamma) + sum of the untouched suffix. Returns log CDF of R(0) and
    log P(R(b) <= tau - 1) for b = 0..U."""
    U = UH + UL
    p = np.r_[np.full(UH, qH), np.full(UL, qL)]
    T = np.full((U + 1, U + 1), -np.inf)  # T[b] = log pmf of sum_{i >= b} Bern(p_i)
    T[U, 0] = 0.0
    for b in range(U - 1, -1, -1):
        T[b] = T[b + 1] + np.log1p(-p[b])
        np.logaddexp(T[b, 1:], T[b + 1, :-1] + np.log(p[b]), out=T[b, 1:])
    logF = np.logaddexp.accumulate(T, axis=1)
    a = np.arange(tau)
    LA = binom.logpmf(a[None, :], np.arange(U + 1)[:, None], gamma)
    h = logsumexp(LA + logF[:, np.minimum(tau - 1 - a, U)], axis=1)
    return logF[0], h


def log_beta_gc(logPM, B, logF0, h, tau):
    """log P1(S <= tau - 1); the green count depends on the leaked greens only through M = K_H + K_L."""
    U = len(h) - 1
    m = np.arange(len(logPM))
    x = tau - 1 - (m - B)                       # M >= B: S = (M - B) + R(0)
    hm = np.where(x >= 0, logF0[np.clip(x, 0, U)], -np.inf)
    lo = m < B                                  # M < B: S = R(B - M), at most U edits land
    hm[lo] = h[np.minimum(B - m[lo], U)]
    return float(logsumexp(logPM + hm))


# ---------------------------------------------------------------- oracle NP (linear space)
def np_tables(nH, nL, LH, LL, UH, UL, qH, qL, gamma):
    pKH, pKL = pmf(LH, qH), pmf(LL, qL)
    t = dict(pKH=pKH, pKL=pKL, pM=np.convolve(pKH, pKL), XH=pmf(UH, qH), XL=pmf(UL, qL), GH=pmf(UH, gamma))
    t["cL1"] = np.convolve(pKL, t["XL"])
    t["YH"] = np.array([np.convolve(pmf(b, gamma), pmf(UH - b, qH)) for b in range(UH + 1)])  # b uniform edits
    t["ZL"] = np.array([np.convolve(pmf(j, gamma), pmf(UL - j, qL)) for j in range(UL + 1)])  # j spike edits
    t["logP0"] = (binom.logpmf(np.arange(nH + 1), nH, gamma)[:, None]
                  + binom.logpmf(np.arange(nL + 1), nL, gamma)[None, :])
    t["P0"] = np.exp(t["logP0"])
    return t


def np_alt(t, B, nH, nL, LH, UH, UL):
    """P1(S_H, S_L): exact mixture over (K_H, K_L), which collapses to four outer products."""
    pKH, pKL, pM = t["pKH"], t["pKL"], t["pM"]
    L = len(pM) - 1
    P1 = np.zeros((nH + 1, nL + 1))
    if B <= LH:  # K_H >= B: whole budget on uniform-level leaked greens
        P1 += np.outer(pad(np.convolve(pKH[B:], t["XH"]), nH + 1), t["cL1"])
    if B >= 1:
        low = np.convolve(pKH[:B], pKL)  # law of M on {K_H < B}
        if len(low) > B:  # K_H < B <= M: the rest on spike-level leaked greens
            P1 += np.outer(pad(t["XH"], nH + 1), pad(np.convolve(low[B:], t["XL"]), nL + 1))
        b = np.arange(max(1, B - L), B + 1)  # M < B: every leaked green flipped, b = B - M edits left
        w = pM[B - b]
        s = b <= UH  # leftover lands on unleaked uniform positions
        if s.any():
            P1 += np.outer(pad(w[s] @ t["YH"][b[s]], nH + 1), pad(t["XL"], nL + 1))
        if (~s).any():  # uniform level exhausted, the rest on unleaked spike positions
            j = np.minimum(b[~s] - UH, UL)
            P1 += np.outer(pad(t["GH"], nH + 1),
                           pad(np.bincount(j, weights=w[~s], minlength=UL + 1) @ t["ZL"], nL + 1))
    return P1


def np_test(P1, t, alpha):
    """Non-randomised NP: reject cells by decreasing P1/P0 until the null mass would exceed alpha."""
    with np.errstate(divide="ignore"):
        llr = np.log(P1) - t["logP0"]
    order = np.argsort(-llr, axis=None, kind="stable")
    cum = np.cumsum(t["P0"].ravel()[order])
    k = int(np.searchsorted(cum, alpha, side="right"))
    return float(P1.ravel()[order[k:]].sum()), (float(cum[k - 1]) if k else 0.0)


# ---------------------------------------------------------------- rate function, KL, rho*
def class_weights(fr, rho, qH, qL):
    """Class fractions with K at its mean, fr = (L_H, L_L, U_H, U_L) / n, rho = B / n: weights of
    q' = 1 (leaked unflipped), gamma (edited), q_H, q_L (unedited); the rest (flipped) has q' = 0."""
    LH, LL, UH, UL = fr
    k = LH * qH + LL * qL
    r = max(rho - k, 0.0)
    eH = min(r, UH)
    eL = min(r - eH, UL)
    return np.array([k - min(rho, k), eH + eL, UH - eH, UL - eL])


def rate(w, t, gamma, qH, qL):
    """I(t) = sup_{theta <= 0} [theta t - sum_c w_c log(1 - q_c + q_c e^theta)] over classes q_c = 1, gamma, q_H, q_L."""
    probs = np.array([gamma, qH, qL])
    w1, wr = w[0], w[1:]
    if w1 + wr @ probs <= t:
        return 0.0
    if w1 >= t:  # surviving leaked greens alone reach t n
        return np.inf
    dLam = lambda th: w1 + wr @ (probs * np.exp(th) / (1 - probs + probs * np.exp(th)))
    lo = -1.0
    while dLam(lo) > t:
        lo *= 2
    th = brentq(lambda x: dLam(x) - t, lo, 0.0, xtol=1e-14)
    return float(th * t - (w1 * th + wr @ np.log1p(probs * np.expm1(th))))


def kl_bern(a, b):
    """D(Bern(a) || Bern(b)) when b > a, else 0."""
    return float(a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))) if b > a else 0.0


def rho_star(fr, target, gamma, qH, qL):
    """Root in rho of the K-at-mean green rate (piecewise linear) minus target; 1.0 if it never drops below."""
    f = lambda r: class_weights(fr, r, qH, qL) @ np.array([1.0, gamma, qH, qL]) - target
    return 1.0 if f(1.0) >= -1e-12 else float(brentq(f, 0.0, 1.0, xtol=1e-12))


# ---------------------------------------------------------------- figure
def plot(exps, star, lam_grid, n_ref, n_big, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, muted, gridc, axc, surf = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    c_gc, c_np, c_rate = "#2a78d6", "#eb6834", "#1baf7a"
    col = lambda lam, n, k: np.array([x[k] if np.isfinite(x[k]) else np.nan
                                      for x in exps if x["lambda"] == lam and x["n"] == n], float)
    vals = np.concatenate([col(lam, n, k) for lam in lam_grid for n in (n_ref, n_big)
                           for k in ("E_gc", "E_np", "I_n_tau", "I_n_gamma", "kl_mean")])
    top = 1.08 * np.nanmax(vals)
    fig, axes = plt.subplots(1, len(lam_grid), figsize=(3.0 * len(lam_grid), 3.5), sharey=True, facecolor=surf)
    axes = np.atleast_1d(axes)
    mk = dict(marker="o", ms=4.5, mec=surf, mew=1.2)
    for ax, lam in zip(axes, lam_grid):
        rho = col(lam, n_ref, "rho")
        s = next(x for x in star if x["lambda"] == lam and x["n"] == n_ref)
        ax.set_facecolor(surf)
        if lam == 0:
            ax.plot(rho, col(lam, n_ref, "kl_mean"), color=muted, lw=1.5, ls="-.",
                    label="KL(Bern(γ) ‖ Bern(q̄ρ)): lower bound on I(γ)")
            ax.plot(rho, col(lam, n_ref, "I_n_gamma"), color=c_rate, lw=2, label="I(γ): Cramér rate at γ (n → ∞)")
            ax.plot(rho, col(lam, n_ref, "I_n_tau"), color=c_rate, lw=1.5, ls="--",
                    label=f"$I_{{{n_ref}}}(\\tau/n)$: Cramér rate at the n = {n_ref} threshold")
            ax.plot(rho, col(lam, n_big, "E_gc"), color=c_gc, lw=1.5, ls="--", marker="o", ms=4.5, mfc=surf,
                    mec=c_gc, mew=1.2, label=f"green count: $E_{{{n_big}}}$ (λ = 0 only)")
        ax.plot(rho, col(lam, n_ref, "E_np"), color=c_np, lw=2, label=f"oracle NP: $E_{{{n_ref}}}$ = −ln β/n", **mk)
        ax.plot(rho, col(lam, n_ref, "E_gc"), color=c_gc, lw=2, label=f"green count: $E_{{{n_ref}}}$ = −ln β/n", **mk)
        ax.axvline(s["rho_star"], color=muted, lw=1.2, ls=":", label="ρ*: mean crossing at γ (n → ∞)")
        ax.axvline(s["rho_star_n"], color=ink2, lw=1.0, ls="--", label=f"$\\rho^*_{{{n_ref}}}$: mean crossing at τ/n")
        if np.isfinite(s["rho_hat_n"]):
            ax.axvline(s["rho_hat_n"], color=ink, lw=1.0, label=f"$\\hat\\rho^*_{{{n_ref}}}$: β = 1/2")
        ax.set_title(f"λ = {lam:g}", color=ink, fontsize=10)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(min(-0.01, np.nanmin(vals) - 0.01), top)
        ax.grid(color=gridc, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(axc)
        ax.tick_params(colors=ink2, labelsize=8)
        ax.set_xlabel("edit fraction ρ", color=ink2, fontsize=9)
    axes[0].set_ylabel("exponent (nat/token)", color=ink2, fontsize=9)
    hl = {}
    for ax in axes:
        for hd, lb in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(lb, hd)
    fig.legend(list(hl.values()), list(hl.keys()), loc="lower center", ncol=3, frameon=False, fontsize=8,
               labelcolor=ink, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surf, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main
def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--V", type=int, default=50_000)
    ap.add_argument("--pi", type=float, default=0.5, help="probability of the uniform (high-entropy) level")
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--rho-grid", type=float, nargs="+", default=[round(0.05 * i, 2) for i in range(20)])
    ap.add_argument("--lambda-grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--n-list", type=int, nargs="+", default=[100, 200, 500, 1000, 2000])
    ap.add_argument("--trials", type=int, default=100_000, help="Monte Carlo key draws (q_L and the diagnostic)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data", help="writes <out>/results/<slug>/ and <out>/figures/<slug>/")
    args = ap.parse_args()
    t_start = time.perf_counter()
    g, d, n_list = args.gamma, args.delta, sorted(args.n_list)
    rng = np.random.default_rng(args.seed)
    qH, qL, qL_se, qL_formula = q_levels(g, d, args.V, args.trials, rng)
    diag = spike_diagnostic(g, d, args.V, args.trials, rng)
    print(f"q_H = {qH:.6f} (closed form); q_L = {qL:.6f} +- {qL_se:.6f} (MC, {args.trials} keys) "
          f"vs two-case formula {qL_formula:.6f}")
    res_dir, fig_dir = Path(args.out) / "results" / SLUG, Path(args.out) / "figures" / SLUG
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows, grid_rel, mass_err = [], 0.0, 0.0
    for n, lam in [(n, lam) for n in n_list for lam in args.lambda_grid] + [(N_BIG, 0.0)]:
        big = n == N_BIG  # lambda = 0: no K mixture; NP grid skipped
        tau, a_gc = threshold(n, g, args.alpha)
        t_cell = time.perf_counter()
        nH, nL, LH, LL, UH, UL = level_counts(n, lam, args.pi)
        logpKH = binom.logpmf(np.arange(LH + 1), LH, qH)
        logPM = logconv(logpKH, binom.logpmf(np.arange(LL + 1), LL, qL))
        logF0, h = gc_tables(UH, UL, qH, qL, g, tau)
        if not big:
            t = np_tables(nH, nL, LH, LL, UH, UL, qH, qL, g)
            below = (np.arange(nH + 1)[:, None] + np.arange(nL + 1)[None, :]) <= tau - 1
        pKH, pM = np.exp(logpKH), np.exp(logPM)
        m = np.arange(len(pM))
        for rho in args.rho_grid:
            B = int(np.floor(rho * n + 1e-9))
            lbg = log_beta_gc(logPM, B, logF0, h, tau)
            b_np = a_np = np.nan
            if not big:
                P1 = np_alt(t, B, nH, nL, LH, UH, UL)
                b_np, a_np = np_test(P1, t, args.alpha)
                mass_err = max(mass_err, abs(P1.sum() - 1.0))
                if lbg > np.log(1e-250):  # linear 2-D grid vs log-space path, where the grid has no underflow
                    grid_rel = max(grid_rel, abs(P1[below].sum() / np.exp(lbg) - 1.0))
            EfH = float(pKH @ np.minimum(B, np.arange(LH + 1)))
            Ef = float(pM @ np.minimum(B, m))
            EeH = float(pM @ np.clip(B - m, 0, UH))
            EeL = float(pM @ np.clip(B - m - UH, 0, UL))
            rows.append({"n": n, "lambda": lam, "rho": rho, "B": B, "tau": tau, "alpha_gc": a_gc,
                         "log10_beta_gc": lbg / np.log(10), "alpha_np": a_np,
                         "log10_beta_np": -np.inf if b_np == 0 else np.log10(b_np),
                         "n_H": nH, "n_L": nL, "L_H": LH, "L_L": LL, "U_H": UH, "U_L": UL,
                         "E_K_H": LH * qH, "E_K_L": LL * qL, "E_flip_H": EfH, "E_flip_L": Ef - EfH,
                         "E_edit_unleaked_H": EeH, "E_edit_unleaked_L": EeL, "E_edits_used": Ef + EeH + EeL})
        print(f"n={n:5d} lambda={lam:.2f} tau={tau} alpha_gc={a_gc:.3e} "
              f"({time.perf_counter() - t_cell:.1f}s)", flush=True)
    print(f"checks: max |sum P1 - 1| = {mass_err:.2e}; max rel. diff green-count beta, 2-D grid vs log space "
          f"(beta > 1e-250) = {grid_rel:.2e}")

    n_ref = n_list[-1]
    exps = []
    for x in rows:
        n, lam = x["n"], x["lambda"]
        e = {"lambda": lam, "rho": x["rho"], "n": n, "tau_over_n": x["tau"] / n,
             "E_gc": -np.log(10) * x["log10_beta_gc"] / n,
             "E_np": -np.log(10) * x["log10_beta_np"] / n if np.isfinite(x["log10_beta_np"]) else np.nan,
             "I_n_tau": np.nan, "I_n_gamma": np.nan, "kl_mean": np.nan, "qbar": np.nan}
        if lam == 0.0:  # K = 0: class counts exact. No analytic curve at lambda > 0 (K at its mean is wrong there)
            w = class_weights(np.array(level_counts(n, lam, args.pi)[2:]) / n, x["B"] / n, qH, qL)
            qb = float(w @ np.array([1.0, g, qH, qL]))
            e.update(I_n_tau=rate(w, x["tau"] / n, g, qH, qL), I_n_gamma=rate(w, g, g, qH, qL),
                     kl_mean=kl_bern(g, qb), qbar=qb)
        exps.append(e)
    star = []
    for lam in args.lambda_grid:
        fr_inf = np.array([lam * args.pi, lam * (1 - args.pi), (1 - lam) * args.pi, (1 - lam) * (1 - args.pi)])
        rs = rho_star(fr_inf, g, g, qH, qL)
        for n in sorted({x["n"] for x in rows if x["lambda"] == lam}):
            r = [x for x in rows if x["lambda"] == lam and x["n"] == n]
            rho, beta = np.array([x["rho"] for x in r]), 10.0 ** np.array([x["log10_beta_gc"] for x in r])
            i = int(np.argmax(beta > 0.5))
            hat = (np.nan if beta[i] <= 0.5 else float(rho[0]) if i == 0
                   else float(np.interp(0.5, beta[i - 1: i + 1], rho[i - 1: i + 1])))
            rn = rho_star(np.array(level_counts(n, lam, args.pi)[2:]) / n, r[0]["tau"] / n, g, qH, qL)
            star.append({"lambda": lam, "n": n, "tau_over_n": r[0]["tau"] / n, "rho_hat_n": hat,
                         "rho_star_n": rn, "diff": hat - rn, "rho_star": rs})

    write_csv(res_dir / "beta.csv", rows)
    write_csv(res_dir / "exponents.csv", exps)
    write_csv(res_dir / "rhostar.csv", star)
    fig_path = fig_dir / "exponent_vs_rho.png"
    plot(exps, star, args.lambda_grid, n_ref, N_BIG, fig_path)
    curated = Path("results/figures") / f"{SLUG}-exponent_vs_rho.png"
    curated.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(fig_path, curated)

    # Works-when checks (a)-(f); (g) is the diagnostic block
    e0 = [x for x in exps if x["lambda"] == 0.0]
    big0 = {x["rho"]: x for x in e0 if x["n"] == N_BIG}
    ref0 = [x for x in e0 if x["n"] == n_ref]
    emin = min(e0, key=lambda x: x["E_gc"])
    c_cells = [{"rho": x["rho"], "neg_ln_beta": n_ref * x["E_gc"], "E": x["E_gc"], "I_n_tau": x["I_n_tau"],
                "rel_gap": (x["E_gc"] - x["I_n_tau"]) / x["I_n_tau"]} for x in ref0 if n_ref * x["E_gc"] >= 100]
    d_cells = [{"rho": x["rho"], "E_ref": x["E_gc"], "E_big": big0[x["rho"]]["E_gc"], "I_gamma": x["I_n_gamma"],
                "gap_ref": abs(x["E_gc"] - x["I_n_gamma"]), "gap_big": abs(big0[x["rho"]]["E_gc"] - x["I_n_gamma"])}
               for x in ref0 if min(x["E_gc"], big0[x["rho"]]["E_gc"]) > 0.01]
    viol = [x for x in rows if x["log10_beta_np"] > x["log10_beta_gc"] + 1e-9]
    outputs = [res_dir / f for f in ("beta.csv", "exponents.csv", "rhostar.csv")] + [fig_path, curated]
    checks = {
        "a_outputs_exist": all(p.exists() for p in outputs),
        "b_lambda0_E_positive_all_rho_n": all(x["E_gc"] > 0 for x in e0),
        "b_lambda0_min_E": {"E": emin["E_gc"], "rho": emin["rho"], "n": emin["n"]},
        "c_cells": c_cells,
        "c_max_abs_rel_gap": max((abs(c["rel_gap"]) for c in c_cells), default=np.nan),
        "c_within_5pct": bool(c_cells) and all(abs(c["rel_gap"]) <= 0.05 for c in c_cells),
        "d_cells": d_cells,
        "d_big_closer_all": bool(d_cells) and all(c["gap_big"] < c["gap_ref"] for c in d_cells),
        "d_I_gamma_spread_over_n": max(np.ptp([x["I_n_gamma"] for x in e0 if x["rho"] == r]) for r in big0),
        "e_max_abs_diff": float(np.nanmax([abs(s["diff"]) for s in star])),
        "e_within_0.05_all": all(abs(s["diff"]) <= 0.05 for s in star),
        "f_np_above_gc_cells": [{k: x[k] for k in ("n", "lambda", "rho", "log10_beta_gc", "log10_beta_np",
                                                    "alpha_gc", "alpha_np")} for x in viol],
        "f_all_explained_by_discreteness": bool(all(x["alpha_gc"] > x["alpha_np"] for x in viol)),
        "max_abs_P1_mass_error": mass_err, "max_rel_diff_gc_grid_vs_log": grid_rel,
    }
    wall = time.perf_counter() - t_start
    out = {"q_H_closed": qH, "q_L_mc": qL, "q_L_mc_se": qL_se, "q_L_two_case_formula": qL_formula,
           "q_L_abs_diff": abs(qL - qL_formula), "qbar0": args.pi * qH + (1 - args.pi) * qL,
           "spike_diagnostic": diag, "checks": checks,
           "run": {"seed": args.seed, "gamma": g, "delta": d, "V": args.V, "pi": args.pi, "alpha": args.alpha,
                   "n_list": n_list, "n_big_lambda0": N_BIG, "lambda_grid": args.lambda_grid,
                   "rho_grid": args.rho_grid, "trials": args.trials, "wall_time_s": wall}}
    with open(res_dir / "diagnostic.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    print(json.dumps(diag, indent=1))
    print("lambda=0:  rho  E_%d  I_%d(tau/n)  rel_gap  E_%d  I(gamma)  KL" % (n_ref, n_ref, N_BIG))
    for x in ref0:
        y = big0[x["rho"]]
        print(f"  {x['rho']:.2f}  {x['E_gc']:.5f}  {x['I_n_tau']:.5f}  {(x['E_gc'] - x['I_n_tau']) / x['I_n_tau'] if x['I_n_tau'] > 0 else np.inf:+.4f}"
              f"  {y['E_gc']:.5f}  {x['I_n_gamma']:.5f}  {x['kl_mean']:.5f}")
    print("rho_hat*_n vs rho*_n (and rho*):")
    for s in star:
        print(f"  lambda={s['lambda']:.2f} n={s['n']:5d} tau/n={s['tau_over_n']:.4f} rho_hat*_n={s['rho_hat_n']:.4f}"
              f" rho*_n={s['rho_star_n']:.4f} diff={s['diff']:+.4f} rho*={s['rho_star']:.4f}")
    print(json.dumps({k: v for k, v in checks.items() if k not in ("c_cells", "d_cells", "f_np_above_gc_cells")},
                     indent=1, default=float))
    print(f"NP above green count in {len(viol)} cells")
    print(f"wrote {res_dir}/{{beta,exponents,rhostar}}.csv, diagnostic.json; {fig_path} -> {curated}")
    print(f"wall time {wall:.1f} s")

if __name__ == "__main__":
    main()
