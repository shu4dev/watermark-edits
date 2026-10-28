#!/usr/bin/env python
"""Exact false-negative exponent of green-list detection under budgeted key-blind substitutions.

A green-list watermark (Kirchenbauer et al. 2023) raises the logits of a secret subset of the vocabulary at every
position; the detector counts how many tokens landed in those subsets. This repository computes, without sampling,
how fast the detector's miss rate decays with text length when an attacker has replaced a fraction rho of the tokens,
knowing the scheme but not the key, or seeing the key at a fraction lambda of the positions. The computation runs in
a token model with two entropy levels, where every probability is a finite sum, so the miss rate is exact down to
10^-700 and the exponent and the critical edit fraction come out as numbers rather than estimates.

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
from scipy.stats import binom

from wmsim.attack import class_weights
from wmsim.exponent import kl_bern, rate, rho_star
from wmsim.greencount import gc_tables, log_beta_gc, threshold
from wmsim.oraclenp import np_alt, np_tables, np_test
from wmsim.tokenmodel import level_counts, q_levels, spike_diagnostic
from wmsim.util import logconv

SLUG = "2026-10-25-greenlist-substitution-exponent"
N_BIG = 10_000      # extra n, lambda = 0 only: green count, no NP grid (plan amendment 2026-10-28)


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
            lbg = log_beta_gc(logPM, B, logF0, h, tau, UH, UL)
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
