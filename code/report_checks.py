#!/usr/bin/env python
"""Numerical checks of the report's statements (main.tex sections 3.1-3.5) on five small instances.

Single-position law by enumeration over the green lists G with mu(G) > 0: N(G) = 1 + (e^delta - 1) p(G),
P_{Y|G}(y) = p(y) e^{delta 1[y in G]} / N(G), P_GY = mu P_{Y|G}, P_Y its marginal, q = sum_G mu(G) P_{Y|G}(G).
Instances: A (m 2, lists of exactly k = 1 token, e^delta 3, uniform p), B (m 4, k 1, delta 2, uniform p), and
C1, C2, C3 (m 4, Bernoulli lists, gamma 0.25, delta 2; p uniform, (0.9, 1/30, 1/30, 1/30), Dirichlet(1,1,1,1)).
K1 Prop. 1, all five: F(Q) = D(mu x Q || P_GY); F(p) = E(0), F(Q) - D(Q||p) = E(0) for 1000 Dirichlet(1) Q.
K2 Cor. 1 by LP, C1-C3: R = Sigma; R(y) = {y, 2, 3} on C2; R(y) = {y, y+1 mod 4} must be infeasible.
K3 q and r of section 3.3 by enumeration, A and B. K4 inequality (2) on 1000 random Bernoulli instances.
K5 key-blind, A and B: rho_0* = (m-1)/m by identity and by LP; green-count miss rate under j = floor(rho n)
blind substitutions, beta_n = P(Bin(n-j, q) + Bin(j, r) <= tau_n - 1), against the rate at tau_n/n and E_0(rho).
K6 informed, A and B: beta_n = P(Bin(n, q) <= tau_n - 1 + j) against d((tau_n + j)/n || q) and E_1(rho).
K5 and K6 rows hold the exponent rules: (i) E_lo within 5 % of the rate at its threshold where -ln beta_lo >= 100;
(ii) E_hi closer to the limit than E_lo where both exceed 0.01; (iii) beta falls from n_lo to n_hi at grid rho
below rho*, beta_hi >= 0.999 above. A grid rho equal to rho* has no pass rule.
K7 Claim 3, all five: rho_priv(lambda) by LP over A0(z|y), A1(z|y,G), nu with R = Sigma; equals the K2 LP at
lambda 0, does not increase, equals q - gamma at 1 (C1-C3: not in main.tex), stays in [q - gamma, rho_priv(0)].
Tolerances: 1e-12 for identities, 1e-8 for LP values. The detector flags at green count >= tau_n.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import linprog
from scipy.special import logsumexp, rel_entr
from scipy.stats import binom

from wmsim.exponent import kl_bern, rate
from wmsim.greencount import threshold
from wmsim.singlepos import LABELS, instances, law
from wmsim.util import write_csv

SLUG = "2026-11-12-report-checks"
ID_TOL, LP_TOL = 1e-12, 1e-8
LAM_REPORT = (0.0, 0.25, 0.5, 0.75, 1.0)


def lp_blind(L, R=None):
    """Claim 2's LP: min P(Y != Z) over channels A(z|y), zero outside R(y), with sum_y P_{Y|G}(y) A(z|y) = nu(z)
    for every G and z. Variables A (y, z) row-major, then nu. Returns (value or nan, HiGHS status, message)."""
    m, PYG = L["m"], L["PYG"]
    I = np.eye(m)
    Aeq = np.block([[np.kron(I, np.ones(m)), np.zeros((m, m))],
                    [np.kron(PYG, I), -np.tile(I, (len(PYG), 1))]])
    beq = np.r_[np.ones(m), np.zeros(len(PYG) * m)]
    c = np.r_[-(I * L["PY"][:, None]).ravel(), np.zeros(m)]
    ok = np.ones((m, m), bool) if R is None else R
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=[(0, None if a else 0) for a in ok.ravel()] + [(0, None)] * m,
                  method="highs")
    return (1 + res.fun if res.status == 0 else np.nan), res.status, res.message


def lp_leak(L, lam):
    """Claim 3's LP at leakage lam, R = Sigma: blind A0(z|y) w.p. 1 - lam, informed A1(z|y,G) w.p. lam, with
    (1 - lam) sum_y P_{Y|G}(y) A0(z|y) + lam sum_y P_{Y|G}(y) A1(z|y,G) = nu(z) for every G and z.
    Variables A0 (y, z), A1 (G, y, z), nu. Returns (value or nan, HiGHS status)."""
    m, PYG, PGY = L["m"], L["PYG"], L["PGY"]
    nG, I = len(PYG), np.eye(m)
    Aeq = np.block([
        [np.kron(I, np.ones(m)), np.zeros((m, nG * m * m + m))],
        [np.zeros((nG * m, m * m)), np.kron(np.eye(nG * m), np.ones(m)), np.zeros((nG * m, m))],
        [(1 - lam) * np.kron(PYG, I), lam * block_diag(*[np.kron(row, I) for row in PYG]), -np.tile(I, (nG, 1))]])
    beq = np.r_[np.ones(m + nG * m), np.zeros(nG * m)]
    c = np.r_[-(1 - lam) * (I * L["PY"][:, None]).ravel(), -lam * (I * PGY[:, :, None]).ravel(), np.zeros(m)]
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
    return (1 + res.fun if res.status == 0 else np.nan), res.status


def plot(lams, rp, ends, path):
    """One panel: rho_priv against lambda, one curve per instance; the closed-form end points as black rings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, gridc, axc, surf = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")  # categorical slots 1-5, fixed order
    fig, ax = plt.subplots(figsize=(8.2, 4.2), facecolor=surf)
    ax.set_facecolor(surf)
    for (name, y), col, mk in zip(rp.items(), colors, "osD^v"):  # marker shapes as well: curves share points
        ax.plot(lams, y, color=col, lw=2, marker=mk, ms=4.5, mec=surf, mew=1.2, label=LABELS[name])
    ax.plot(*zip(*ends), ls="none", marker="o", ms=10, mfc="none", mec=ink, mew=1.3,
            label="closed form at λ = 0 and 1")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, None)
    ax.grid(color=gridc, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(axc)
    ax.tick_params(colors=ink2, labelsize=10)
    ax.set_xlabel("leakage λ", color=ink2, fontsize=10)
    ax.set_ylabel(r"$\rho_{\mathrm{priv}}(\lambda)$, least edit fraction with $G \perp Z$", color=ink2, fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=10, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surf)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--n-list", type=int, nargs="+", default=[2000, 10_000], help="rules compare min and max n")
    ap.add_argument("--rho-grid", type=float, nargs="+", default=[round(0.05 * i, 2) for i in range(20)])
    ap.add_argument("--lambda-step", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0, help="C3's p, K1's Q and K4's draws")
    ap.add_argument("--out", default="data", help="writes <out>/results/<slug>/ and <out>/figures/<slug>/")
    args = ap.parse_args()
    t_start = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    n_lo, n_hi = min(args.n_list), max(args.n_list)
    lams = np.round(np.linspace(0, 1, int(round(1 / args.lambda_step)) + 1), 10)
    res_dir, fig_dir = Path(args.out) / "results" / SLUG, Path(args.out) / "figures" / SLUG
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    laws = instances(rng)
    qr = {}  # section 3.3's closed forms, fixed-size instances
    for name in ("A", "B"):
        L = laws[name]
        ed = np.exp(L["delta"])
        q = L["gamma"] * ed / (1 - L["gamma"] + L["gamma"] * ed)
        qr[name] = (q, (L["k"] - q) / (L["m"] - 1))
    rows = []

    def add(check, inst, statement, value, ref, tol, ok=None, note=""):
        value, ref = float(value), float(ref)
        rows.append({"check": check, "instance": inst, "statement": statement, "value": value, "reference": ref,
                     "diff": value - ref, "tol": tol, "pass": bool(abs(value - ref) <= tol if ok is None else ok),
                     "note": note})

    # K1 Prop. 1
    e0 = {}
    for name, L in laws.items():
        e0[name] = float(L["mu"] @ np.log(L["N"]) - L["delta"] * L["gamma"])
        F = lambda Q: float(rel_entr(np.outer(L["mu"], Q), L["PGY"]).sum())
        add("K1", name, "F(p) = E(0)", F(L["p"]), e0[name], ID_TOL)
        gap = np.array([F(Q) - rel_entr(Q, L["p"]).sum() - e0[name] for Q in rng.dirichlet(np.ones(L["m"]), 1000)])
        add("K1", name, "F(Q) - D(Q||p) = E(0) for 1000 Dirichlet(1) Q (worst shown)",
            e0[name] + gap[np.argmax(np.abs(gap))], e0[name], ID_TOL)

    # K2 Cor. 1 (and K5's LP on A and B): R = Sigma on all five
    lp0 = {name: lp_blind(L) for name, L in laws.items()}
    for name in ("C1", "C2", "C3"):
        add("K2", name, "LP = 1 - max_z P_Y(z), R = Sigma", lp0[name][0], 1 - laws[name]["PY"].max(), LP_TOL)
    R23 = np.eye(4, dtype=bool)
    R23[:, [2, 3]] = True
    L = laws["C2"]
    add("K2", "C2", "LP = 1 - max_{z in {2,3}} P_Y(z), R(y) = {y, 2, 3}", lp_blind(L, R23)[0],
        1 - L["PY"][[2, 3]].max(), LP_TOL)
    R_next = np.eye(4, dtype=bool) | np.roll(np.eye(4, dtype=bool), 1, axis=1)
    for name in ("C1", "C2", "C3"):
        _, st, msg = lp_blind(laws[name], R_next)
        add("K2", name, "LP infeasible (HiGHS status 2), R(y) = {y, y+1 mod 4}, C empty", st, 2, 0, note=msg)

    # K3 q and r, K5 rho_0* (A and B)
    for name in ("A", "B"):
        L, (q, r) = laws[name], qr[name]
        m, g = L["m"], L["gamma"]
        cond = L["PGY"].T @ L["G"] / L["PY"][:, None]  # [y, z] = P(z in G | Y = y)
        off = cond[~np.eye(m, dtype=bool)]
        add("K3", name, "q by enumeration = gamma e^delta / (1 - gamma + gamma e^delta)", L["q"], q, ID_TOL)
        add("K3", name, "P(z in G | Y = y), every z != y (worst shown) = r = (k - q)/(m - 1)",
            off[np.argmax(np.abs(off - r))], r, ID_TOL)
        add("K3", name, "r < gamma", r, g, 0, ok=r < g)
        rs = (m - 1) / m
        add("K5", name, "(1 - rho) q + rho r = gamma at rho = (m-1)/m", (1 - rs) * q + rs * r, g, ID_TOL)
        add("K5", name, "K2 LP with the fixed-size mu, R = Sigma = (m-1)/m", lp0[name][0], rs, LP_TOL)

    # K4 inequality (2), Bernoulli membership
    k4 = []
    for _ in range(1000):
        m, conc = int(rng.integers(2, 9)), float(rng.choice([0.1, 1.0, 10.0]))
        p = rng.dirichlet(np.full(m, conc))
        g, d = float(rng.uniform(0.05, 0.95)), float(rng.uniform(0.1, 4.0))
        bound = g - np.expm1(-d) * g * (1 - g) * (1 - p @ p)
        k4.append({"gap": law(m, g, d, p)["q"] - bound, "m": m, "concentration": conc, "gamma": g, "delta": d,
                   "p": p.tolist()})
    k4w = min(k4, key=lambda x: x["gap"])
    add("K4", "1000 draws", "min over draws of q_t - bound >= -1e-12", k4w["gap"], 0, ID_TOL,
        ok=k4w["gap"] >= -ID_TOL, note=f"at m {k4w['m']}, concentration {k4w['concentration']:g}, "
                                       f"gamma {k4w['gamma']:.4f}, delta {k4w['delta']:.4f}, "
                                       f"sum p^2 {np.square(k4w['p']).sum():.6f}")

    # K5, K6 exact miss rates of the green-count test (A and B)
    ex = []
    for name in ("A", "B"):
        L, (q, r) = laws[name], qr[name]
        m, g = L["m"], L["gamma"]
        for n in sorted(args.n_list):
            tau, a_size = threshold(n, g, args.alpha)
            lq = binom.logpmf(np.arange(n + 1), n, q)
            Fq = np.logaddexp.accumulate(lq)
            for rho in args.rho_grid:
                j = int(np.floor(rho * n + 1e-9))
                Fa = np.logaddexp.accumulate(binom.logpmf(np.arange(n - j + 1), n - j, q))
                l = np.arange(min(j, tau - 1) + 1)
                lb0 = float(logsumexp(binom.logpmf(l, j, r) + Fa[np.minimum(tau - 1 - l, n - j)]))
                cells = (("E0", lb0, j, rate(np.array([0, 0, 1 - j / n, j / n]), tau / n, g, q, r),
                          rate(np.array([0, 0, 1 - rho, rho]), g, g, q, r), (m - 1) / m),
                         ("E1", float(Fq[min(tau - 1 + j, n)]), float(np.exp(lq) @ np.minimum(j, np.arange(n + 1))),
                          kl_bern((tau + j) / n, q), kl_bern(g + rho, q), q - g))
                for which, lb, used, rt, lim, star in cells:
                    ex.append({"instance": name, "exponent": which, "n": n, "rho": rho, "j": j, "edits_used": used,
                               "tau": tau, "alpha_size": a_size, "log10_beta": lb / np.log(10), "neg_ln_beta": -lb,
                               "E_n": -lb / n, "rate_at_threshold": rt, "limit": lim, "rho_star": star,
                               "on_threshold": abs(rho - star) < 1e-9})
    for name in ("A", "B"):
        for which, check in (("E0", "K5"), ("E1", "K6")):
            cell = {(x["n"], x["rho"]): x for x in ex if x["instance"] == name and x["exponent"] == which}
            star = cell[n_lo, args.rho_grid[0]]["rho_star"]
            free = [r for r in args.rho_grid if abs(r - star) >= 1e-9]
            on = "".join(f"; rho {r:g} on rho*, no pass rule: beta_{n_lo} {10 ** cell[n_lo, r]['log10_beta']:.5g}, "
                         f"beta_{n_hi} {10 ** cell[n_hi, r]['log10_beta']:.5g}"
                         for r in args.rho_grid if abs(r - star) < 1e-9)
            c = [cell[n_lo, r] for r in free if cell[n_lo, r]["neg_ln_beta"] >= 100]
            v = [abs(x["E_n"] / x["rate_at_threshold"] - 1) for x in c]
            w = c[int(np.argmax(v))] if c else {}
            add(check, name, f"{which}: |E_{n_lo} / rate at threshold - 1| <= 5 % where -ln beta_{n_lo} >= 100",
                max(v, default=np.nan), 0, 0.05, ok=bool(c) and max(v) <= 0.05,
                note=f"{len(c)} cells" + (f"; worst rho {w['rho']:g}: E {w['E_n']:.5f}, rate "
                                          f"{w['rate_at_threshold']:.5f}, -ln beta {w['neg_ln_beta']:.1f}" if c else ""))
            c = [r for r in free if cell[n_lo, r]["E_n"] > 0.01 and cell[n_hi, r]["E_n"] > 0.01]
            v = [np.divide(abs(cell[n_hi, r]["E_n"] - cell[n_lo, r]["limit"]),
                           abs(cell[n_lo, r]["E_n"] - cell[n_lo, r]["limit"])) for r in c]
            w = c[int(np.argmax(v))] if c else None
            add(check, name, f"{which}: |E_{n_hi} - limit| / |E_{n_lo} - limit| < 1 where both E > 0.01 (largest)",
                max(v, default=np.nan), 1, 0, ok=bool(c) and max(v) < 1,
                note=f"{len(c)} cells" + (f"; worst rho {w:g}: E_{n_lo} {cell[n_lo, w]['E_n']:.5f}, E_{n_hi} "
                                          f"{cell[n_hi, w]['E_n']:.5f}, limit {cell[n_lo, w]['limit']:.5f}" if c else ""))
            c = [r for r in free if r < star]
            v = [cell[n_hi, r]["log10_beta"] - cell[n_lo, r]["log10_beta"] for r in c]
            w = c[int(np.argmax(v))] if c else None
            add(check, name, f"{which}: log10 beta_{n_hi} - log10 beta_{n_lo} < 0 at grid rho < rho* {star:.4f} "
                             f"(largest)", max(v, default=np.nan), 0, 0, ok=bool(c) and max(v) < 0,
                note=f"{len(c)} cells" + (f"; tightest rho {w:g}: beta {10 ** cell[n_lo, w]['log10_beta']:.5g} -> "
                                          f"{10 ** cell[n_hi, w]['log10_beta']:.5g}" if c else "") + on)
            c = [r for r in free if r > star]
            v = [10 ** cell[n_hi, r]["log10_beta"] for r in c]
            w = c[int(np.argmin(v))] if c else None
            add(check, name, f"{which}: beta_{n_hi} >= 0.999 at grid rho > rho* {star:.4f} (smallest)",
                min(v, default=np.nan), 0.999, 0, ok=bool(c) and min(v) >= 0.999,
                note=f"{len(c)} cells" + (f"; smallest at rho {w:g}" if c else "") + on)

    # K7 Claim 3
    rp, rp_rows = {}, []
    for name, L in laws.items():
        out = [lp_leak(L, lam) for lam in lams]
        v = rp[name] = np.array([o[0] for o in out])
        qg = L["q"] - L["gamma"]
        add("K7", name, "rho_priv(0) = K2 LP value, R = Sigma", v[0], lp0[name][0], LP_TOL)
        up = float(np.max(np.diff(v)))
        add("K7", name, "rho_priv does not increase with lambda (largest step up)", up, 0, LP_TOL, ok=up <= LP_TOL)
        add("K7", name, "rho_priv(1) = q - gamma", v[-1], qg, LP_TOL,
            note="" if L["k"] else "not in main.tex: hand derivation of 2026-11-12")
        viol = float(max(np.max(qg - v), np.max(v - v[0])))
        add("K7", name, "q - gamma <= rho_priv(lambda) <= rho_priv(0) at every lambda (largest violation)", viol, 0,
            LP_TOL, ok=viol <= LP_TOL)
        rp_rows += [{"instance": name, "lambda": lam, "rho_priv": x, "status": st, "q_minus_gamma": qg}
                    for lam, (x, st) in zip(lams, out)]
    closed0 = {name: (L["m"] - 1) / L["m"] if L["k"] else 1 - L["PY"].max() for name, L in laws.items()}
    ends = [(0.0, closed0[name]) for name in laws] + [(1.0, L["q"] - L["gamma"]) for L in laws.values()]

    write_csv(res_dir / "checks.csv", rows)
    write_csv(res_dir / "exponents.csv", ex)
    write_csv(res_dir / "rho_priv.csv", rp_rows)
    fig_path = fig_dir / "rho_priv_vs_lambda.png"
    plot(lams, rp, ends, fig_path)
    copies = [Path("results/figures") / f"{SLUG}-rho_priv_vs_lambda.png", Path("figures/rho_priv_vs_lambda.png")]
    for dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fig_path, dst)

    counts = {c: [sum(x["pass"] for x in rows if x["check"] == c), sum(x["check"] == c for x in rows)]
              for c in sorted({x["check"] for x in rows})}
    at = lambda name: {f"{lam:g}": float(rp[name][np.isclose(lams, lam)][0]) for lam in LAM_REPORT}
    wall = time.perf_counter() - t_start
    diag = {"instances": {name: {"m": L["m"], "k": L["k"], "gamma": L["gamma"], "delta": L["delta"],
                                 "p": L["p"].tolist(), "P_Y": L["PY"].tolist(), "q": L["q"],
                                 "r": qr[name][1] if name in qr else None, "E0_prop1": e0[name],
                                 "lp_blind_R_Sigma": lp0[name][0], "closed_form_lambda0": closed0[name],
                                 "rho_priv": at(name)} for name, L in laws.items()},
            "k4_worst": k4w, "check_counts_pass_total": counts, "failing_rows": [x for x in rows if not x["pass"]],
            "run": {"seed": args.seed, "alpha": args.alpha, "n_list": args.n_list, "rho_grid": args.rho_grid,
                    "lambda_step": args.lambda_step, "wall_time_s": wall}}
    with open(res_dir / "diagnostic.json", "w") as f:
        json.dump(diag, f, indent=2, default=float)

    for x in rows:
        print(f"{x['check']} {x['instance']:<10} {'PASS' if x['pass'] else 'FAIL'}  {x['statement']}: value "
              f"{x['value']:.12g}, reference {x['reference']:.12g}, diff {x['diff']:.3g}"
              + (f"  [{x['note']}]" if x["note"] else ""))
    print("pass/total per check:", json.dumps(counts))
    print("rho_priv at lambda", LAM_REPORT)
    for name in laws:
        print(f"  {name}: " + ", ".join(f"{v:.6f}" for v in at(name).values())
              + f"  (closed form at 0: {closed0[name]:.6f}, q - gamma: {laws[name]['q'] - laws[name]['gamma']:.6f})")
    print(f"wrote {res_dir}/{{checks,exponents,rho_priv}}.csv, diagnostic.json; {fig_path} -> "
          + ", ".join(map(str, copies)))
    print(f"wall time {wall:.1f} s")


if __name__ == "__main__":
    main()
