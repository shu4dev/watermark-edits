#!/usr/bin/env python
"""Claim 1's variational program for E(rho) (main.tex section 3.1) on the five instances of the report checks.

V(rho) = min over Q(G, y, z) >= 0, G over the lists with mu(G) > 0, of f(Q) = sum Q ln[Q Q_Y / (P_GY Q_YZ)] with
0 ln 0 = 0, which equals D(Q_GY || P_GY) + I_Q(G; Z | Y), subject to sum Q = 1, Q_GZ(G, z) = mu(G) Q_Z(z) for every z
and every list but the last (implied by the others), Q(Y != Z) <= rho, and R = Sigma; at rho = 0 only the z = y
variables exist. Log-barrier Newton method with the exact Hessian: barrier -sum ln Q - ln(rho - Q(Y != Z)) (second term
only when rho > 0), equalities through the KKT system, backtracking line search, t raised tenfold from 1 until
(number of inequalities)/t <= 1e-9; a centring has converged when the Newton decrement of f + barrier/t is <= 1e-10.
Three strictly feasible starts mu x s x W with W = (1 - eps) I + eps / (m - 1) off the diagonal:
(s, eps) = (p, 0.999 rho), (Dirichlet(1), rho / 2), (Dirichlet(1), rho / 4).
V(rho) is the lowest value among the starts whose constraint violation is at most 1e-9.
K8a all five: D(Q_GYZ || P_GY Q_Z|Y) = D(Q_GY || P_GY) + I_Q(G; Z | Y) = D(Q_Y || P_Y) + sum Q_YZ D(Q_G|yz || P_G|y),
    and the solver's f equals the first, on 1000 Dirichlet(1) Q, to 1e-12.
K8b all five: V(0) = E(0) of Prop. 1, to 1e-6.
K8c A and B: V(rho) = E_0(rho) of section 3.3 (wmsim.exponent.rate) at every grid rho < (m - 1)/m, to 1e-6.
K8d all five, no solver: Q = P_GY x (every token overwritten with argmax P_Y) satisfies the constraints at every grid
    rho >= rho* and has f = 0, to 1e-12; rho* = (m - 1)/m on A and B, 1 - max_z P_Y(z) on C1-C3.
K8e all five: V does not increase with rho, its second differences are >= -1e-6, V > 1e-6 at grid rho <= rho* - 0.05.
K8f all five: V(rho) >= d(gamma + rho || q) - 1e-6 at every grid rho < q - gamma (section 3.5's bound).
K8g every solved cell: the three starts agree within 1e-7.
K8h every solved cell: the last centring step converged and the gap bound is <= 1e-8, for all three starts.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from wmsim.claim1 import ID_TOL, forms, objective, program, solve
from wmsim.exponent import kl_bern, rate
from wmsim.singlepos import LABELS, instances
from wmsim.util import write_csv

SLUG = "2026-11-12-claim1-program"
SOLVE_TOL, START_TOL, FEAS_TOL, GAP_TOL = 1e-6, 1e-7, 1e-9, 1e-8
RHO_REPORT = (0.0, 0.25, 0.5, 0.75)


def plot(rhos, V, rings, path):
    """One panel: the solved E(rho) against rho, one curve per instance; main.tex's closed forms as black rings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, gridc, axc, surf = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")  # categorical slots 1-5, fixed order
    fig, ax = plt.subplots(figsize=(8.2, 4.2), facecolor=surf)
    ax.set_facecolor(surf)
    for (name, y), col, mk in zip(V.items(), colors, "osD^v"):  # marker shapes as well: curves share points
        ax.plot(rhos, y, color=col, lw=2, marker=mk, ms=4.5, mec=surf, mew=1.2, label=LABELS[name])
    ax.plot(*zip(*rings), ls="none", marker="o", ms=10, mfc="none", mec=ink, mew=1.3,
            label="closed form: $E_0$ (A, B), Prop. 1 at ρ = 0, 0 from ρ*")
    ax.set_xlim(rhos.min() - 0.03, rhos.max() + 0.03)
    ax.set_ylim(-0.015, None)  # room for the rings at zero
    ax.grid(color=gridc, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(axc)
    ax.tick_params(colors=ink2, labelsize=10)
    ax.set_xlabel("edit fraction ρ", color=ink2, fontsize=10)
    ax.set_ylabel(r"$E(\rho)$ from Claim 1's program, nat/token", color=ink2, fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=10, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surf)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rho-grid", type=float, nargs="+", default=[round(0.05 * i, 2) for i in range(20)])
    ap.add_argument("--seed", type=int, default=0, help="C3's p (first draw), K8a's Q, the Dirichlet starts")
    ap.add_argument("--out", default="data", help="writes <out>/results/<slug>/ and <out>/figures/<slug>/")
    args = ap.parse_args()
    t_start = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    rhos = np.array(sorted(args.rho_grid))
    res_dir, fig_dir = Path(args.out) / "results" / SLUG, Path(args.out) / "figures" / SLUG
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    laws = instances(rng)
    rows = []

    def add(check, inst, statement, value, ref, tol, ok=None, note=""):
        value, ref = float(value), float(ref)
        rows.append({"check": check, "instance": inst, "statement": statement, "value": value, "reference": ref,
                     "diff": value - ref, "tol": tol, "pass": bool(abs(value - ref) <= tol if ok is None else ok),
                     "note": note})

    # closed forms: Prop. 1's E(0), rho*, section 3.3's E_0 (A and B), section 3.5's d(gamma + rho || q)
    info = {}
    for name, L in laws.items():
        g, m = L["gamma"], L["m"]
        star = (m - 1) / m if L["k"] else 1 - L["PY"].max()
        E0 = np.full(len(rhos), np.nan)
        if L["k"]:
            ed = np.exp(L["delta"])
            q = g * ed / (1 - g + g * ed)
            r = (L["k"] - q) / (m - 1)
            E0 = np.array([rate(np.array([0, 0, 1 - rho, rho]), g, g, q, r) for rho in rhos])
        info[name] = {"E0_prop1": float(L["mu"] @ np.log(L["N"]) - L["delta"] * g), "rho_star": float(star),
                      "E0": E0, "kl": np.array([kl_bern(g + rho, L["q"]) if rho < L["q"] - g else np.nan
                                                for rho in rhos])}

    # K8a the three forms
    for name, L in laws.items():
        nG, m = len(L["mu"]), L["m"]
        keep, logPGY = np.ones((nG, m, m), bool), np.log(L["PGY"])
        F = np.array([(*forms(x.reshape(nG, m, m), L["PGY"]), objective(x, keep, logPGY, L["PGY"] / L["PY"])[0])
                      for x in rng.dirichlet(np.ones(nG * m * m), 1000)])
        for j, stmt in ((1, "D(Q_GY||P_GY) + I_Q(G;Z|Y)"), (2, "D(Q_Y||P_Y) + sum_yz Q_YZ D(Q_G|yz||P_G|y)"),
                        (3, "solver's f = sum Q ln[Q Q_Y / (P_GY Q_YZ)]")):
            w = int(np.argmax(np.abs(F[:, j] - F[:, 0])))
            add("K8a", name, f"{stmt} = D(Q_GYZ||P_GY Q_Z|Y) on 1000 Dirichlet(1) Q (worst shown)", F[w, j], F[w, 0],
                ID_TOL)

    # the program at every grid rho, three starts
    cells, solver = [], {}
    for name, L in laws.items():
        t = time.perf_counter()
        for i, rho in enumerate(rhos):
            s1, s2 = rng.dirichlet(np.ones(L["m"]), 2)
            # start 0 at 0.999 rho rather than rho: the barrier needs the budget inequality strict
            runs = solve(L, rho, [(L["p"], 0.999 * rho), (s1, rho / 2), (s2, rho / 4)])
            ok = [x for x in runs if x["violation"] <= FEAS_TOL]
            best = min(ok, key=lambda x: x["value"]) if ok else runs[0]
            st, E0 = info[name]["rho_star"], info[name]["E0"][i]
            closed = (info[name]["E0_prop1"] if rho == 0 else 0.0 if rho >= st - 1e-9 else E0)
            cells.append({"instance": name, "rho": float(rho), "V": best["value"] if ok else np.nan,
                          "violation": best["violation"], "starts_feasible": len(ok),
                          "start_spread": float(np.ptp([x["value"] for x in ok])) if ok else np.nan,
                          **{f"V_start{j}": x["value"] for j, x in enumerate(runs)},
                          "gap_bound": max(x["gap_bound"] for x in runs),
                          "last_decrement": max(x["last_decrement"] for x in runs),
                          "starts_last_centring_converged": sum(x["last_centring_converged"] for x in runs),
                          "newton_steps": max(x["newton_steps"] for x in runs),
                          "earlier_centrings_not_converged": sum(x["earlier_centrings_not_converged"] for x in runs),
                          "max_violation_starts": max(x["violation"] for x in runs),
                          "closed_form": closed, "E0_sec33": E0, "kl_bound_sec35": info[name]["kl"][i],
                          "rho_star": st, "seconds": sum(x["seconds"] for x in runs)})
        solver[name] = {"seconds": time.perf_counter() - t}
    V = {name: np.array([c["V"] for c in cells if c["instance"] == name]) for name in laws}
    cell = {(c["instance"], c["rho"]): c for c in cells}

    for name, L in laws.items():
        st, v = info[name]["rho_star"], V[name]
        # K8b
        add("K8b", name, "V(0) = E(0) of Prop. 1", cell.get((name, 0.0), {"V": np.nan})["V"], info[name]["E0_prop1"],
            SOLVE_TOL)
        # K8c
        if L["k"]:
            for i, rho in enumerate(rhos):
                if rho < st - 1e-9:
                    add("K8c", name, f"V = E_0 at rho {rho:g}", v[i], info[name]["E0"][i], SOLVE_TOL)
        # K8d
        m, nG = L["m"], len(L["mu"])
        Qo = (L["PGY"][:, :, None] * (np.arange(m) == np.argmax(L["PY"]))).ravel()
        above = rhos[rhos >= st - 1e-9]
        vo = [program(L, rho)[4](Qo) for rho in above]
        add("K8d", name, f"overwrite point satisfies the constraints at every grid rho >= rho* {st:.4f} (largest "
                         f"violation)", max(vo, default=np.nan), 0, ID_TOL, ok=bool(vo) and max(vo) <= ID_TOL,
            note=f"{len(vo)} grid rho")
        logPGY = np.log(L["PGY"])
        add("K8d", name, "f(overwrite point) = 0", objective(Qo, np.ones((nG, m, m), bool), logPGY,
                                                             L["PGY"] / L["PY"])[0], 0, ID_TOL)
        # K8e
        up = float(np.max(np.diff(v)))
        add("K8e", name, "V does not increase with rho (largest step up)", up, 0, SOLVE_TOL, ok=up <= SOLVE_TOL,
            note=f"at rho {rhos[1 + int(np.argmax(np.diff(v)))]:g}")
        d2 = np.diff(v, 2)
        add("K8e", name, "second differences of V >= -1e-6 (smallest)", d2.min(), 0, SOLVE_TOL,
            ok=d2.min() >= -SOLVE_TOL, note=f"at rho {rhos[1 + int(np.argmin(d2))]:g}")
        low = rhos <= st - 0.05 + 1e-9
        add("K8e", name, f"V > 1e-6 at every grid rho <= rho* - 0.05 = {st - 0.05:.4f} (smallest)", v[low].min(),
            SOLVE_TOL, 0, ok=v[low].min() > SOLVE_TOL, note=f"{low.sum()} grid rho; no rule at "
                                                             + (", ".join(f"{r:g}" for r in rhos[~low & (rhos < st)])
                                                                or "none"))
        # K8f
        for i, rho in enumerate(rhos):
            if rho < L["q"] - L["gamma"]:
                kl = info[name]["kl"][i]
                add("K8f", name, f"V >= d(gamma + rho || q) - 1e-6 at rho {rho:g}", v[i], kl, SOLVE_TOL,
                    ok=v[i] >= kl - SOLVE_TOL)
        # K8g
        for rho in rhos:
            c = cell[name, float(rho)]
            add("K8g", name, f"three starts agree at rho {rho:g} (largest - smallest)", c["start_spread"], 0, START_TOL,
                ok=c["starts_feasible"] == 3 and c["start_spread"] <= START_TOL,
                note=f"{c['starts_feasible']} of 3 starts feasible")
        # K8h
        for rho in rhos:
            c = cell[name, float(rho)]
            conv = c["starts_last_centring_converged"]
            add("K8h", name, f"last centring converged and gap bound <= 1e-8 at rho {rho:g} (largest gap bound)",
                c["gap_bound"], 0, GAP_TOL, ok=conv == 3 and c["gap_bound"] <= GAP_TOL,
                note=f"last centring converged in {conv} of 3 starts; largest last decrement "
                     f"{c['last_decrement']:.2g}")
        cs = [c for c in cells if c["instance"] == name]
        solver[name].update({
            "largest_gap_bound": max(c["gap_bound"] for c in cs),
            "largest_last_decrement": max(c["last_decrement"] for c in cs),
            "cells_last_centring_not_converged": sum(c["starts_last_centring_converged"] < 3 for c in cs),
            "cells_with_earlier_centring_not_converged": sum(c["earlier_centrings_not_converged"] > 0 for c in cs),
            "largest_violation_V": max(c["violation"] for c in cs),
            "largest_violation_any_start": max(c["max_violation_starts"] for c in cs),
            "largest_start_spread": max(c["start_spread"] for c in cs),
            "max_newton_steps": max(c["newton_steps"] for c in cs)})

    write_csv(res_dir / "checks.csv", rows)
    write_csv(res_dir / "claim1.csv", cells)
    rings = [(c["rho"], c["closed_form"]) for c in cells if np.isfinite(c["closed_form"])]
    fig_path = fig_dir / "claim1_exponent.png"
    plot(rhos, V, rings, fig_path)
    copies = [Path("results/figures") / f"{SLUG}-claim1_exponent.png", Path("figures/claim1_exponent.png")]
    for dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fig_path, dst)

    counts = {c: [sum(x["pass"] for x in rows if x["check"] == c), sum(x["check"] == c for x in rows)]
              for c in sorted({x["check"] for x in rows})}
    at = lambda name, key: {f"{r:g}": float(cell[name, r][key]) for r in RHO_REPORT if (name, r) in cell}
    worst = {name: float(np.nanmax(np.abs(V[name] - info[name]["E0"]))) for name in ("A", "B")}
    wall = time.perf_counter() - t_start
    diag = {"instances": {name: {"m": L["m"], "k": L["k"], "gamma": L["gamma"], "delta": L["delta"],
                                 "p": L["p"].tolist(), "P_Y": L["PY"].tolist(), "q": L["q"],
                                 "rho_star": info[name]["rho_star"], "E0_prop1": info[name]["E0_prop1"],
                                 "V": at(name, "V"), "E0_sec33": at(name, "E0_sec33") if L["k"] else None,
                                 "worst_abs_V_minus_E0_grid": worst.get(name), "solver": solver[name]}
                          for name, L in laws.items()},
            "check_counts_pass_total": counts, "failing_rows": [x for x in rows if not x["pass"]],
            "run": {"seed": args.seed, "rho_grid": rhos.tolist(), "wall_time_s": wall}}
    with open(res_dir / "diagnostic.json", "w") as f:
        json.dump(diag, f, indent=2, default=float)

    for x in rows:
        print(f"{x['check']} {x['instance']:<3} {'PASS' if x['pass'] else 'FAIL'}  {x['statement']}: value "
              f"{x['value']:.12g}, reference {x['reference']:.12g}, diff {x['diff']:.3g}"
              + (f"  [{x['note']}]" if x["note"] else ""))
    print("pass/total per check:", json.dumps(counts))
    print("V at rho", RHO_REPORT, "(E_0 of section 3.3 in brackets on A and B)")
    for name in laws:
        print(f"  {name}: " + ", ".join(f"{v:.8f}" + (f" [{e:.8f}]" if laws[name]["k"] else "")
                                       for v, e in zip(at(name, "V").values(), at(name, "E0_sec33").values()))
              + (f"; worst |V - E_0| over the grid {worst[name]:.3g}" if name in worst else ""))
    for name in laws:
        print(f"  solver {name}: {json.dumps(solver[name])}")
    print(f"wrote {res_dir}/{{checks,claim1}}.csv, diagnostic.json; {fig_path} -> " + ", ".join(map(str, copies)))
    print(f"wall time {wall:.1f} s")


if __name__ == "__main__":
    main()
