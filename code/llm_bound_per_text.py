#!/usr/bin/env python
"""Section 4's bound with each text's own average q (plans/2026-11-17-llm-bound-per-text.md).

CPU-only re-analysis of the report-scheme run's saved arrays: no model and no torch, so wmsim.llm (which imports torch)
is not used. Per text: S = green emitted tokens, qbar = mean of q_t over its T positions, G = green positions with a red
candidate other than y_t. The key holder removes one green per edit, so its green count at budget j is S - min(j, G).
Statement under test, derived by the session and not checked by the theory side: for a cut-off c and s/T < c,
P(S <= s and qbar >= c) <= e^{-T d(s/T || c)}. A text is missed at budget j only if S <= tau - 1 + j, so the miss rate
is at most F(c) + e^{-T d((tau - 1 + j)/T || c)} for every c, F(c) = share of texts with qbar < c, the second term 1
where (tau - 1 + j)/T >= c. The combined ceiling is the minimum over c = 0.260, 0.261, ..., 0.700; since c is chosen on
the same texts, the ceiling is also given at the fixed cut-offs 0.50, 0.55 and 0.60.
Check 1: these missed counts equal the key holder's rows of the parent's miss_rates.csv (the script stops if not).
Check 2: at each fixed c and grid rho, the share of all texts with qbar >= c that are missed has no 95 % Clopper-Pearson
lower limit above e^{-T d((tau - 1 + j)/T || c)}. Check 3: the measured miss rate's lower limit lies above the combined
ceiling at no grid rho.
Writes <out>/results/<slug>/{ceiling,joint,checks}.csv and summary.json, and <out>/figures/<slug>/llm_bound_per_text.png,
copied to results/figures/ and figures/.
"""
import argparse
import csv
import json
import math
import platform
import shutil
import sys
import textwrap
import time
from pathlib import Path

import matplotlib
import numpy as np
import scipy
from scipy.stats import beta

from wmsim.exponent import kl_bern
from wmsim.util import write_csv

SLUG, FIG = "2026-11-17-llm-bound-per-text", "llm_bound_per_text"
CUTS = np.round(np.arange(260, 701) / 1000, 3)  # cut-off grid 0.260 .. 0.700
FIXED = (0.50, 0.55, 0.60)
RHO_FIG = 0.4


def clopper_pearson(k, N):
    """95 % Clopper-Pearson interval for k successes in N trials; same formula as wmsim.llm.clopper_pearson."""
    return (float(beta.ppf(0.025, k, N - k + 1)) if k else 0.0), (float(beta.ppf(0.975, k + 1, N - k)) if k < N else 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="data/results/2026-11-16-llm-blind-attack/report",
                    help="parent run's outputs: positions.npz, summary.json, miss_rates.csv, ceilings.csv")
    ap.add_argument("--out", default="data", help="writes <out>/results/<slug>/ and <out>/figures/<slug>/")
    args = ap.parse_args()
    t0 = time.perf_counter()
    run = Path(args.run_dir)
    res_dir, fig_dir = Path(args.out) / "results" / SLUG, Path(args.out) / "figures" / SLUG
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(run / "positions.npz")
    gm, cand, y, q = d["green"], d["cand"], d["y"], d["q"].astype(np.float64)
    with open(run / "summary.json") as f:
        parent = json.load(f)
    with open(run / "miss_rates.csv") as f:
        kh = [r for r in csv.DictReader(f) if r["attacker"] == "key holder"]
    with open(run / "ceilings.csv") as f:
        pc = list(csv.DictReader(f))
    tau, (N, T) = int(parent["tau_used"]), y.shape
    rhos, budgets = np.array([float(r["rho"]) for r in kh]), np.array([int(r["budget"]) for r in kh])
    assert [float(r["rho"]) for r in pc] == list(rhos)
    ceil_qavg = np.array([float(r["ceiling_q_avg"]) for r in pc])

    S, qbar = gm[..., 0].sum(1), q.mean(1)
    red = ~gm[..., 1:] & (cand != y[..., None])  # as key_holder() in code/llm_edits.py
    G = (gm[..., 0] & red.any(-1)).sum(1)
    used = np.minimum(budgets[:, None], G)  # (rho, text): edits the key holder makes
    missed = S - used < tau
    k = missed.sum(1)
    k_parent = np.array([int(r["missed"]) for r in kh])
    d_edits = float(np.abs(used.mean(1) - np.array([float(r["edits_mean"]) for r in kh])).max())
    if not np.array_equal(k, k_parent):
        sys.exit("Check 1 FAIL, missed counts differ from the parent's key-holder rows: " + "; ".join(
            f"rho {rhos[i]:g} budget {budgets[i]}: here {k[i]}, parent {k_parent[i]}" for i in np.flatnonzero(k != k_parent)))

    tail = lambda j, c: math.exp(-T * kl_bern((tau - 1 + j) / T, c))  # 1 where (tau - 1 + j)/T >= c
    F = (qbar < CUTS[:, None]).mean(1)
    total = F + np.array([[tail(j, c) for c in CUTS] for j in budgets])  # (rho, cut-off)
    best = total.argmin(1)  # first minimum: the smallest such cut-off
    ceil, c_best = total[np.arange(len(rhos)), best], CUTS[best]
    fixed = {c: (qbar < c).mean() + np.array([tail(j, c) for j in budgets]) for c in FIXED}
    assert all((ceil <= v + 1e-12).all() for v in fixed.values())  # the fixed cut-offs lie on the grid
    lo, hi = np.array([clopper_pearson(int(x), N) for x in k]).T

    ceil_rows = [{"rho": r, "budget": j, "missed": int(m), "miss_rate": m / N, "ci_lo": a, "ci_hi": b,
                  "combined_ceiling": cb, "cut_off": c, "F_at_cut_off": F[i], "tail_at_cut_off": cb - F[i],
                  **{f"ceiling_c{c0:.2f}": v[n] for c0, v in fixed.items()}, "parent_ceiling_q_avg": pq,
                  "fails": bool(a > cb)}
                 for n, (r, j, m, a, b, cb, c, i, pq) in enumerate(zip(rhos, budgets, k, lo, hi, ceil, c_best, best,
                                                                          ceil_qavg))]
    joint = []
    for c in FIXED:
        above = qbar >= c
        for r, j, m in zip(rhos, budgets, missed):
            kk = int((m & above).sum())
            a, b = clopper_pearson(kk, N)
            bound = tail(j, c)
            joint.append({"cut_off": c, "rho": r, "budget": j, "texts_at_or_above": int(above.sum()),
                          "missed_at_or_above": kk, "share_of_all_texts": kk / N, "ci_lo": a, "ci_hi": b,
                          "bound": bound, "lo_over_bound": a / bound, "fails": bool(a > bound)})
    # worst cell per cut-off among those with a bound below 1 (a bound of 1 cannot fail)
    worst = {c: max((x for x in joint if x["cut_off"] == c and x["bound"] < 1),
                    key=lambda x: (x["lo_over_bound"], x["share_of_all_texts"] / x["bound"])) for c in FIXED}
    n_joint_fail, n3_fail = sum(x["fails"] for x in joint), sum(x["fails"] for x in ceil_rows)
    # tightest point of check 3 where the ceiling says something: smallest ceiling / measured rate with ceiling < 1
    margin3 = min((x for x in ceil_rows if x["missed"] and x["combined_ceiling"] < 1),
                  key=lambda x: x["combined_ceiling"] / x["miss_rate"])
    checks = [
        {"check": "1 key-holder rows", "statement": "missed counts from S - min(j, G) equal the key holder's rows of "
                                                    "the parent's miss_rates.csv at every grid rho",
         "value": int((k != k_parent).sum()), "reference": 0, "pass": True,
         "note": f"{len(rhos)} grid rho; largest |mean edits - parent edits_mean| {d_edits:.3g}"},
        {"check": "2 joint", "statement": "at c = 0.50, 0.55, 0.60 and every grid rho, the 95 % lower limit of the "
                                          "share of texts with qbar >= c that are missed is not above "
                                          "e^{-T d((tau - 1 + j)/T || c)} (failing cells)",
         "value": n_joint_fail, "reference": 0, "pass": n_joint_fail == 0,
         "note": "; ".join(f"worst at c {c:.2f}: rho {w['rho']:g}, {w['missed_at_or_above']} of {N}, lower limit "
                           f"{w['ci_lo']:.3g} against {w['bound']:.3g}" if w["missed_at_or_above"] else
                           f"at c {c:.2f} no text with qbar >= c is missed where the bound is below 1"
                           for c, w in worst.items())},
        {"check": "3 combined", "statement": "the measured miss rate's 95 % lower limit is not above the combined "
                                             "ceiling at any grid rho (failing rho)",
         "value": n3_fail, "reference": 0, "pass": n3_fail == 0,
         "note": f"tightest where the ceiling is below 1: rho {margin3['rho']:g}, ceiling "
                 f"{margin3['combined_ceiling']:.3g} against {margin3['miss_rate']:.3g} "
                 f"[{margin3['ci_lo']:.3g}, {margin3['ci_hi']:.3g}]"}]

    write_csv(res_dir / "ceiling.csv", ceil_rows)
    write_csv(res_dir / "joint.csv", joint)
    write_csv(res_dir / "checks.csv", checks)

    # figure, in the style of wmsim.llm.figure (not imported: that module needs torch)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, gridc, axc, surf, green = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb", "#1baf7a"
    run_args, qavg = parent["run"], parent["q_avg"]
    sel = rhos <= RHO_FIG + 1e-9
    pos = sel & (k > 0)  # a log axis cannot show a miss rate of 0
    shown = sel & (ceil < 1)
    labels = ("key holder, measured: knows every green list, swaps green tokens for the likeliest red one",
              "combined ceiling: share of texts whose own average q is below a cut-off, plus a Chernoff tail for "
              "the texts at or above it, at the cut-off that gives the smallest sum; derived in this round from the "
              "report's Chernoff argument, not yet checked by the theory side",
              f"ceiling of the parent run: Section 4's formula with the average q of all texts ({qavg:.3f}) in place "
              f"of the smallest q; not proven in the report")
    fig, ax = plt.subplots(figsize=(8.6, 9.4), facecolor=surf)
    ax.set_facecolor(surf)
    ax.fill_between(rhos[pos], lo[pos], hi[pos], color=green, alpha=0.15, lw=0)
    ax.plot(rhos[pos], k[pos] / N, color=green, lw=2, marker="D", ms=6, mec=surf, mew=1.0,
            label=textwrap.fill(labels[0], 90))
    ax.plot(rhos[sel], ceil[sel], color=green, lw=2, ls="--", label=textwrap.fill(labels[1], 90))
    ax.plot(rhos[sel], ceil_qavg[sel], color=green, lw=2, ls=":", label=textwrap.fill(labels[2], 90))
    ax.set_yscale("log")
    ax.set_xlim(-0.01, RHO_FIG + 0.01)
    ax.set_ylim(5e-9, 2)
    ax.grid(color=gridc, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(axc)
    ax.tick_params(colors=ink2, labelsize=10)
    ax.set_xlabel(f"edit fraction (share of the {T} tokens replaced)", color=ink2, fontsize=11)
    ax.set_ylabel(f"miss rate, log scale\n(share of the {N} watermarked texts not flagged)", color=ink2, fontsize=11)
    ax.set_title("Report's scheme: the key holder against a ceiling from each text's own average q", color=ink,
                 fontsize=12)
    vac, zero = rhos[sel & (ceil >= 1)], sel & (k == 0)
    note = (f"OPT-1.3B continues {N} news prompts by {T} tokens (T). Green share of the vocabulary (γ) "
            f"{run_args['gamma']:g}; bias added to green tokens (δ) {run_args['delta']:g}. The detector flags a text "
            f"with at least {tau} green tokens (τ), which keeps false alarms on unwatermarked text at most "
            f"{run_args['alpha']:g} (α). q: chance that a watermarked token is green, from the model's probabilities; "
            f"a text's average q is its mean over the {T} positions. Cut-off (c): a value of the average q, tried from "
            f"{CUTS[0]:.2f} to {CUTS[-1]:.2f}. Texts below it count as missed; the rest are bounded by the tail "
            + "exp(−T·d((τ − 1 + edits)/T ‖ c))".replace(" ", "\xa0")  # keep the formula on one line
            + f", d the relative entropy between two coins of these biases. The cut-offs behind the dashed line are "
            f"{', '.join(f'{c:.3f}' for c in c_best[shown])} at edit fractions {rhos[shown][0]:g} to "
            f"{rhos[shown][-1]:g}" + (f"; from {vac[0]:g} on no cut-off brings it below 1" if len(vac) else "")
            + ". Shaded band: 95 % Clopper-Pearson interval of the measured miss rate."
            + (f" No text is missed at edit fraction {', '.join(f'{r:g}' for r in rhos[zero])} (interval up to "
               f"{hi[zero].max():.4f}), which a log scale cannot show." if zero.any() else ""))
    fig.tight_layout(rect=(0, 0.393, 1, 1))  # room below for the legend and the note
    x0 = ax.get_position().x0
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(x0 - 0.01, 0.383), frameon=False,
               fontsize=10.5, labelcolor=ink)
    fig.text(x0, 0.015, textwrap.fill(note, 92, break_on_hyphens=False), ha="left", va="bottom", color=ink2,
             fontsize=10.5)
    fig_path = fig_dir / f"{FIG}.png"
    fig.savefig(fig_path, dpi=150, facecolor=surf)
    plt.close(fig)
    copies = [Path("results/figures") / f"{SLUG}-{FIG}.png", Path("figures") / f"{FIG}.png"]
    for dst in copies:
        shutil.copy(fig_path, dst)

    wall = time.perf_counter() - t0
    summary = {"statement": "for a cut-off c and s/T < c, P(S <= s and qbar >= c) <= e^{-T d(s/T || c)}; derived by "
                            "the session, not checked by the theory side",
               "tau": tau, "T": T, "texts": N,
               "qbar": {"min": float(qbar.min()), "max": float(qbar.max()), "mean": float(qbar.mean())},
               "share_below_fixed_cut_offs": {f"{c:.2f}": float((qbar < c).mean()) for c in FIXED},
               "green_count_S": {"mean": float(S.mean()), "sd_ddof1": float(S.std(ddof=1))},
               "G_equals_S_texts": int((G == S).sum()),
               "checks": {c["check"]: c["pass"] for c in checks},
               "worst_joint_cell": {f"{c:.2f}": w for c, w in worst.items()},
               "run": {**vars(args), "python": platform.python_version(), "numpy": np.__version__,
                       "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                       "platform": platform.platform(), "wall_time_s": wall}}
    with open(res_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"tau {tau} of T {T}, {N} texts; qbar from {qbar.min():.3f} to {qbar.max():.3f}, mean {qbar.mean():.4f}")
    print("rho   j  missed  rate [95 % CI]           combined  cut-off  " + "  ".join(f"c {c:.2f}" for c in FIXED)
          + "   parent q_avg")
    for x in ceil_rows:
        print(f"{x['rho']:<4g} {x['budget']:>3} {x['missed']:>6}  {x['miss_rate']:.3f} [{x['ci_lo']:.2e}, "
              f"{x['ci_hi']:.2e}]  {x['combined_ceiling']:.3e}  {x['cut_off']:.3f}  "
              + "  ".join(f"{x[f'ceiling_c{c:.2f}']:.3e}" for c in FIXED) + f"  {x['parent_ceiling_q_avg']:.3e}"
              + ("  FAIL" if x["fails"] else ""))
    for c in checks:
        print(f"check {c['check']:<18} {'PASS' if c['pass'] else 'FAIL'}  {c['statement']}: {c['value']} "
              f"(reference {c['reference']})  [{c['note']}]")
    print(f"wrote {res_dir}/ and {fig_path} -> " + ", ".join(map(str, copies)) + f"; wall time {wall:.1f} s")


if __name__ == "__main__":
    main()
