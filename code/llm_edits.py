#!/usr/bin/env python
"""Key-blind edits on OPT-1.3B text under the report's scheme (plans/2026-11-16-llm-blind-attack.md).

Scheme: the green list at generated position t of a text with key k is {v : u_v < gamma}, u the uniforms of numpy's
default_rng([k, t]): an independent Bernoulli(gamma) list at every position. Generation adds delta to the green
logits (wmsim.llm.generate: exactly n tokens, temperature 1, EOS suppressed).
Three attackers with budget floor(rho n), each replacing y_t only by one of the K = 20 likeliest tokens under p_t
other than y_t: random and key-blind (wmsim.llm), and the key holder, which replaces green tokens, chosen uniformly,
by the likeliest red candidate and skips a position with no red candidate. An edit changes only its own position's
colour, so green counts after edits come from the memberships of y_t and the candidates.
Detector: green count >= tau = threshold(T, gamma, alpha) with T = n scored tokens.
Checks: null (20 fresh keys per text), budgets, inequality (2) at every position, and g at M lists against 65 536
lists on the first 20 positions of text 0 (within 0.02).
Writes <out>/results/<slug>/report/{miss_rates,rho_half,ceilings,inequality2,checks}.csv, positions.npz, summary.json
and <out>/figures/<slug>/llm_miss_rate_report.png, copied to results/figures/ and figures/.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import LogitsProcessor

from wmsim import llm
from wmsim.greencount import threshold
from wmsim.util import write_csv

SLUG, FIG = "2026-11-16-llm-blind-attack", "llm_miss_rate_report"
NULL_KEYS, M_CHECK, M_TOL, CHECK_POS = 20, 65_536, 0.02, 20


def greens(key, t, toks, gamma):
    """Whether each token in toks is in the green list at (key, t). Uniforms are drawn only up to the largest token;
    numpy fills them in order, so u_v is the same as in a draw over the whole vocabulary."""
    return np.random.default_rng([key, t]).random(int(np.max(toks)) + 1)[toks] < gamma


class Watermark(LogitsProcessor):
    """Adds delta to the logits of the green list at (key of the row's text, position)."""

    def __init__(self, keys, prompt_len, gamma, delta):
        self.keys, self.prompt_len, self.gamma, self.delta = keys, prompt_len, gamma, delta

    def __call__(self, input_ids, scores):
        t, V = input_ids.shape[1] - self.prompt_len, scores.shape[1]
        G = np.stack([greens(k, t, np.arange(V), self.gamma) for k in self.keys])
        return scores + self.delta * torch.from_numpy(G).to(scores.device, scores.dtype)


def key_holder(rng, gm, cand, y):
    """Green positions with a red candidate in uniform random order, each replaced by its likeliest red candidate
    (candidates come likeliest first). Returns the ranks and the replacement's column in [y, candidates]."""
    red = ~gm[..., 1:] & (cand != y[..., None])
    return llm.ranks(rng.random(y.shape), gm[..., 0] & red.any(-1)), 1 + red.argmax(-1)


def green_counts(rank, col, gm, j):
    """Green count of each text and its edits used when the positions with rank < j take the column col."""
    edited = rank < j
    return np.where(edited, np.take_along_axis(gm, col[..., None], -1)[..., 0], gm[..., 0]).sum(1), edited.sum(1)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gamma", type=float, default=0.25, help="green share of the vocabulary")
    ap.add_argument("--delta", type=float, default=2.0, help="bias added to green logits")
    ap.add_argument("--alpha", type=float, default=1e-3, help="detector's false-positive rate")
    ap.add_argument("--n", type=int, default=200, help="new tokens per text")
    ap.add_argument("--texts", type=int, default=500)
    ap.add_argument("--rho-grid", type=float, nargs="+", default=[round(0.05 * i, 2) for i in range(21)])
    ap.add_argument("--lists", type=int, default=4096, help="M, the Bernoulli lists of the key-blind posterior")
    ap.add_argument("--seed", type=int, default=0, help="keys, sampling, posterior lists, attackers' coins")
    ap.add_argument("--out", default="data", help="writes <out>/results/<slug>/report/ and <out>/figures/<slug>/")
    args = ap.parse_args()
    t_start = last = time.perf_counter()
    timing = {}

    def lap(name):  # seconds since the previous lap
        nonlocal last
        timing[name], last = round(time.perf_counter() - last, 1), time.perf_counter()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    s_np, s_torch, s_lists = (int(s) for s in np.random.SeedSequence(args.seed).generate_state(3))
    rng = np.random.default_rng(s_np)
    torch.manual_seed(s_torch)
    gamma, delta, n, rhos = args.gamma, args.delta, args.n, np.array(args.rho_grid)
    res_dir, fig_dir = Path(args.out) / "results" / SLUG / "report", Path(args.out) / "figures" / SLUG
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tok, model = llm.load_model(device)
    prompts = llm.load_prompts(tok, args.texts)
    keys = rng.integers(2**63, size=len(prompts))
    lap("load_s")
    y = llm.generate(model, prompts, n, lambda rows: Watermark(keys[rows], prompts.shape[1], gamma, delta))
    lap("generate_s")
    V = model.config.vocab_size
    B = list(llm.lists(V, gamma, args.lists, s_lists, device))
    cand, g, q, rhs = llm.position_stats(model, prompts, y, B, gamma, delta)
    lap("posterior_s")
    yn = y.numpy()
    idx = np.concatenate([yn[..., None], cand], -1)  # columns: y_t, then the K candidates
    gm = np.array([[greens(k, s, idx[i, s], gamma) for s in range(n)] for i, k in enumerate(keys)])
    assert (gm[0, 0] == greens(keys[0], 0, np.arange(V), gamma)[idx[0, 0]]).all()  # partial draw = full draw
    lap("memberships_s")

    # null: the watermarked texts under fresh keys are key-independent text
    T = n
    tau, size = threshold(T, gamma, args.alpha)
    null_keys = rng.integers(2**63, size=(len(keys), NULL_KEYS))
    null = np.array([[sum(greens(k, s, yn[i, s], gamma) for s in range(n)) for k in ks]
                     for i, ks in enumerate(null_keys)])
    tau_used, nul = llm.null_check(null.ravel(), tau, args.alpha)
    lap("null_s")

    attacks = {"random": llm.random_attack(rng, cand, yn), "key-blind": llm.blind_attack(g, cand, yn)[:2],
               "key holder": key_holder(rng, gm, cand, yn)}
    budgets = [int(np.floor(r * n + 1e-9)) for r in rhos]
    rows = [llm.miss_row(name, float(r), j, *green_counts(rank, col, gm, j), tau_used)
            for name, (rank, col) in attacks.items() for r, j in zip(rhos, budgets)]
    col_of = lambda name, k: np.array([x[k] for x in rows if x["attacker"] == name])
    rh = {name: llm.rho_half(rhos, col_of(name, "miss_rate")) for name in attacks}
    share = llm.key_share(rh["random"], rh["key-blind"], rh["key holder"])
    qmin, qavg = float(q.min()), float(q.mean())
    ceil = {"q_min": llm.ceiling(rhos, T, tau_used, qmin), "q_avg": llm.ceiling(rhos, T, tau_used, qavg)}
    ineq = llm.ineq2(q, rhs, gamma, float(gm[..., 0].mean()))

    # g at M lists against M_CHECK lists (the first M shared) on the first CHECK_POS positions of text 0
    p0, y0 = llm.base_law(model, prompts[:1], y[:1])[0, :CHECK_POS], y[0, :CHECK_POS].to(device)
    g_m = llm.posterior(p0, y0, B, gamma, delta)[1]
    g_big = llm.posterior(p0, y0, llm.lists(V, gamma, M_CHECK, s_lists, device), gamma, delta)[1]
    dg = (g_m - g_big).abs().cpu().numpy()
    lap("attacks_checks_s")

    gap_stats = [float(ineq[k]) for k in ("gap mean", "gap 25th percentile", "gap median", "gap 75th percentile")]
    over = max(x["edits_max"] - x["budget"] for x in rows)
    checks = [
        {"check": "null", "statement": f"95 % interval of the flag rate on {nul['trials']} key-independent trials "
                                       f"not above alpha", "value": nul["flag_rate"], "reference": args.alpha,
         "pass": bool(nul["ci_lo"] <= args.alpha),
         "note": f"interval [{nul['ci_lo']:.2e}, {nul['ci_hi']:.2e}]; tau {tau} (exact size {size:.2e}) "
                 + ("kept" if tau_used == tau else f"moved to the empirical quantile {tau_used}")},
        {"check": "budget", "statement": "every attack uses at most floor(rho n) edits (largest excess)",
         "value": over, "reference": 0, "pass": all(x["within_budget"] for x in rows), "note": ""},
        {"check": "lists", "statement": f"max |g at M = {args.lists} - g at M = {M_CHECK}|, {CHECK_POS} positions of "
                                        f"text 0, all K + 1 tokens", "value": float(dg.max()), "reference": M_TOL,
         "pass": bool(dg.max() <= M_TOL),
         "note": f"g(y_t) only: {dg[:, 0].max():.4f}; mean over all {dg.size}: {dg.mean():.4f}"},
        {"check": "inequality (2)", "statement": "gap mean and quartiles >= -0.01 (smallest)",
         "value": min(gap_stats), "reference": -0.01, "pass": bool(min(gap_stats) >= -0.01),
         "note": f"min gap over positions {ineq['gap min']:.4f}, share below -0.01 "
                 f"{ineq['share of positions with gap < -0.01']:.4f}"},
        {"check": "green fraction", "statement": "green fraction at rho 0 against the average q_t (reported only)",
         "value": ineq["green fraction at rho 0"], "reference": qavg, "pass": "",
         "note": f"difference {ineq['green fraction at rho 0'] - qavg:+.4f}"}]

    write_csv(res_dir / "miss_rates.csv", rows)
    write_csv(res_dir / "rho_half.csv",
              [{"quantity": f"rho_half {a}", "value": v, "note": "" if np.isfinite(v) else f"above {rhos[-1]:g}"}
               for a, v in rh.items()]
              + [{"quantity": "share of the key's value recovered", "value": share,
                  "note": "(rho_half random - rho_half key-blind) / (rho_half random - rho_half key holder)"}])
    write_csv(res_dir / "ceilings.csv", [{"rho": r, "budget": j, "tau_over_T": tau_used / T, "ceiling_q_min": a,
                                          "q_min": qmin, "ceiling_q_avg": b, "q_avg": qavg}
                                         for r, j, a, b in zip(rhos, budgets, ceil["q_min"], ceil["q_avg"])])
    write_csv(res_dir / "inequality2.csv", [{"statistic": k, "value": float(v)} for k, v in ineq.items()])
    write_csv(res_dir / "checks.csv", checks)
    np.savez_compressed(res_dir / "positions.npz", prompts=prompts.numpy(), y=yn, keys=keys, cand=cand, g=g, q=q,
                        rhs=rhs, green=gm, null_counts=null, check_dg=dg)

    labels = ("random edits: random positions, each token swapped for one of the 20 likeliest tokens",
              "key-blind edits: swap where the model alone says the token is most likely green",
              "key holder: knows every green list, swaps green tokens for the likeliest red one")
    curves = [(lab, col_of(name, "miss_rate"), col_of(name, "ci_lo"), col_of(name, "ci_hi"))
              for lab, name in zip(labels, attacks)]
    ceilings = [(f"ceiling for the key holder: bound of Section 4 with the smallest q (q_min = {qmin:.3f})",
                 ceil["q_min"]),
                (f"same formula with the average q ({qavg:.3f}); not proven in the report", ceil["q_avg"])]
    note = (f"OPT-1.3B continues {len(prompts)} news prompts by {n} tokens. Green share of the vocabulary (γ) "
            f"{gamma:g}; bias added to green tokens (δ) {delta:g}. The detector flags a text with at least "
            f"{tau_used} green tokens (τ), which keeps false alarms on unwatermarked text at most {args.alpha:g} "
            f"(α). q: chance that a watermarked token is green, from the model's probabilities. "
            f"Shaded bands: 95 % Clopper-Pearson intervals.")
    fig_path = fig_dir / f"{FIG}.png"
    llm.figure(rhos, curves, ceilings, n, len(prompts),
               "Report's scheme: an independent random green list at every position", note, fig_path)
    copies = [Path("results/figures") / f"{SLUG}-{FIG}.png", Path("figures") / f"{FIG}.png"]
    for dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fig_path, dst)

    wall = time.perf_counter() - t_start
    summary = {"rho_half": rh, "share": share, "tau": tau, "tau_used": tau_used, "alpha_size": size, "null": nul,
               "q_min": qmin, "q_avg": qavg, "inequality2": ineq, "lists_check_max_dg": float(dg.max()),
               "failing_checks": [c["check"] for c in checks if c["pass"] is False],
               "run": {**vars(args), "texts_loaded": len(prompts), "seeds_np_torch_lists": [s_np, s_torch, s_lists],
                       "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
                       "torch": torch.__version__, "transformers": transformers.__version__,
                       "timing_s": timing, "wall_time_s": wall}}
    with open(res_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"tau {tau_used} of T {T} (exact size {size:.2e}); null flag rate {nul['flag_rate']:.2e} "
          f"[{nul['ci_lo']:.2e}, {nul['ci_hi']:.2e}] over {nul['trials']} trials")
    print("miss rate by rho: " + " ".join(f"{r:g}" for r in rhos))
    for name in attacks:
        print(f"  {name:<10} " + " ".join(f"{m:.2f}" for m in col_of(name, "miss_rate"))
              + f"   edits used at rho 1: mean {col_of(name, 'edits_mean')[-1]:.1f}")
    print("rho_half: " + ", ".join(f"{a} {v:.3f}" if np.isfinite(v) else f"{a} above {rhos[-1]:g}"
                                   for a, v in rh.items()) + f"; share of the key's value {share:.3f}")
    print("inequality (2): " + ", ".join(f"{k} {float(v):.4f}" for k, v in ineq.items()))
    for c in checks:
        print(f"{c['check']:<15} {'PASS' if c['pass'] is True else 'FAIL' if c['pass'] is False else '----'}  "
              f"{c['statement']}: {c['value']:.6g} (reference {c['reference']:g})  [{c['note']}]")
    print(f"wrote {res_dir}/ and {fig_path} -> " + ", ".join(map(str, copies)))
    print(f"timing {timing}; wall time {wall:.1f} s")


if __name__ == "__main__":
    main()
