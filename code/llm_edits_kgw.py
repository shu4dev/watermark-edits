#!/usr/bin/env python
"""Key-blind edits on OPT-1.3B text under the published scheme, KGW (plans/2026-11-16-llm-blind-attack.md).

Scheme: jwkirchenbauer/lm-watermarking at commit 82922516930c02f8aa322765defdb5863d07a00e, its clone on PYTHONPATH.
extended_watermark_processor.py with the scheme "simple_1" (additive PRF, context width 1, no self-hash) and one hash
key k per text, seeding_scheme "ff-additive_prf-1-False-<k>": the green list of a token is seeded by k and the token
before it (the prompt's last token for the first new token). The library keys every row of a batch alike, so
generation (wmsim.llm.generate) applies one library processor per row, each with its own text's key.
Detector: green count of the n new tokens, each scored with its predecessor (ignore_repeated_ngrams=False, so
T = n), flagged at >= tau = threshold(T, gamma, alpha). An edit at t changes the token scored at t and reseeds the list
at t + 1, so counts after edits come from a colour table: for every text and position, the colour of each of the 21
tokens [y_t, candidates] under the list seeded by each of the 21 tokens [y_{t-1}, candidates at t - 1]. Colours come
from the library's own _get_greenlist_ids on the GPU, as in generation; the library's WatermarkDetector checks them.
Three attackers with budget floor(rho n), each replacing y_t only by one of the K = 20 likeliest tokens under p_t
other than y_t: random (wmsim.llm); key-blind, gain g_t(y_t) - g_t(z_t) + g_{t+1}(y_{t+1}) - gamma, positive gains
largest first, a position whose predecessor is already chosen taken last; key holder, greedy on the colour table
(largest drop of the green count first, so a token and the next one both turned red before one green removed; never
an edit that does not lower the count), a reference that is not claimed optimal.
Checks: null (20 fresh keys per text), budgets, inequality (2), g at M lists against 65 536 lists on the first 20
positions of text 0 (within 0.02), and the colour table against the library's detector on the first texts.
Writes <out>/results/<slug>/kgw/{miss_rates,rho_half,ceilings,inequality2,checks}.csv, positions.npz, summary.json
and <out>/figures/<slug>/llm_miss_rate_kgw.png, copied to results/figures/ and figures/.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import extended_watermark_processor as ewp  # lm-watermarking clone on PYTHONPATH
import numpy as np
import torch
import transformers
from transformers import LogitsProcessor

from wmsim import llm
from wmsim.greencount import threshold
from wmsim.util import write_csv

SLUG, FIG = "2026-11-16-llm-blind-attack", "llm_miss_rate_kgw"
LIB_COMMIT = "82922516930c02f8aa322765defdb5863d07a00e"
NULL_KEYS, M_CHECK, M_TOL, CHECK_POS, CHECK_TEXTS = 20, 65_536, 0.02, 20, 4


def scheme(key):
    """The library's "simple_1" (previous token only) with the text's own hash key."""
    return f"ff-additive_prf-1-False-{int(key)}"


def library(key, vocab, gamma, delta, device=None):
    """The library's processor for one key; with device, its generator is made there now (generation makes it lazily
    on the prompts' device)."""
    p = ewp.WatermarkLogitsProcessor(vocab=vocab, gamma=gamma, delta=delta, seeding_scheme=scheme(key))
    if device is not None:
        p.rng = torch.Generator(device=device)
    return p


def lib_commit():
    """Commit of the lm-watermarking clone, read from its .git folder."""
    git = Path(ewp.__file__).resolve().parent / ".git"
    ref = (git / "HEAD").read_text().split()[-1]
    if not ref.startswith("refs/"):
        return ref
    if (git / ref).exists():
        return (git / ref).read_text().strip()
    return next(line.split()[0] for line in (git / "packed-refs").read_text().splitlines() if line.endswith(" " + ref))


class PerText(LogitsProcessor):
    """The library's processors applied row by row, each row with its own text's key."""

    def __init__(self, procs):
        self.procs = procs

    def __call__(self, input_ids, scores):
        return torch.cat([p(input_ids[b:b + 1], scores[b:b + 1]) for b, p in enumerate(self.procs)])


def colours(proc, prev, toks, V):
    """out[..., a, b]: whether toks[..., b] is in the library's green list seeded by prev[..., a] (proc's key), from
    proc._get_greenlist_ids on proc's generator device; one list per distinct seeding token."""
    dev = proc.rng.device
    uniq, inv = np.unique(prev, return_inverse=True)
    seeds = torch.as_tensor(uniq, device=dev)
    G = torch.zeros(len(uniq), V, dtype=torch.bool, device=dev)
    for r in range(len(uniq)):
        G[r, proc._get_greenlist_ids(seeds[r:r + 1])] = True
    inv, toks = torch.as_tensor(inv.reshape(prev.shape), device=dev), torch.as_tensor(toks, device=dev)
    return G[inv[..., :, None], toks[..., None, :]].cpu().numpy()


def green_counts(rank, col, C, j):
    """Green count of each text and its edits used when the positions with rank < j take the column col: the token at t
    is column c_t of [y_t, candidates], scored under the list seeded by column c_{t-1} (the prompt's last token at
    t = 0, which is column 0 of C[:, 0])."""
    edited = rank < j
    c = np.where(edited, col, 0)
    i, t = np.indices(c.shape)
    return C[i, t, np.pad(c[:, :-1], ((0, 0), (1, 0))), c].sum(1), edited.sum(1)


def edited_tokens(rank, col, idx, j):
    """The texts' tokens when the positions with rank < j take the column col of idx = [y, candidates]."""
    return np.take_along_axis(idx, np.where(rank < j, col, 0)[..., None], -1)[..., 0]


def blind_kgw(g, cand, y, gamma):
    """Key-blind attacker under KGW: z_t = the candidate with the smallest g_t (wmsim.llm.blind_attack), gain
    g_t(y_t) - g_t(z_t) plus g_{t+1}(y_{t+1}) - gamma for the list at t + 1 that the edit reseeds; positive gains,
    largest first, then the positions whose predecessor was chosen before them. Returns the ranks and z_t's column."""
    _, col, gain = llm.blind_attack(g, cand, y)
    gain = gain + np.pad(g[:, 1:, 0] - gamma, ((0, 0), (0, 1)))
    rank = np.full(y.shape, y.shape[1])
    for i in range(len(y)):
        first, last, chosen = [], [], set()
        for t in np.argsort(-gain[i], kind="stable"):
            if gain[i, t] > 0:
                (last if t - 1 in chosen else first).append(t)
                chosen.add(t)
        rank[i, first + last] = np.arange(len(chosen))
    return rank, col


def key_holder(rng, C, cand, y):
    """Greedy key holder on the colour table: each step edits the unedited position and candidate with the largest drop
    of the green count given the edits so far (ties: uniform position, likeliest candidate); it stops when no edit
    lowers the count. Not claimed optimal. Returns the ranks, the replacement's column and each edit's drop."""
    texts, n = y.shape
    rank, col, drop = np.full((texts, n), n), np.zeros((texts, n), int), np.zeros((texts, n), int)
    ok = np.concatenate([np.zeros((texts, n, 1), bool), cand != y[..., None]], -1)  # column 0 (y_t) is no edit
    for i in range(texts):
        c, Ci = np.zeros(n, int), C[i].astype(int)

        def best(s):  # largest drop of an edit at s and its column, given the current columns c
            a = c[s - 1] if s else 0
            d = Ci[s, a, c[s]] - Ci[s, a]
            if s + 1 < n:
                d = d + Ci[s + 1, c[s], c[s + 1]] - Ci[s + 1, :, c[s + 1]]
            d = np.where(ok[i, s], d, -9)
            return d.max(), d.argmax()

        D, B = (np.array(v) for v in zip(*[best(s) for s in range(n)]))
        tie, free = rng.random(n), np.ones(n, bool)
        for step in range(n):
            s = int(np.argmax(np.where(free, D + 0.5 * tie, -99)))
            if not free[s] or D[s] <= 0:
                break
            c[s], free[s], rank[i, s], col[i, s], drop[i, s] = B[s], False, step, B[s], D[s]
            for u in (s - 1, s + 1):
                if 0 <= u < n and free[u]:
                    D[u], B[u] = best(u)
    return rank, col, drop


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gamma", type=float, default=0.25, help="green share of the vocabulary")
    ap.add_argument("--delta", type=float, default=2.0, help="bias added to green logits")
    ap.add_argument("--alpha", type=float, default=1e-3, help="detector's false-positive rate")
    ap.add_argument("--n", type=int, default=200, help="new tokens per text")
    ap.add_argument("--texts", type=int, default=500)
    ap.add_argument("--rho-grid", type=float, nargs="+", default=[round(0.05 * i, 2) for i in range(21)])
    ap.add_argument("--lists", type=int, default=16_384, help="M, the Bernoulli lists of the key-blind posterior")
    ap.add_argument("--seed", type=int, default=0, help="keys, sampling, posterior lists, attackers' coins")
    ap.add_argument("--out", default="data", help="writes <out>/results/<slug>/kgw/ and <out>/figures/<slug>/")
    args = ap.parse_args()
    t_start = last = time.perf_counter()
    timing = {}

    def lap(name):  # seconds since the previous lap
        nonlocal last
        timing[name], last = round(time.perf_counter() - last, 1), time.perf_counter()

    head = lib_commit()
    assert head == LIB_COMMIT, f"lm-watermarking clone at {head}, the plan pins {LIB_COMMIT}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s_np, s_torch, s_lists = (int(s) for s in np.random.SeedSequence(args.seed).generate_state(3))
    rng = np.random.default_rng(s_np)
    torch.manual_seed(s_torch)
    gamma, delta, n, rhos = args.gamma, args.delta, args.n, np.array(args.rho_grid)
    res_dir, fig_dir = Path(args.out) / "results" / SLUG / "kgw", Path(args.out) / "figures" / SLUG
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tok, model = llm.load_model(device)
    prompts = llm.load_prompts(tok, args.texts)
    keys = rng.integers(1, 2**63, size=len(prompts))
    vocab = list(tok.get_vocab().values())  # the library uses only len(vocab): lists of token ids below it
    lap("load_s")
    y = llm.generate(model, prompts, n, lambda rows: PerText([library(keys[r], vocab, gamma, delta) for r in rows]))
    lap("generate_s")
    V = model.config.vocab_size
    B = list(llm.lists(V, gamma, args.lists, s_lists, device))
    cand, g, q, rhs = llm.position_stats(model, prompts, y, B, gamma, delta)
    lap("posterior_s")
    yn, last_tok = y.numpy(), prompts[:, -1].numpy()
    idx = np.concatenate([yn[..., None], cand], -1)  # columns: y_t, then the K candidates
    prev = np.concatenate([np.repeat(last_tok[:, None, None], idx.shape[-1], -1), idx[:, :-1]], 1)  # seeds at t
    C = np.stack([colours(library(k, vocab, gamma, delta, device), prev[i], idx[i], V) for i, k in enumerate(keys)])
    lists_built = sum(len(np.unique(p)) for p in prev)
    lap("colours_s")

    # null: the watermarked texts under fresh keys are key-independent text
    T = n
    tau, size = threshold(T, gamma, args.alpha)
    null_keys = rng.integers(1, 2**63, size=(len(keys), NULL_KEYS))
    yprev = np.concatenate([last_tok[:, None], yn[:, :-1]], 1)  # each new token's predecessor
    null = np.array([[colours(library(k, vocab, gamma, delta, device), yprev[i, :, None], yn[i, :, None], V).sum()
                      for k in ks] for i, ks in enumerate(null_keys)])
    tau_used, nul = llm.null_check(null.ravel(), tau, args.alpha)
    lap("null_s")

    attacks = {"random": llm.random_attack(rng, cand, yn), "key-blind": blind_kgw(g, cand, yn, gamma)}
    kh_rank, kh_col, kh_drop = key_holder(rng, C, cand, yn)
    attacks["key holder"] = (kh_rank, kh_col)
    budgets = [int(np.floor(r * n + 1e-9)) for r in rhos]
    count0 = C[:, :, 0, 0].sum(1)
    for j in budgets:  # the table's count after the key holder's first j edits = the start less their drops
        assert (green_counts(kh_rank, kh_col, C, j)[0] == count0 - np.where(kh_rank < j, kh_drop, 0).sum(1)).all()
    rows = [llm.miss_row(name, float(r), j, *green_counts(rank, col, C, j), tau_used)
            for name, (rank, col) in attacks.items() for r, j in zip(rhos, budgets)]
    col_of = lambda name, k: np.array([x[k] for x in rows if x["attacker"] == name])
    rh = {name: llm.rho_half(rhos, col_of(name, "miss_rate")) for name in attacks}
    share = llm.key_share(rh["random"], rh["key-blind"], rh["key holder"])
    qmin, qavg = float(q.min()), float(q.mean())
    ceil = {"q_min": llm.ceiling(rhos, T, tau_used, qmin), "q_avg": llm.ceiling(rhos, T, tau_used, qavg)}
    ineq = llm.ineq2(q, rhs, gamma, float(C[:, :, 0, 0].mean()))

    # the colour table against the library's own detector on the first texts: unedited, one fresh key, and each
    # attacker at n/4 and n/2 edits; the prompt's last token goes first as context, so the detector scores T = n
    m = min(CHECK_TEXTS, len(keys))
    check = [(i, keys[i], yn[i], count0[i]) for i in range(m)] + [(i, null_keys[i, 0], yn[i], null[i, 0])
                                                                  for i in range(m)]
    for rank, col in attacks.values():
        for j in (n // 4, n // 2):
            z, cnt = edited_tokens(rank, col, idx, j), green_counts(rank, col, C, j)[0]
            check += [(i, keys[i], z[i], cnt[i]) for i in range(m)]
    det_diff, det_T = [], set()
    for i, k, z, mine in check:
        det = ewp.WatermarkDetector(vocab=vocab, gamma=gamma, delta=delta, seeding_scheme=scheme(k), device=device,
                                    tokenizer=tok, normalizers=[], ignore_repeated_ngrams=False)
        out = det.detect(tokenized_text=torch.as_tensor(np.r_[last_tok[i], z], device=device))
        det_diff.append(int(out["num_green_tokens"]) - int(mine))
        det_T.add(int(out["num_tokens_scored"]))
    ewp.WatermarkDetector._get_ngram_score_cached.cache_clear()  # keyed on the detector object, not on its key

    # g at M lists against M_CHECK lists (the first M shared) on the first CHECK_POS positions of text 0
    p0, y0 = llm.base_law(model, prompts[:1], y[:1])[0, :CHECK_POS], y[0, :CHECK_POS].to(device)
    g_m = llm.posterior(p0, y0, B, gamma, delta)[1]
    g_big = llm.posterior(p0, y0, llm.lists(V, gamma, M_CHECK, s_lists, device), gamma, delta)[1]
    dg = (g_m - g_big).abs().cpu().numpy()
    lap("attacks_checks_s")

    gap_stats = [float(ineq[k]) for k in ("gap mean", "gap 25th percentile", "gap median", "gap 75th percentile")]
    over = max(x["edits_max"] - x["budget"] for x in rows)
    worst = max(map(abs, det_diff))
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
        {"check": "detector", "statement": f"green count from the colour table (processor's _get_greenlist_ids) minus "
                                           f"the library's WatermarkDetector on {len(check)} texts (largest |difference|)",
         "value": worst, "reference": 0, "pass": bool(worst == 0 and det_T == {T}),
         "note": f"first {m} texts: unedited, one fresh key, each attacker at {n // 4} and {n // 2} edits; "
                 f"tokens scored by the detector {sorted(det_T)}"},
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
                        rhs=rhs, colours=C, null_counts=null, check_dg=dg)

    labels = ("random edits: random positions, each token swapped for one of the 20 likeliest tokens",
              "key-blind edits: swap where the model alone rates this token and the next as likely green",
              "key holder: knows every green list, greedy swaps that remove the most green tokens")
    curves = [(lab, col_of(name, "miss_rate"), col_of(name, "ci_lo"), col_of(name, "ci_hi"))
              for lab, name in zip(labels, attacks)]
    ceilings = [(f"ceiling for the key holder: bound of Section 4 with the smallest q (q_min = {qmin:.3f})",
                 ceil["q_min"]),
                (f"same formula with the average q ({qavg:.3f}); not proven in the report", ceil["q_avg"])]
    note = (f"OPT-1.3B continues {len(prompts)} news prompts. Green share of the vocabulary (γ) {gamma:g}; bias added to "
            f"green tokens (δ) {delta:g}. The detector flags a text with at least {tau_used} green tokens (τ), "
            f"false-alarm rate at most {args.alpha:g} (α) on unwatermarked text. q: chance that a watermarked token is "
            f"green. A swap also reseeds the next list, so the greedy key holder (not proven optimal) can cross the "
            f"ceilings, built for one green lost per swap. Shaded bands: 95 % Clopper-Pearson intervals.")
    fig_path = fig_dir / f"{FIG}.png"
    llm.figure(rhos, curves, ceilings, n, len(prompts),
               "Published scheme (KGW): the green list is seeded by the previous token", note, fig_path)
    copies = [Path("results/figures") / f"{SLUG}-{FIG}.png", Path("figures") / f"{FIG}.png"]
    for dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fig_path, dst)

    wall = time.perf_counter() - t_start
    mem = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    summary = {"rho_half": rh, "share": share, "tau": tau, "tau_used": tau_used, "alpha_size": size, "null": nul,
               "q_min": qmin, "q_avg": qavg, "inequality2": ineq, "lists_check_max_dg": float(dg.max()),
               "detector_check_max_diff": worst,
               "failing_checks": [c["check"] for c in checks if c["pass"] is False],
               "library": {"repo": "jwkirchenbauer/lm-watermarking", "commit": LIB_COMMIT,
                           "file": "extended_watermark_processor.py", "seeding_scheme": "ff-additive_prf-1-False-<key>",
                           "vocab_len": len(vocab), "list_size": int(len(vocab) * gamma),
                           "colour_lists_built": lists_built},
               "run": {**vars(args), "texts_loaded": len(prompts), "seeds_np_torch_lists": [s_np, s_torch, s_lists],
                       "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
                       "torch": torch.__version__, "transformers": transformers.__version__,
                       "timing_s": timing, "wall_time_s": wall, "max_gpu_mem_gib": mem}}
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
    print(f"timing {timing}; {lists_built} colour lists; peak GPU memory {mem:.1f} GiB; wall time {wall:.1f} s")


if __name__ == "__main__":
    main()
