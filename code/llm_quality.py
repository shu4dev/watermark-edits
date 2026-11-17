#!/usr/bin/env python
"""Text quality after the three attackers' edits (plans/2026-11-17-llm-text-quality.md).

Rebuilds one scheme's edited texts from the detection run (plans/2026-11-16-llm-blind-attack.md) with the parent
script's own attackers and its own order of random draws for the seed: keys, null keys, the random attacker, the key
holder (the key-blind attacker draws nothing). Report's scheme: code/llm_edits.py with the saved memberships; KGW:
code/llm_edits_kgw.py with the saved colour table (its lm-watermarking clone must be on PYTHONPATH).
Check 1: at the parent's tau_used, the rebuilt edits give the parent's missed count and edits used for every attacker
and grid rho, else the run stops. The draws are made for all the parent's texts, so Check 1 runs at every --texts;
--texts N then scores the first N texts only (smoke).
Score: the perplexity of a text is exp of the mean negative log-probability of its n new tokens after the edits,
given the prompt (context, not scored), under the oracle (facebook/opt-2.7b) in half precision with log-probabilities
in float32; each distinct sequence is scored once. Before scoring, the oracle's tokenizer must match OPT-1.3B's and
its logits must have OPT-1.3B's vocabulary size, else the run stops.
Writes <out>/results/<slug>/<scheme>/{quality,crossing,checks}.csv, summary.json, perplexity.npz and
<out>/figures/<slug>/llm_quality_<scheme>.png, copied to results/figures/ and figures/.
"""
import argparse
import csv
import json
import shutil
import textwrap
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from wmsim import llm
from wmsim.util import write_csv

SLUG, PARENT = "2026-11-17-llm-text-quality", "2026-11-16-llm-blind-attack"
RANDOM = "random edits: random positions, each token swapped for one of the 20 likeliest tokens"
LABELS = {"report": (RANDOM, "key-blind edits: swap where the model alone says the token is most likely green",
                     "key holder: knows every green list, swaps green tokens for the likeliest red one"),
          "kgw": (RANDOM, "key-blind edits: swap where the model alone rates this token and the next as likely green",
                  "key holder: knows every green list, greedy swaps that remove the most green tokens")}
TITLES = {"report": "Report's scheme: an independent random green list at every position",
          "kgw": "Published scheme (KGW): the green list is seeded by the previous token"}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def edited(rank, col, idx, j):
    """Tokens after the edits with rank < j: llm_edits_kgw.edited_tokens, which the report run cannot import."""
    return np.take_along_axis(idx, np.where(rank < j, col, 0)[..., None], -1)[..., 0]


def replay(scheme, d, gamma, seed):
    """The parent run's attacks on all its texts, drawn in its order. Returns the keys, {attacker: (ranks, columns)},
    the parent module and the table its green_counts reads."""
    rng = np.random.default_rng(int(np.random.SeedSequence(seed).generate_state(3)[0]))
    cand, y, texts = d["cand"], d["y"], len(d["y"])
    if scheme == "report":
        import llm_edits as parent
        table = d["green"]
        keys = rng.integers(2**63, size=texts)
        rng.integers(2**63, size=(texts, parent.NULL_KEYS))
        attacks = {"random": llm.random_attack(rng, cand, y), "key-blind": llm.blind_attack(d["g"], cand, y)[:2],
                   "key holder": parent.key_holder(rng, table, cand, y)}
    else:
        import llm_edits_kgw as parent
        table = d["colours"]
        keys = rng.integers(1, 2**63, size=texts)
        rng.integers(1, 2**63, size=(texts, parent.NULL_KEYS))
        attacks = {"random": llm.random_attack(rng, cand, y), "key-blind": parent.blind_kgw(d["g"], cand, y, gamma)}
        attacks["key holder"] = parent.key_holder(rng, table, cand, y)[:2]
    return keys, attacks, parent, table


@torch.inference_mode()
def mean_nll(model, seqs, L, batch):
    """Mean negative log-probability of each sequence's tokens after the first L, given all tokens before them."""
    out = []
    for s in range(0, len(seqs), batch):
        x = torch.as_tensor(seqs[s:s + batch], device=model.device)
        logp = torch.log_softmax(model(x, use_cache=False).logits[:, L - 1:-1].float(), -1)
        out.append(-logp.gather(-1, x[:, L:, None])[..., 0].mean(1).cpu())
    return torch.cat(out).numpy()


def figure(rhos, med, q25, q75, labels, base, cross, title, ylabel, note, n, path):
    """Median perplexity (log axis) against edit fraction, one line per attacker with its quartile band, a dashed line
    at the unedited baseline and a ring on each line at that attacker's rho_half."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    ink, ink2, gridc, axc, surf = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    colors = ("#2a78d6", "#eb6834", "#1baf7a")  # the parent figures' slots: random, key-blind, key holder
    fig, ax = plt.subplots(figsize=(8.6, 8.2), facecolor=surf)
    ax.set_facecolor(surf)
    ax.set_yscale("log")
    for label, m, lo, hi, col, mk in zip(labels, med, q25, q75, colors, "osD"):
        ax.fill_between(rhos, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(rhos, m, color=col, lw=2, marker=mk, ms=6, mec=surf, mew=1.0, label=textwrap.fill(label, 90))
    ax.axhline(base, color=ink2, lw=1.2, ls="--", zorder=1, label="watermarked text before any edit (median)")
    ax.plot([c["rho_half"] for c in cross], [c["median"] for c in cross], ls="none", marker="o", ms=14, mfc="none",
            mec=ink, mew=1.6, label="edit fraction where this attacker defeats the detector")
    lo, hi = min(q25.min(), base) / 1.15, q75.max() * 1.15
    ticks = [m * 10**e for e in range(6) for m in (1, 2, 3, 5) if lo <= m * 10**e <= hi]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(lo, hi)
    ax.set_xlim(rhos[0] - 0.02, rhos[-1] + 0.02)
    ax.grid(color=gridc, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(axc)
    ax.tick_params(colors=ink2, labelsize=10)
    ax.set_xlabel(f"edit fraction (share of the {n} tokens replaced)", color=ink2, fontsize=11)
    ax.set_ylabel(ylabel, color=ink2, fontsize=11)
    ax.set_title(title, color=ink, fontsize=12)
    fig.tight_layout(rect=(0, 0.34, 1, 1))  # room below for the legend and the note
    x0 = ax.get_position().x0
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(x0 - 0.01, 0.325), frameon=False,
               fontsize=10.5, labelcolor=ink)
    fig.text(x0, 0.015, textwrap.fill(note, 92, break_on_hyphens=False), ha="left", va="bottom", color=ink2,
             fontsize=10.5)
    fig.savefig(path, dpi=150, facecolor=surf)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheme", choices=("report", "kgw"), required=True)
    ap.add_argument("--oracle", default="facebook/opt-2.7b", help="the larger model that scores the texts")
    ap.add_argument("--seed", type=int, default=0, help="the parent run's seed, whose draws are replayed")
    ap.add_argument("--batch", type=int, default=32, help="sequences per forward pass of the oracle")
    ap.add_argument("--texts", type=int, default=None, help="score only the first N texts (smoke); default all")
    ap.add_argument("--run-root", default="data", help=f"reads <run-root>/results/{PARENT}/<scheme>/")
    ap.add_argument("--out", default="data", help="writes <out>/results/<slug>/<scheme>/ and <out>/figures/<slug>/")
    args = ap.parse_args()
    t_start = last = time.perf_counter()
    timing = {}

    def lap(name):  # seconds since the previous lap
        nonlocal last
        timing[name], last = round(time.perf_counter() - last, 1), time.perf_counter()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    src = Path(args.run_root) / "results" / PARENT / args.scheme
    res_dir, fig_dir = Path(args.out) / "results" / SLUG / args.scheme, Path(args.out) / "figures" / SLUG
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = dict(np.load(src / "positions.npz"))
    run = json.loads((src / "summary.json").read_text())
    par = read_csv(src / "miss_rates.csv")
    rh = {r["quantity"].removeprefix("rho_half "): float(r["value"]) for r in read_csv(src / "rho_half.csv")
          if r["quantity"].startswith("rho_half ")}
    gamma, n, tau = run["run"]["gamma"], run["run"]["n"], run["tau_used"]
    s_np = int(np.random.SeedSequence(args.seed).generate_state(3)[0])
    assert s_np == run["run"]["seeds_np_torch_lists"][0], f"--seed {args.seed} is not the parent run's seed"

    # Check 1: the replayed edits are the ones the detector saw
    keys, attacks, parent, table = replay(args.scheme, d, gamma, args.seed)
    checks = [{"check": "keys", "statement": "replayed text keys equal the parent's saved keys",
               "value": int((keys == d["keys"]).sum()), "reference": len(keys),
               "pass": bool((keys == d["keys"]).all()), "note": ""}]
    for name, (rank, col) in attacks.items():
        cells = [r for r in par if r["attacker"] == name]
        bad = []
        for r in cells:
            counts, used = parent.green_counts(rank, col, table, int(r["budget"]))
            mine = (int((counts < tau).sum()), float(used.mean()), int(used.max()))
            ref = (int(r["missed"]), float(r["edits_mean"]), int(r["edits_max"]))
            if mine != ref:
                bad.append(f"rho {r['rho']}: (missed, edits mean, edits max) {mine} against {ref}")
        checks.append({"check": "replay", "statement": f"{name}: missed count at tau {tau}, mean and max edits equal "
                                                       f"the parent's at every grid rho",
                       "value": len(cells) - len(bad), "reference": len(cells), "pass": not bad,
                       "note": "; ".join(bad[:3])})
    rhos = np.array([float(r["rho"]) for r in par if r["attacker"] == "random"])
    budgets = [int(r["budget"]) for r in par if r["attacker"] == "random"]
    if not all(c["pass"] for c in checks):
        write_csv(res_dir / "checks.csv", checks)
        raise SystemExit("Check 1 failed, the rebuilt edits are not the ones the detector saw:\n"
                         + "\n".join(f"{c['statement']}: {c['value']}/{c['reference']} {c['note']}"
                                     for c in checks if not c["pass"]))
    lap("replay_s")

    texts = len(d["y"]) if args.texts is None else args.texts
    prompts, idx = d["prompts"][:texts], np.concatenate([d["y"][..., None], d["cand"]], -1)[:texts]
    edit = getattr(parent, "edited_tokens", edited)
    Z = np.stack([[edit(rank[:texts], col[:texts], idx, j) for j in budgets] for rank, col in attacks.values()])
    seqs, where = [], np.empty(Z.shape[:3], int)  # each text's distinct sequences, and where each (attacker, rho) is
    for i in range(texts):
        u, inv = np.unique(Z[:, :, i].reshape(-1, n), axis=0, return_inverse=True)
        where[:, :, i] = len(seqs) + inv.reshape(Z.shape[:2])
        seqs += [np.r_[prompts[i], row] for row in u]
    seqs, L = np.array(seqs), prompts.shape[1]

    # the oracle reads the same token ids as OPT-1.3B
    base_tok, tok = AutoTokenizer.from_pretrained(llm.MODEL), AutoTokenizer.from_pretrained(args.oracle)
    model = AutoModelForCausalLM.from_pretrained(args.oracle, dtype=torch.float16).to(device).eval()
    V_base = AutoConfig.from_pretrained(llm.MODEL).vocab_size
    with torch.inference_mode():
        V = model(torch.as_tensor(seqs[:1], device=device), use_cache=False).logits.shape[-1]
    sample = [base_tok.decode(np.r_[prompts[i, 1:], d["y"][i]]) for i in range(min(20, texts))]
    same_ids = sum(tok(s, add_special_tokens=False)["input_ids"] == base_tok(s, add_special_tokens=False)["input_ids"]
                   for s in sample)
    same_vocab = tok.get_vocab() == base_tok.get_vocab()
    checks.append({"check": "oracle", "statement": f"{args.oracle} has OPT-1.3B's vocabulary, the same ids on "
                                                   f"{len(sample)} decoded texts and logits of size {V_base}",
                   "value": V, "reference": V_base, "pass": bool(same_vocab and same_ids == len(sample) and V == V_base),
                   "note": f"vocabulary maps equal: {same_vocab}; texts with equal ids: {same_ids}/{len(sample)}"})
    checks.append({"check": "texts", "statement": "texts scored (the replay and Check 1 cover all of the parent's)",
                   "value": texts, "reference": len(d["y"]), "pass": "",
                   "note": "" if texts == len(d["y"]) else f"smoke: first {texts} of {len(d['y'])} texts"})
    if not checks[-2]["pass"]:
        write_csv(res_dir / "checks.csv", checks)
        raise SystemExit(f"oracle check failed: {checks[-2]['note']}; logits {V} against {V_base}")
    lap("oracle_load_s")

    nll = mean_nll(model, seqs, L, args.batch)
    lap("score_s")
    ppl = np.exp(nll[where].astype(float))  # (attacker, rho, text)
    assert (ppl[:, 0] == ppl[0, 0]).all()  # rho 0 is the same unedited text for every attacker
    names = list(attacks)
    q25, med, q75 = np.quantile(ppl, [0.25, 0.5, 0.75], axis=2)
    rows = [{"attacker": name, "rho": r, "budget": j, "median": med[a, k], "q25": q25[a, k], "q75": q75[a, k],
             "geometric_mean": float(np.exp(np.log(ppl[a, k]).mean())), "texts": texts}
            for a, name in enumerate(names) for k, (r, j) in enumerate(zip(rhos, budgets))]
    base = float(med[0, 0])
    cross = [{"attacker": name, "rho_half": rh[name], "median": float(np.interp(rh[name], rhos, med[a])),
              "ratio_to_baseline": float(np.interp(rh[name], rhos, med[a])) / base}
             for a, name in enumerate(names) if np.isfinite(rh[name])]

    write_csv(res_dir / "quality.csv", rows)
    write_csv(res_dir / "crossing.csv", [{"attacker": "baseline: watermarked text before any edit", "rho_half": 0.0,
                                          "median": base, "ratio_to_baseline": 1.0}] + cross)
    write_csv(res_dir / "checks.csv", checks)
    np.savez_compressed(res_dir / "perplexity.npz", ppl=ppl, rhos=rhos, budgets=budgets, attackers=names)

    oracle = args.oracle.split("/")[-1].upper()
    note = (f"Perplexity of a text: the exponential of the average negative log-probability that {oracle} gives its "
            f"{n} new tokens after the 50-token news prompt. Lines: median over the {texts} texts; shaded bands: the "
            f"middle half of the texts (between the quartiles). Rings: each attacker's edit fraction where the detector "
            f"misses half of the watermarked texts (ρ_half), from the detection run.")
    fig_path = fig_dir / f"llm_quality_{args.scheme}.png"
    figure(rhos, med, q25, q75, LABELS[args.scheme], base, cross, TITLES[args.scheme],
           f"perplexity under a larger model ({oracle})\nlower reads more natural (log scale)", note, n, fig_path)
    copies = [Path("results/figures") / f"{SLUG}-{fig_path.name}", Path("figures") / fig_path.name]
    for dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fig_path, dst)

    wall = time.perf_counter() - t_start
    mem = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    summary = {"scheme": args.scheme, "oracle": args.oracle, "parent": str(src), "tau_used": tau,
               "baseline_median": base, "crossing": cross, "rho_half": rh,
               "failing_checks": [c["check"] for c in checks if c["pass"] is False],
               "run": {**vars(args), "texts_scored": texts, "parent_texts": len(d["y"]),
                       "distinct_sequences": len(seqs), "sequences_per_s": len(seqs) / max(timing["score_s"], 1e-9),
                       "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
                       "torch": torch.__version__, "transformers": transformers.__version__,
                       "timing_s": timing, "wall_time_s": wall, "max_gpu_mem_gib": mem}}
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    for c in checks:
        print(f"{c['check']:<7} {'PASS' if c['pass'] is True else 'FAIL' if c['pass'] is False else '----'}  "
              f"{c['statement']}: {c['value']} (reference {c['reference']})" + (f"  [{c['note']}]" if c["note"] else ""))
    print(f"{len(seqs)} distinct sequences of {L} + {n} tokens scored for {texts} texts")
    print("median perplexity by rho: " + " ".join(f"{r:g}" for r in rhos))
    for a, name in enumerate(names):
        print(f"  {name:<10} " + " ".join(f"{m:.1f}" for m in med[a]))
    print(f"baseline median {base:.2f}; at rho_half: " + ", ".join(
        f"{c['attacker']} {c['rho_half']:.3f} -> {c['median']:.2f} ({c['ratio_to_baseline']:.2f} x)" for c in cross))
    print(f"wrote {res_dir}/ and {fig_path} -> " + ", ".join(map(str, copies)))
    print(f"timing {timing}; peak GPU memory {mem:.1f} GiB; wall time {wall:.1f} s")


if __name__ == "__main__":
    main()
