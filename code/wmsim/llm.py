"""Shared pieces of the OPT-1.3B rounds (plans/2026-11-16-llm-blind-attack.md), free of any one watermark scheme.

Model facebook/opt-1.3b in half precision, probabilities in float32. Prompts: [BOS] + the first 50 tokens of the first
C4 realnewslike validation documents (streamed) with at least 250 tokens. generate() samples exactly n new tokens at
temperature 1 (no top-k or top-p, EOS suppressed) under any logits processor; base_law() is the teacher-forced p_t.
posterior(): M shared Bernoulli(gamma) lists B_m; at a position with emitted token y, U_m = sum_v B_m(v) p_t(v),
w_m = e^{delta B_m(y)} / (1 + (e^delta - 1) U_m) and g_t(v) = sum_m w_m B_m(v) / sum_m w_m for v = y and the K = 20
likeliest tokens under p_t; q_t = mean_m e^delta U_m / (1 + (e^delta - 1) U_m) is the left side of inequality (2) and
gamma - (e^{-delta} - 1) gamma (1 - gamma) (1 - sum_v p_t(v)^2) its right side.
Attackers edit positions in rank order (budget j edits the ranks below j) and replace y_t only by a candidate other
than y_t: random (uniform positions, uniform candidate) and key-blind (z_t = the candidate with the smallest g_t,
positive gains g_t(y_t) - g_t(z_t), largest first). The key holder is scheme-specific. Detector: green count >= tau =
wmsim.greencount.threshold(T, gamma, alpha). Tables with Clopper-Pearson 95 % intervals, rho_half by linear
interpolation, the key share and the ceilings e^{-T d(tau/T + rho || q)}, and the figure.
Needs the `llm` dependency group (torch, transformers, datasets): GPU boxes only.
"""
import math
import textwrap

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import beta
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from .exponent import kl_bern

MODEL, K, PROMPT_LEN, MIN_TOKENS = "facebook/opt-1.3b", 20, 50, 250


def load_model(device):
    tok = AutoTokenizer.from_pretrained(MODEL)
    return tok, AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()


def load_prompts(tok, texts):
    """(texts, 1 + PROMPT_LEN) token ids: [BOS] + the first PROMPT_LEN tokens of each of the first `texts` C4
    realnewslike validation documents, in stream order, that have at least MIN_TOKENS tokens."""
    out = []
    for doc in load_dataset("allenai/c4", "realnewslike", split="validation", streaming=True):
        ids = tok(doc["text"], add_special_tokens=False, truncation=True, max_length=MIN_TOKENS)["input_ids"]
        if len(ids) == MIN_TOKENS:
            out.append([tok.bos_token_id] + ids[:PROMPT_LEN])
            if len(out) == texts:
                break
    return torch.tensor(out)


@torch.inference_mode()
def generate(model, prompts, n, processor, batch=32):
    """Exactly n new tokens per prompt, (texts, n) on the CPU, sampled at temperature 1 with no top-k or top-p and EOS
    suppressed. processor(rows) returns the Hugging Face LogitsProcessor for prompts[rows] (rows: an index array), so
    a scheme can key each text; `lambda rows: proc` shares one processor."""
    gc, out = model.generation_config, []
    for s in range(0, len(prompts), batch):
        x = prompts[s:s + batch].to(model.device)
        z = model.generate(x, attention_mask=torch.ones_like(x), do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                           min_new_tokens=n, max_new_tokens=n, pad_token_id=gc.pad_token_id,
                           logits_processor=LogitsProcessorList([processor(np.arange(s, s + len(x)))]))
        out.append(z[:, x.shape[1]:].cpu())
    y = torch.cat(out)
    assert y.shape == (len(prompts), n) and not (y == gc.eos_token_id).any()
    return y


@torch.inference_mode()
def base_law(model, prompts, y):
    """Base p_t, (texts, n, V) float32, from one teacher-forced pass over prompt + y. EOS gets probability 0 and the
    rest is renormalised, since generate() suppresses it."""
    ids = torch.cat([prompts, y], 1).to(model.device)
    logits = model(ids, use_cache=False).logits[:, prompts.shape[1] - 1:-1].float()
    logits[..., model.generation_config.eos_token_id] = -math.inf
    return logits.softmax(-1)


def lists(V, gamma, M, seed, device, chunk=4096):
    """M Bernoulli(gamma) lists over the vocabulary as (rows, V) float 0/1 chunks from one seeded generator, so the
    first lists are the same for every M."""
    gen = torch.Generator(device).manual_seed(seed)
    for s in range(0, M, chunk):
        yield (torch.rand(min(chunk, M - s), V, generator=gen, device=device) < gamma).float()


@torch.inference_mode()
def posterior(p, y, chunks, gamma, delta):
    """At P positions with base law p (P, V) and emitted tokens y (P,), over the lists in chunks: the K candidates
    (P, K), likeliest first; g (P, K + 1), the posterior that a token is green, for [y, candidates]; q_t (P,); and the
    right side of inequality (2) (P,)."""
    cand = p.topk(K, dim=-1).indices
    idx, e = torch.cat([y[:, None], cand], 1), math.exp(delta)
    num = den = qsum = 0.0
    M = 0
    for B in chunks:
        U = p @ B.T
        w = torch.exp(delta * B[:, y].T) / (1 + (e - 1) * U)
        num = num + (w @ B).gather(1, idx)
        den = den + w.sum(1)
        qsum = qsum + (e * U / (1 + (e - 1) * U)).sum(1)
        M += len(B)
    rhs = gamma - math.expm1(-delta) * gamma * (1 - gamma) * (1 - (p * p).sum(1))
    return cand, num / den[:, None], qsum / M, rhs


def position_stats(model, prompts, y, chunks, gamma, delta, batch=8):
    """posterior() at every generated position, `batch` texts per pass (chunks: a list, reused by every pass).
    Returns numpy arrays cand (texts, n, K), g (texts, n, K + 1), q (texts, n), rhs (texts, n)."""
    out = []
    for s in range(0, len(y), batch):
        p = base_law(model, prompts[s:s + batch], y[s:s + batch])
        b, n, V = p.shape
        r = posterior(p.reshape(b * n, V), y[s:s + batch].reshape(-1).to(p.device), chunks, gamma, delta)
        out.append([x.reshape(b, n, *x.shape[1:]).cpu().numpy() for x in r])
    return [np.concatenate(z) for z in zip(*out)]


def ranks(key, ok):
    """Edit order as ranks, (texts, n): positions with ok get 0, 1, ... by increasing key, the others n (never
    edited). Budget j edits the positions with rank < j."""
    r = np.argsort(np.argsort(np.where(ok, key, np.inf), -1, kind="stable"), -1)
    return np.where(ok, r, key.shape[-1])


def random_attack(rng, cand, y):
    """Random attacker: uniform positions, each replaced by a uniform candidate other than y_t. Returns the ranks and
    the replacement's column in [y, candidates] (1..K)."""
    ok = cand != y[..., None]
    col = 1 + np.argmax(np.where(ok, rng.random(ok.shape), -1.0), -1)
    return ranks(rng.random(y.shape), np.ones(y.shape, bool)), col


def blind_attack(g, cand, y):
    """Key-blind attacker: z_t = the candidate other than y_t with the smallest g_t, gain g_t(y_t) - g_t(z_t); edits
    at the largest positive gains first. Returns the ranks, z_t's column in [y, candidates] and the gain."""
    gz = np.where(cand != y[..., None], g[..., 1:], np.inf)
    col, gain = 1 + gz.argmin(-1), g[..., 0] - gz.min(-1)
    return ranks(-gain, gain > 0), col, gain


def clopper_pearson(k, N):
    """95 % Clopper-Pearson interval for k successes in N trials."""
    return (float(beta.ppf(0.025, k, N - k + 1)) if k else 0.0), (float(beta.ppf(0.975, k + 1, N - k)) if k < N else 1.0)


def miss_row(attacker, rho, j, counts, used, tau):
    """Table row at budget j: the detector misses a text when its green count is below tau; used = edits per text."""
    miss = int((counts < tau).sum())
    lo, hi = clopper_pearson(miss, len(counts))
    return {"attacker": attacker, "rho": rho, "budget": j, "edits_mean": float(used.mean()),
            "edits_max": int(used.max()), "within_budget": bool(used.max() <= j), "texts": len(counts),
            "missed": miss, "miss_rate": miss / len(counts), "ci_lo": lo, "ci_hi": hi}


def rho_half(rhos, miss):
    """Edit fraction where the miss rate first reaches 1/2, by linear interpolation on the grid; nan if it never
    does on the grid."""
    i = int(np.argmax(miss >= 0.5))
    if miss[i] < 0.5:
        return np.nan
    if i == 0:
        return float(rhos[0])
    return float(rhos[i - 1] + (0.5 - miss[i - 1]) * (rhos[i] - rhos[i - 1]) / (miss[i] - miss[i - 1]))


def key_share(rh_random, rh_blind, rh_key):
    """Share of the key's value the key-blind attacker recovers, from the three rho_half."""
    return (rh_random - rh_blind) / (rh_random - rh_key)


def ceiling(rhos, T, tau, q):
    """e^{-T d(tau/T + rho || q)} at each rho, 1 where tau/T + rho >= q (kl_bern is 0 there)."""
    return np.array([math.exp(-T * kl_bern(tau / T + r, q)) for r in rhos])


def null_check(counts, tau, alpha):
    """Flag rate on key-independent text (counts: green counts under fresh keys). If its 95 % interval lies above
    alpha, tau moves to the smallest t whose empirical flag rate is at most alpha. Returns (tau to use, row)."""
    k = int((counts >= tau).sum())
    lo, hi = clopper_pearson(k, counts.size)
    used = tau if lo <= alpha else next(t for t in range(int(counts.max()) + 2) if (counts >= t).mean() <= alpha)
    return used, {"trials": counts.size, "flagged": k, "flag_rate": k / counts.size, "ci_lo": lo, "ci_hi": hi,
                  "alpha": alpha, "tau": tau, "tau_used": used}


def ineq2(q, rhs, gamma, green0):
    """Inequality (2), q_t >= right side, over every generated position: the gap's mean, quartiles, minimum and share
    below -0.01; q_min; the average q_t; the share of positions with q_t < gamma + 0.01; and green0, the green
    fraction at rho = 0, which estimates the average q_t."""
    gap = (q - rhs).ravel()
    return {"gap mean": gap.mean(), "gap 25th percentile": np.quantile(gap, 0.25), "gap median": np.median(gap),
            "gap 75th percentile": np.quantile(gap, 0.75), "gap min": gap.min(),
            "share of positions with gap < -0.01": (gap < -0.01).mean(), "q_min": q.min(), "average q_t": q.mean(),
            "share of positions with q_t < gamma + 0.01": (q < gamma + 0.01).mean(), "green fraction at rho 0": green0}


def figure(rhos, curves, ceilings, n, texts, title, note, path):
    """Miss rate against edit fraction: one line per attacker (curves: (label, miss, ci_lo, ci_hi), key holder last)
    with its 95 % band, the ceilings (label, values) dashed and dotted in the key holder's colour, a grey line at one
    half. The labels and the note under the legend explain every symbol in words."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, gridc, axc, surf = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    colors = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3, fixed order
    fig, ax = plt.subplots(figsize=(8.6, 8.2), facecolor=surf)
    ax.set_facecolor(surf)
    for (label, miss, lo, hi), col, mk in zip(curves, colors, "osD"):
        ax.fill_between(rhos, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(rhos, miss, color=col, lw=2, marker=mk, ms=6, mec=surf, mew=1.0, label=textwrap.fill(label, 90))
    for (label, v), ls in zip(ceilings, ("--", ":")):
        ax.plot(rhos, v, color=colors[-1], lw=2, ls=ls, label=textwrap.fill(label, 90))
    ax.axhline(0.5, color=ink2, lw=1, zorder=1, label="half of the texts missed")
    ax.set_xlim(rhos[0] - 0.02, rhos[-1] + 0.02)
    ax.set_ylim(-0.02, 1.04)
    ax.grid(color=gridc, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(axc)
    ax.tick_params(colors=ink2, labelsize=10)
    ax.set_xlabel(f"edit fraction (share of the {n} tokens replaced)", color=ink2, fontsize=11)
    ax.set_ylabel(f"miss rate\n(share of the {texts} watermarked texts not flagged)", color=ink2, fontsize=11)
    ax.set_title(title, color=ink, fontsize=12)
    fig.tight_layout(rect=(0, 0.34, 1, 1))  # room below for the legend and the note
    x0 = ax.get_position().x0
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(x0 - 0.01, 0.325), frameon=False,
               fontsize=10.5, labelcolor=ink)
    fig.text(x0, 0.015, textwrap.fill(note, 92, break_on_hyphens=False), ha="left", va="bottom", color=ink2,
             fontsize=10.5)
    fig.savefig(path, dpi=150, facecolor=surf)
    plt.close(fig)
