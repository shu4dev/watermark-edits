# watermark-edits

Exact false-negative exponents for green-list text watermarks under substitution attacks.

A green-list watermark (Kirchenbauer et al. 2023) raises the logits of a secret subset of the vocabulary at every
position; the detector counts how many tokens landed in those subsets. This repository computes, without sampling,
how fast the detector's miss rate decays with text length when an attacker has replaced a fraction rho of the tokens,
knowing the scheme but not the key, or seeing the key at a fraction lambda of the positions. The computation runs in
a token model with two entropy levels, where every probability is a finite sum, so the miss rate is exact down to
10^-700 and the exponent and the critical edit fraction come out as numbers rather than estimates.

## What is computed

Setting. A generator emits tokens from a next-token law p_t over a vocabulary V. The key is a sequence of
independent green lists G_t, each token green with probability gamma; the watermarked token is drawn from
p~_t(v) proportional to p_t(v) e^{delta 1[v in G_t]}. The attacker makes at most floor(rho n) substitutions; it knows
gamma, delta and p_t but not the key, and sees G_t at each position independently with probability lambda. The
detector knows the key, sees the edited text, and keeps its false-positive rate at or below alpha on any text written
without the key.

Quantities. beta_n(alpha, rho, lambda) is the miss rate at length n. E_n = -ln beta_n / n is the finite-n exponent
and E(rho) its limit; rho* is the smallest rho at which E(rho) = 0. Two detectors are computed: the green count,
and the Neyman-Pearson test that knows the attack rule and each position's entropy level but not which positions
were edited. Closed-form comparisons: the Cramer rate function at the finite-n threshold, the KL lower bound
D(Bern(gamma) || Bern(qbar)), and the mean-crossing root rho*.

## Install

Requires Python 3.13 (pinned in .python-version) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This creates .venv with the versions pinned in uv.lock (numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2). Without uv,
any recent numpy, scipy and matplotlib work; only the two Monte Carlo numbers (q_L and the diagnostic) can then differ
in the fourth decimal.

## Usage

```bash
uv run python code/exponent_greenlist.py --seed 0
```

The default run covers rho in {0, 0.05, ..., 0.95}, lambda in {0, 0.25, 0.5, 0.75, 1} and n in {100, 200, 500, 1000,
2000}, plus n = 10 000 at lambda = 0, and takes about ten seconds on a laptop CPU. Every flag has a default:

| flag | default | meaning |
|---|---|---|
| `--gamma` | 0.25 | green fraction |
| `--delta` | 2.0 | logit bias on green tokens |
| `--V` | 50000 | vocabulary size |
| `--pi` | 0.5 | probability that a position is high entropy (uniform over V); otherwise top token 0.9 |
| `--alpha` | 1e-3 | false-positive rate the threshold is calibrated to |
| `--rho-grid` | 0 0.05 ... 0.95 | edit fractions |
| `--lambda-grid` | 0 0.25 0.5 0.75 1 | leakage probabilities |
| `--n-list` | 100 200 500 1000 2000 | text lengths (n = 10 000 is added at lambda = 0) |
| `--trials` | 100000 | Monte Carlo key draws for q_L and the diagnostic |
| `--seed` | 0 | seed of the Monte Carlo parts; everything else is deterministic |
| `--out` | data | output root |

Numbers to expect at the defaults (seed 0):

| quantity | value |
|---|---|
| q_H, q_L, qbar | 0.7112, 0.3669, 0.5391 |
| E_2000 at rho = 0 (green count) | 0.1570; 0.1775 at n = 10 000; limit I(gamma) = 0.1955 |
| rho* for lambda = 0, 0.25, 0.5, 0.75, 1 | 1.000, 0.469, 0.312, 0.289, 0.289 |
| rho_hat*_2000 for the same lambda | 0.736, 0.405, 0.268, 0.268, 0.268 |
| P(top token green given a tail token was emitted), spike level | 0.0530 (closed form), 0.0523 (Monte Carlo) |

`diagnostic.json` also records seven self-checks (outputs exist; E_n > 0 at lambda = 0; E_2000 within 5 % of the rate
at the finite-n threshold where -ln beta_2000 >= 100; the n = 10 000 exponent closer to the limit; rho_hat*_n within 0.05
of rho*_n; the NP test never weaker than the green count; the diagnostic reported). All pass at the defaults.

## Layout and workflow

Each block names the source its model or formula comes from and the file that implements it.

```
 watermark-edits/
 +-- README.md
 +-- pyproject.toml, uv.lock, .python-version   numpy, scipy, matplotlib; 3.13
 +-- code/exponent_greenlist.py                 main loop, self-checks, figure
 +-- code/wmsim/                                util.py, tokenmodel.py,
 |                                              attack.py, greencount.py,
 |                                              oraclenp.py, exponent.py
 +-- data/             created by the run, untracked: csv, json, png
 +-- results/figures/  created by the run, untracked: a copy of the figure

 question:  E(rho) = lim -(1/n) log beta_n at fixed alpha
            rho*   = the first rho with E(rho) = 0
         |
         v
 +-- TOKEN MODEL --------------------------------------------------------------+
 |  i.i.d. positions: uniform over V = 50 000 w.p. pi = 0.5,                  |
 |  otherwise top token 0.9 and the rest uniform.                             |
 |  lit:  Li et al. 2025 sec. 5.1 (synthetic spike distributions);            |
 |        Kirchenbauer et al. 2023 Thm 4.2 (entropy)                          |
 |  code: wmsim/tokenmodel.py                                                 |
 +----------------------------------------------------------------------------+
         |
         v
 +-- KEY AND LEAKAGE ----------------------------------------------------------+
 |  Fresh list per position, each token green w.p. gamma.                     |
 |  The attacker sees G_t w.p. lambda.                                        |
 |  lit:  Kirchenbauer et al. 2023 (scheme); Zhao et al. 2024 App. D          |
 |        (what hashing would add); Zhang et al. 2024, Jovanovic et al.       |
 |        2024 (list recovery from the API: the realistic lambda range)       |
 |  code: gamma enters wmsim/greencount.py and wmsim/oraclenp.py;             |
 |        keys are drawn only in wmsim/tokenmodel.py                          |
 +----------------------------------------------------------------------------+
         |
         v
 +-- GREEN PROBABILITY PER LEVEL ----------------------------------------------+
 |  q_H = 0.711 (uniform), q_L = 0.367 (spike);                               |
 |  diagnostic P(top token green | tail token emitted) = 0.053                |
 |  lit:  Kirchenbauer et al. 2023 sec. 3 (the tilt);                         |
 |        Jensen on g(x) = e^delta x / (e^delta x + 1 - x)                    |
 |  code: wmsim/tokenmodel.py                                                 |
 +----------------------------------------------------------------------------+
         |
         v
 +-- ATTACKER -----------------------------------------------------------------+
 |  Budget floor(rho n): leaked greens -> red (one edit each), then           |
 |  unleaked uniform positions, then spike; replacement green w.p.            |
 |  gamma where the list is unseen. The leaked-green count K is               |
 |  random and every law is the exact mixture over K.                         |
 |  lit:  Wu and Chandrasekaran 2024 Thm 1 (colour-agnostic                   |
 |        replacement)                                                        |
 |  code: wmsim/attack.py (allocation, per K value and at its mean);          |
 |        wmsim/greencount.py and wmsim/oraclenp.py (exact mixture over K)    |
 +----------------------------------------------------------------------------+
         |
         v
 +-- DETECTORS (both see the same attacked text) ------------------------------+
 |  Green count, exact in log space: null Bin(n, gamma), threshold            |
 |  at alpha; alternative = binomial convolutions mixed over K.               |
 |  lit:  Kirchenbauer et al. 2023 z-test; Fernandez et al. 2023              |
 |        (exact tests at low false-positive rates)                           |
 |  code: wmsim/greencount.py                                                 |
 |  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -   |
 |  Oracle NP, exact on the (greens_H, greens_L) grid: likelihood             |
 |  ratio that knows the attack rule and the levels, not the edited           |
 |  positions.                                                                |
 |  lit:  Neyman-Pearson lemma; Li et al. 2025 (detection                     |
 |        efficiency = exponent at fixed alpha)                               |
 |  code: wmsim/oraclenp.py                                                   |
 +----------------------------------------------------------------------------+
         |
         v
 +-- EXPONENT AND CRITICAL FRACTION -------------------------------------------+
 |  E_n = -ln beta_n / n at each n; Cramer rate at the finite-n               |
 |  threshold tau_n/n; KL lower bound.                                        |
 |  rho_hat*_n = where beta_n crosses 1/2;                                    |
 |  rho*_n = mean green rate crosses tau_n/n; rho* = the same at gamma.       |
 |  lit:  Cover and Thomas sec. 11.8 (Stein's lemma); Cramer's                |
 |        theorem; Hoeffding 1963 (KL form of the bound)                      |
 |  code: wmsim/exponent.py                                                   |
 +----------------------------------------------------------------------------+
         |
         v
 +-- OUTPUTS ------------------------------------------------------------------+
 |  data/results/<slug>/                                                      |
 |    beta.csv        per (n, lambda, rho): threshold, exact sizes,           |
 |                    log10 beta, class counts                                |
 |    exponents.csv   E_gc, E_np, I_n(tau_n/n), I_n(gamma), KL bound          |
 |    rhostar.csv     rho_hat*_n, rho*_n, rho* per (lambda, n)                |
 |    diagnostic.json q_H, q_L, spike diagnostic, self-checks, params         |
 |  data/figures/<slug>/exponent_vs_rho.png, copied to results/figures/       |
 |  code: exponent_greenlist.py                                               |
 +----------------------------------------------------------------------------+
```

## Conventions

- Deterministic computation everywhere except q_L and the diagnostic, which are seeded Monte Carlo with their standard
  errors reported.
- The threshold is calibrated to alpha on text written without the key (null = fresh random key), never assumed from a
  normal approximation; the realised sizes of both tests are written to beta.csv.
- The attack budget is exact: at most floor(rho n) substitutions, and beta.csv records the expected number used.
- Level and leak counts are fixed at their expectations; the key stays random, so the number of greens the attacker can
  see is binomial and every law is the exact mixture over it. Oracle-NP cells whose miss rate is exactly zero are
  reported as such, not as a small number.

## References

- Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., Goldstein, T. (2023). A watermark for large language
  models. ICML 2023, PMLR 202:17061-17084.
- Li, X., Ruan, F., Wang, H., Long, Q., Su, W. J. (2025). A statistical framework of watermarks for large language
  models: pivot, detection efficiency and optimal rules. Annals of Statistics 53(1):322-351.
- Wu, Q., Chandrasekaran, V. (2024). Bypassing LLM watermarks with color-aware substitutions. ACL 2024, 8549-8581.
- Zhao, X., Ananth, P., Li, L., Wang, Y.-X. (2024). Provable robust watermarking for AI-generated text. ICLR 2024.
- Fernandez, P., Chaffin, A., Tit, K., Chappelier, V., Furon, T. (2023). Three bricks to consolidate watermarks for
  large language models. IEEE WIFS 2023.
- Zhang, Z. et al. (2024). Large language model watermark stealing with mixed integer programming. ACSAC 2024, 46-60.
- Jovanovic, N., Staab, R., Vechev, M. (2024). Watermark stealing in large language models. ICML 2024.
- Cover, T. M., Thomas, J. A. (2006). Elements of Information Theory, 2nd ed., sec. 11.8 (Chernoff-Stein lemma).
- Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables. JASA 58(301):13-30.
