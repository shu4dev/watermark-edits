# watermark-edits

Simulations for the CS578 project **Text watermark detection under edits** (Pham, Shu, Wanduka, Manganaris):
green-list LLM watermarks against a budgeted, key-blind substitution attacker, measured as the false-negative
exponent E(rho) and the critical edit fraction rho*. Proposal: `docs/proposal.md`. Workflow and conventions: `CLAUDE.md`.
CPU-only rounds (numpy/scipy) run on this Mac through the experiment-runner; GPU rounds need a named compute box.

## Repo structure

```
watermark-edits/
+-- CLAUDE.md                    what this is, roles (session plans / runner implements), conventions
+-- README.md                    this page
+-- pyproject.toml               numpy, scipy, matplotlib; dev: pytest      .python-version = 3.13
+-- .gitignore                   data/, .venv/, .claude/
|
+-- docs/
|   +-- proposal.pdf, proposal.md     the CS578 proposal: setting, E(rho), rho*, approach, 6 refs
|
+-- research/                    lit notes, written by the Claude session
|   +-- TEMPLATE.md
|   +-- 2026-10-25-greenlist-substitution-exponent-litreview.md   ARS lit review, 33 verified sources
|   +-- 2026-10-25-greenlist-substitution-exponent.md             question, what's known, gap, picked idea
|
+-- plans/                       one plan per experiment, < 40 lines, written by the session after grilling
|   +-- TEMPLATE.md
|   +-- 2026-10-25-greenlist-substitution-exponent.md             round 1: exact exponent, two-level i.i.d.
|
+-- results/                     run notes, written by the experiment-runner (<= 15 lines each)
|   +-- figures/<slug>-<name>.png     curated copies of figures a note cites
|
+-- code/                        experiment code, written by the runner from a plan
|   +-- exponent_greenlist.py         (planned) one script per plan slug, flags with defaults
|   +-- wmsim/                        (later) shared package once two scripts share something
|
+-- data/                        gitignored, lives on the compute box
    +-- results/<slug>/               csv, json
    +-- figures/<slug>/               png

   research/<slug>.md  --grill-->  plans/<slug>.md  --runner-->  code/ + data/  -->  results/<slug>.md
          ^                                                                               |
          +----------------------- next smallest step (new plan, From: results/) <--------+
```

## Experiment workflow (plans/2026-10-25-greenlist-substitution-exponent.md)

```
 TOKEN MODEL (i.i.d., V = 50k)                           KEY AND LEAKAGE
 +----------------------------------+                    +-----------------------------------+
 | position is                      |                    | G_t: each token green w.p. gamma  |
 |  high-entropy w.p. pi (uniform)  |                    | (independent lists per position)  |
 |  low-entropy  otherwise          |                    | attacker sees G_t w.p. lambda     |
 |  (top token 0.9, rest uniform)   |                    +-----------------+-----------------+
 +----------------+-----------------+                                      |
                  |                                                        |
                  v                                                        |
 +----------------------------------+   per-level green posterior          |
 | q_H = g e^d / (g e^d + 1 - g)    |   (prob the watermarked token        |
 | q_L = Monte Carlo over G, 1e5    |    is green, before any edit)        |
 | diagnostic: P(top in G | tail y) |                                      |
 +----------------+-----------------+                                      |
                  |                                                        |
                  v                                                        v
 +------------------------------------------------------------------------------------+
 | ATTACKER  budget floor(rho n); knows gamma, delta, p_t, text; never the key         |
 |   1. leaked GREEN tokens first     -> forced red; a leaked red costs no edit       |
 |   2. then highest q_t (high-ent.)  -> replacement colour Bern(gamma)               |
 |   output: per-level, per-leak-status substitution counts  ->  q'_t per position    |
 +----------------------------+---------------------------------+---------------------+
                              |                                 |
                              v                                 v
 +------------------------------------------+   +----------------------------------------+
 | GREEN COUNT  (exact)                     |   | ORACLE NP  (exact)                     |
 | null : Bin(n, gamma) -> tau_n at alpha   |   | knows rho, lambda, pi, each position's |
 | alt  : class binomials, mixed over K     |   | level; NOT the edited positions        |
 |        (log space, no beta floor)        |   | LR on the (greens_H, greens_L) grid:   |
 | n in {100..2000}; 10000 at lambda=0      |   | alt: mixture over leaked greens; exact |
 +--------------------+---------------------+   +-------------------+--------------------+
                      |                                             |
                      v                                             |
 +------------------------------------------+                       |
 | E_n(rho) = -ln beta_n / n at each n       |                       |
 |          rate at tau_n/n (lambda=0 only)  |                       |
 |          KL(Bern g || Bern qbar_rho) bound|                       |
 | rho^*_n  where beta_n crosses 1/2         |                       |
 | rho*_n   mean rate = tau_n/n; rho* at g   |                       |
 +--------------------+---------------------+                       |
                      |                                             |
                      v                                             v
 +------------------------------------------------------------------------------------+
 | OUTPUTS  data/results/<slug>/{beta,exponents,rhostar}.csv, diagnostic.json           |
 |          data/figures/<slug>/exponent_vs_rho.png  (one panel per lambda)            |
 |          -> results/figures/<slug>-exponent_vs_rho.png, results/<slug>.md           |
 | WORKS WHEN  lambda=0: E_n > 0 at every rho; E_2000 within 5% of rate at tau_n/n     |
 |             rho^*_n within 0.05 of rho*_n at every lambda and n                     |
 |             oracle NP never below green count; diagnostic numbers given              |
 +------------------------------------------------------------------------------------+

 grid: gamma 0.25, delta 2, alpha 1e-3, pi 0.5, rho in {0, 0.05, ..., 0.95}, lambda in {0, .25, .5, .75, 1}, seed 0
```

- Setup on the box: `uv sync`; run any script with `uv run python code/<script>.py --help`.
