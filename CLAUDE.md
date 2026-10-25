# CLAUDE.md

## What this is
The empirical side of the CS578 project "Text watermark detection under edits" (docs/proposal.md, docs/proposal.pdf).
Green-list watermark (Kirchenbauer-style): green fraction γ, soft bias δ, key = independent green lists G_t. Attacker:
at most ρn substitutions from permitted sets R(y_t), key-blind (knows γ, δ, p_t), later λ-leakage of G_t per position.
Detector: knows the key, sees the edited z_1:n, false-positive rate ≤ α on any key-independent sequence. Theory target:
the minimax false-negative exponent E(ρ) = liminf −(1/n) log β*_n(α, ρ) and the critical edit fraction ρ* where it hits 0.
The code simulates detector–attack pairs on token models and estimates β_n(α, ρ) and its exponent, so the paper's
bounds (green-count test vs Neyman–Pearson, priority-rule attack vs others, i.i.d. vs autoregressive) are checked
against numbers.

## Research workflow (repo)
- Files: research/ (lit notes), plans/ (plans), results/ (run notes). Templates: research/TEMPLATE.md, plans/TEMPLATE.md.
- Roles: the Claude session explores, grills, writes research/ and plans/. experiment-runner turns one plan into code and
  runs it once; refactorer does structural changes. Invoke each with the plan path.
- Code under code/: one script per plan slug plus a small shared package (code/wmsim/) once two scripts share something.
  Experiment code, not a library: flags with defaults, no config files, no abstractions a plan did not ask for.
- Outputs under data/ (gitignored): data/results/<slug>/ for arrays and tables, data/figures/<slug>/ for plots.
  A figure a results note cites is copied to results/figures/<slug>-<name>.png and tracked.
- Runs: synthetic token models (i.i.d. or Markov p_t over a vocabulary) run on the Mac (M2, 16 GB, numpy + scipy,
  no GPU). Anything with a real LM (torch, transformers) is a later round on a GPU box; do not add torch until a plan
  needs it.
- Git is handled by the user: no branches, commits or pushes from sessions; end each phase with a suggested commit
  message (imperative, ~50 chars, no AI attribution lines).

## Commands
```bash
uv sync                                  # env; .python-version pins 3.13
uv run python code/<script>.py --help    # every script has flags for V, γ, δ, ρ, α, n, seed
uv run python -m pytest code/tests -q    # only when a plan added tests
```

## Conventions
- Seeded everything; every number in a results note names its command and seed.
- False-negative rates with Clopper–Pearson 95 % intervals; an exponent is the slope of −log β_n against n with the
  n range and the fit stated; β_n below about 1/trials is "not resolved", not 0.
- The detector's threshold is calibrated to α on key-independent text (null = fresh random key), never assumed.
- Attack budgets are exact: an attack uses at most ⌊ρn⌋ substitutions, and the note says how many it used.
