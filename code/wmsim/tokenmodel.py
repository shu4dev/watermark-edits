"""Token model: level and leak counts, per-level green probabilities, spike diagnostic."""
import numpy as np

TOP = 0.9           # spike level: probability of the top token


def green_prob(x, delta):
    """P(emitted token is green | p_t puts mass x on G_t) under the soft watermark."""
    e = np.exp(delta)
    return e * x / (1.0 + (e - 1.0) * x)


def level_counts(n, lam, pi):
    nH = int(np.floor(pi * n + 1e-9))
    nL = n - nH
    LH, LL = int(np.floor(lam * nH + 1e-9)), int(np.floor(lam * nL + 1e-9))
    return nH, nL, LH, LL, nH - LH, nL - LL


def q_levels(gamma, delta, V, trials, rng):
    qH = float(green_prob(gamma, delta))  # closed form (|G_t|/V -> gamma)
    top_green = rng.random(trials) < gamma
    k = rng.binomial(V - 1, gamma, size=trials)
    q = green_prob(TOP * top_green + (1 - TOP) * k / (V - 1), delta)  # P(green | key), per key draw
    qL_formula = gamma * green_prob(TOP + (1 - TOP) * gamma, delta) + (1 - gamma) * green_prob((1 - TOP) * gamma, delta)
    return qH, float(q.mean()), float(q.std(ddof=1) / np.sqrt(trials)), float(qL_formula)


def spike_diagnostic(gamma, delta, V, trials, rng):
    """P(top token in G_t | a tail token was emitted) at the spike level: closed form and Monte Carlo."""
    e = np.exp(delta)
    c = gamma * e + 1 - gamma                                 # tail reweighting at tail green fraction gamma
    Zg, Zr = TOP * e + (1 - TOP) * c, TOP + (1 - TOP) * c    # Z_t when the top token is green / red
    p_closed = gamma * Zr / (gamma * Zr + (1 - gamma) * Zg)
    top_green = rng.random(trials) < gamma
    k = rng.binomial(V - 1, gamma, size=trials)
    tail = (1 - TOP) * (e * k + (V - 1 - k)) / (V - 1)      # unnormalised watermarked tail mass
    topm = TOP * np.where(top_green, e, 1.0)
    tail_emitted = rng.random(trials) < tail / (tail + topm)  # one emitted token per key draw
    N = int(tail_emitted.sum())
    p_mc = float(top_green[tail_emitted].mean())
    return dict(p_top_green_given_tail_closed=float(p_closed), p_top_green_given_tail_mc=p_mc,
                p_top_green_given_tail_mc_se=float(np.sqrt(p_mc * (1 - p_mc) / N)), mc_keys=trials,
                mc_tail_emissions=N, red_prob_bayes_swap_closed=float(1 - p_closed),
                red_prob_bayes_swap_mc=1 - p_mc, red_prob_keyblind_replacement=1 - gamma,
                Z_top_green=float(Zg), Z_top_red=float(Zr),
                p_tail_emitted_closed=float(gamma * (1 - TOP) * c / Zg + (1 - gamma) * (1 - TOP) * c / Zr))
