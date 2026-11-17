"""Single-position law of main.tex section 3 and the five instances A, B, C1, C2, C3 of the report checks."""
import itertools

import numpy as np

LABELS = {"A": "A: m = 2, |G| = 1, uniform p", "B": "B: m = 4, |G| = 1, uniform p",
          "C1": "C1: m = 4, Bernoulli, uniform p", "C2": "C2: m = 4, Bernoulli, $p_0$ = 0.9",
          "C3": "C3: m = 4, Bernoulli, Dirichlet p"}


def law(m, gamma, delta, p, k=None):
    """Single-position law; k None: Bernoulli membership, else lists of exactly k tokens, uniform."""
    G = np.array([s for s in itertools.product((0, 1), repeat=m) if k is None or sum(s) == k], float)
    size = G.sum(1)
    mu = gamma ** size * (1 - gamma) ** (m - size) if k is None else np.full(len(G), 1 / len(G))
    assert np.isclose(mu.sum(), 1) and np.allclose(mu @ G, gamma)  # P(v in G) = gamma for every v
    N = 1 + np.expm1(delta) * (G @ p)
    PYG = p * np.exp(delta * G) / N[:, None]
    PGY = mu[:, None] * PYG
    return {"m": m, "gamma": gamma, "delta": delta, "p": p, "k": k, "G": G, "mu": mu, "N": N, "PYG": PYG,
            "PGY": PGY, "PY": PGY.sum(0), "q": float((PGY * G).sum())}


def instances(rng):
    """A (m 2, k 1, e^delta 3, uniform p), B (m 4, k 1, delta 2, uniform p), C1-C3 (m 4, Bernoulli, gamma 0.25,
    delta 2; p uniform, (0.9, 1/30, 1/30, 1/30), Dirichlet(1,1,1,1)). C3's p is the first draw from rng."""
    unif4 = np.full(4, 0.25)
    return {"A": law(2, 0.5, np.log(3.0), np.full(2, 0.5), k=1), "B": law(4, 0.25, 2.0, unif4, k=1),
            "C1": law(4, 0.25, 2.0, unif4), "C2": law(4, 0.25, 2.0, np.array([0.9, 1 / 30, 1 / 30, 1 / 30])),
            "C3": law(4, 0.25, 2.0, rng.dirichlet(np.ones(4)))}
