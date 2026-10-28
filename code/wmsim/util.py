"""Shared helpers: binomial pmf, zero padding, log-space convolution of pmfs."""
import numpy as np
from scipy.stats import binom


def pmf(N, p):
    return binom.pmf(np.arange(N + 1), N, p)


def pad(v, size):
    out = np.zeros(size)
    out[: len(v)] = v
    return out


def logconv(a, b):
    """Log of the convolution of two pmfs given in log space (direct sum, no floor)."""
    if len(a) < len(b):
        a, b = b, a
    out = np.full(len(a) + len(b) - 1, -np.inf)
    for j, bj in enumerate(b):
        if bj > -np.inf:
            seg = out[j: j + len(a)]
            np.logaddexp(seg, a + bj, out=seg)
    return out
