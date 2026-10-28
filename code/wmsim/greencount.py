"""Green-count detector, exact in log space."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom

from .attack import allocate


def threshold(n, gamma, alpha):
    """Smallest tau with P(Bin(n, gamma) >= tau) <= alpha, and that exact size."""
    t = np.arange(n + 2)
    sf = binom.sf(t - 1, n, gamma)
    tau = int(t[np.argmax(sf <= alpha)])
    return tau, float(sf[tau])


def gc_tables(UH, UL, qH, qL, gamma, tau):
    """Unleaked positions in attack order (UH uniform, then UL spike); with b of them edited the green count
    there is R(b) = Bin(b, gamma) + sum of the untouched suffix. Returns log CDF of R(0) and
    log P(R(b) <= tau - 1) for b = 0..U."""
    U = UH + UL
    p = np.r_[np.full(UH, qH), np.full(UL, qL)]
    T = np.full((U + 1, U + 1), -np.inf)  # T[b] = log pmf of sum_{i >= b} Bern(p_i)
    T[U, 0] = 0.0
    for b in range(U - 1, -1, -1):
        T[b] = T[b + 1] + np.log1p(-p[b])
        np.logaddexp(T[b, 1:], T[b + 1, :-1] + np.log(p[b]), out=T[b, 1:])
    logF = np.logaddexp.accumulate(T, axis=1)
    a = np.arange(tau)
    LA = binom.logpmf(a[None, :], np.arange(U + 1)[:, None], gamma)
    h = logsumexp(LA + logF[:, np.minimum(tau - 1 - a, U)], axis=1)
    return logF[0], h


def log_beta_gc(logPM, B, logF0, h, tau, UH, UL):
    """log P1(S <= tau - 1); the green count depends on the leaked greens only through M = K_H + K_L."""
    U = len(h) - 1
    m = np.arange(len(logPM))
    x = tau - 1 - (m - B)                       # M >= B: S = (M - B) + R(0)
    hm = np.where(x >= 0, logF0[np.clip(x, 0, U)], -np.inf)
    lo = m < B                                  # M < B: S = R(B - M), at most U edits land
    _, eH, eL = allocate(B, m[lo], UH, UL)
    hm[lo] = h[eH + eL]
    return float(logsumexp(logPM + hm))
