"""Rate function, KL bound and rho*."""
import numpy as np
from scipy.optimize import brentq

from .attack import class_weights


def rate(w, t, gamma, qH, qL):
    """I(t) = sup_{theta <= 0} [theta t - sum_c w_c log(1 - q_c + q_c e^theta)] over classes q_c = 1, gamma, q_H, q_L."""
    probs = np.array([gamma, qH, qL])
    w1, wr = w[0], w[1:]
    if w1 + wr @ probs <= t:
        return 0.0
    if w1 >= t:  # surviving leaked greens alone reach t n
        return np.inf
    dLam = lambda th: w1 + wr @ (probs * np.exp(th) / (1 - probs + probs * np.exp(th)))
    lo = -1.0
    while dLam(lo) > t:
        lo *= 2
    th = brentq(lambda x: dLam(x) - t, lo, 0.0, xtol=1e-14)
    return float(th * t - (w1 * th + wr @ np.log1p(probs * np.expm1(th))))


def kl_bern(a, b):
    """D(Bern(a) || Bern(b)) when b > a, else 0."""
    return float(a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))) if b > a else 0.0


def rho_star(fr, target, gamma, qH, qL):
    """Root in rho of the K-at-mean green rate (piecewise linear) minus target; 1.0 if it never drops below."""
    f = lambda r: class_weights(fr, r, qH, qL) @ np.array([1.0, gamma, qH, qL]) - target
    return 1.0 if f(1.0) >= -1e-12 else float(brentq(f, 0.0, 1.0, xtol=1e-12))
