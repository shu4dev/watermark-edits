"""Oracle NP detector, exact in linear space on the (S_H, S_L) grid."""
import numpy as np
from scipy.stats import binom

from .attack import allocate
from .util import pad, pmf


def np_tables(nH, nL, LH, LL, UH, UL, qH, qL, gamma):
    pKH, pKL = pmf(LH, qH), pmf(LL, qL)
    t = dict(pKH=pKH, pKL=pKL, pM=np.convolve(pKH, pKL), XH=pmf(UH, qH), XL=pmf(UL, qL), GH=pmf(UH, gamma))
    t["cL1"] = np.convolve(pKL, t["XL"])
    t["YH"] = np.array([np.convolve(pmf(b, gamma), pmf(UH - b, qH)) for b in range(UH + 1)])  # b uniform edits
    t["ZL"] = np.array([np.convolve(pmf(j, gamma), pmf(UL - j, qL)) for j in range(UL + 1)])  # j spike edits
    t["logP0"] = (binom.logpmf(np.arange(nH + 1), nH, gamma)[:, None]
                  + binom.logpmf(np.arange(nL + 1), nL, gamma)[None, :])
    t["P0"] = np.exp(t["logP0"])
    return t


def np_alt(t, B, nH, nL, LH, UH, UL):
    """P1(S_H, S_L): exact mixture over (K_H, K_L), which collapses to four outer products."""
    pKH, pKL, pM = t["pKH"], t["pKL"], t["pM"]
    L = len(pM) - 1
    P1 = np.zeros((nH + 1, nL + 1))
    if B <= LH:  # K_H >= B: whole budget on uniform-level leaked greens
        P1 += np.outer(pad(np.convolve(pKH[B:], t["XH"]), nH + 1), t["cL1"])
    if B >= 1:
        low = np.convolve(pKH[:B], pKL)  # law of M on {K_H < B}
        if len(low) > B:  # K_H < B <= M: the rest on spike-level leaked greens
            P1 += np.outer(pad(t["XH"], nH + 1), pad(np.convolve(low[B:], t["XL"]), nL + 1))
        b = np.arange(max(1, B - L), B + 1)  # M < B: every leaked green flipped, b = B - M edits left
        w = pM[B - b]
        _, eH, eL = allocate(B, B - b, UH, UL)
        s = b <= UH  # leftover lands on unleaked uniform positions
        if s.any():
            P1 += np.outer(pad(w[s] @ t["YH"][eH[s]], nH + 1), pad(t["XL"], nL + 1))
        if (~s).any():  # uniform level exhausted, the rest on unleaked spike positions
            j = eL[~s]
            P1 += np.outer(pad(t["GH"], nH + 1),
                           pad(np.bincount(j, weights=w[~s], minlength=UL + 1) @ t["ZL"], nL + 1))
    return P1


def np_test(P1, t, alpha):
    """Non-randomised NP: reject cells by decreasing P1/P0 until the null mass would exceed alpha."""
    with np.errstate(divide="ignore"):
        llr = np.log(P1) - t["logP0"]
    order = np.argsort(-llr, axis=None, kind="stable")
    cum = np.cumsum(t["P0"].ravel()[order])
    k = int(np.searchsorted(cum, alpha, side="right"))
    return float(P1.ravel()[order[k:]].sum()), (float(cum[k - 1]) if k else 0.0)
