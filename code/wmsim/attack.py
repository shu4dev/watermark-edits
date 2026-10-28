"""Attacker: the budget allocation across classes, and the class weights with K at its mean."""
import numpy as np


def allocate(B, k, UH, UL):
    """Budget B against k leaked greens (K at its mean, or an array of K values): flip min(B, k) leaked greens,
    then edit unleaked uniform, then unleaked spike positions. Returns (flips, uniform edits, spike edits)."""
    r = np.maximum(B - k, 0)
    eH = np.minimum(r, UH)
    eL = np.minimum(r - eH, UL)
    return np.minimum(B, k), eH, eL


def class_weights(fr, rho, qH, qL):
    """Class fractions with K at its mean, fr = (L_H, L_L, U_H, U_L) / n, rho = B / n: weights of
    q' = 1 (leaked unflipped), gamma (edited), q_H, q_L (unedited); the rest (flipped) has q' = 0."""
    LH, LL, UH, UL = fr
    k = LH * qH + LL * qL
    flip, eH, eL = allocate(rho, k, UH, UL)
    return np.array([k - flip, eH + eL, UH - eH, UL - eL])
