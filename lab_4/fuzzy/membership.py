from __future__ import annotations
import numpy as np

__all__ = ["tri_mf", "trap_mf", "fuzzify"]

def tri_mf(x, a, b, c):
    # triangle mf
    x_arr = np.asarray(x)
    mu = np.zeros_like(x_arr, dtype=float)
    rising = (a < x_arr) & (x_arr <= b)
    mu[rising] = (x_arr[rising] - a) / (b - a + 1e-9)
    falling = (b < x_arr) & (x_arr < c)
    mu[falling] = (c - x_arr[falling]) / (c - b + 1e-9)
    mu[x_arr == b] = 1.0
    return float(mu) if mu.ndim == 0 else mu

def trap_mf(x, a, b, c, d):
    # trapezoid mf
    x_arr = np.asarray(x)
    mu = np.zeros_like(x_arr, dtype=float)
    rising = (a < x_arr) & (x_arr <= b)
    mu[rising] = (x_arr[rising] - a) / (b - a + 1e-9)
    top = (b < x_arr) & (x_arr <= c)
    mu[top] = 1.0
    falling = (c < x_arr) & (x_arr < d)
    mu[falling] = (d - x_arr[falling]) / (d - c + 1e-9)
    return float(mu) if mu.ndim == 0 else mu

def fuzzify(var, crisp, mf_points):
    # map term to membership value
    pts = mf_points[var]
    res: dict[str, float] = {}
    for term, params in pts.items():
        if len(params) == 3:
            m = tri_mf(crisp, *params)
        else:
            m = trap_mf(crisp, *params)
        res[term] = float(m)
    return res