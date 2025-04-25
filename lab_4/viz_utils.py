import matplotlib.pyplot as plt
import numpy as np

from .config import LUX_RANGE, HOUR_RANGE, BRIGHT_RANGE, MF_POINTS
from .fuzzy.membership import tri_mf, trap_mf

def plot_mf(var, out_path=None):
    # plot membership functions for variable
    if var not in MF_POINTS:
        raise KeyError(f"unknown var '{var}'")
    pts = MF_POINTS[var]
    # select range
    if var == 'lux':
        rng = LUX_RANGE
    elif var == 'hour':
        rng = HOUR_RANGE
    else:
        rng = BRIGHT_RANGE
    x = np.linspace(rng[0], rng[1], 400)
    # draw each mf
    for label, p in pts.items():
        if len(p) == 3:
            y = tri_mf(x, p[0], p[1], p[2])
        else:
            y = trap_mf(x, p[0], p[1], p[2], p[3])
        plt.plot(x, y, label=label)
    plt.ylim(-0.05, 1.05)
    plt.title(f"mf - {var}")
    plt.xlabel(var)
    plt.legend()
    if out_path:
        plt.savefig(out_path, dpi=150)
    else:
        plt.show()
    plt.clf()

def plot_results(records, out_path=None):
    # scatter brightness vs lux
    x = []
    y = []
    for r in records:
        x.append(r['lux'])
        y.append(r['brightness'])
    plt.scatter(x, y)
    plt.xlabel('lux')
    plt.ylabel('brightness %')
    if out_path:
        plt.savefig(out_path, dpi=150)
    else:
        plt.show()
    plt.clf()