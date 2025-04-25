import numpy as np
from .membership import fuzzify
from ..config import MF_POINTS, BRIGHT_RANGE, COLOR_TERMS
from .rules import RULES

class Controller:
    def __init__(self):
        # store brightness membership definitions
        self.out_mf = MF_POINTS["brightness"]

    def compute(self, lux, hour, presence, eco):
        # fuzzify each input variable
        inputs = {}
        inputs["lux"] = fuzzify("lux", lux, MF_POINTS)
        inputs["hour"] = fuzzify("hour", hour, MF_POINTS)
        inputs["presence"] = fuzzify("presence", float(presence), MF_POINTS)
        inputs["eco"] = fuzzify("eco", float(eco), MF_POINTS)

        # init aggregated degrees
        agg_bright = {}
        for term in self.out_mf:
            agg_bright[term] = 0.0
        agg_color = {}
        for term in COLOR_TERMS:
            agg_color[term] = 0.0

        # evaluate each rule
        for rule in RULES:
            deg = 1.0
            for var, term in rule["if"].items():
                val = inputs[var][term]
                if val < deg:
                    deg = val
            # update consequents
            for var, term in rule["then"].items():
                if var == "brightness" and deg > agg_bright[term]:
                    agg_bright[term] = deg
                if var == "color_temp" and deg > agg_color[term]:
                    agg_color[term] = deg

        # defuzzify brightness and pick color
        brightness = self.defuzz_centroid(agg_bright)
        best_color = None
        best_deg = -1.0
        for term, level in agg_color.items():
            if level > best_deg:
                best_deg = level
                best_color = term

        return brightness, best_color

    def defuzz_centroid(self, levels):
        # build x axis
        start, stop = BRIGHT_RANGE
        N = 201
        x = np.linspace(start, stop, N)
        mu = np.zeros(N)

        # aggregate each term’s shape
        for term, level in levels.items():
            pts = self.out_mf[term]
            if len(pts) == 3:
                a, b, c = pts
                for i in range(N):
                    xi = x[i]
                    if xi <= a or xi >= c:
                        val = 0.0
                    elif xi == b:
                        val = 1.0
                    elif xi < b:
                        val = (xi - a) / (b - a)
                    else:
                        val = (c - xi) / (c - b)
                    val = val * level
                    if val > mu[i]:
                        mu[i] = val
            else:
                a, b, c, d = pts
                for i in range(N):
                    xi = x[i]
                    if xi <= a or xi >= d:
                        val = 0.0
                    elif xi <= b:
                        val = (xi - a) / (b - a) * level
                    elif xi <= c:
                        val = level
                    else:
                        val = (d - xi) / (d - c) * level
                    if val > mu[i]:
                        mu[i] = val

        # compute centroid
        total = 0.0
        weighted = 0.0
        for i in range(N):
            weighted += x[i] * mu[i]
            total += mu[i]
        if total == 0.0:
            return 0.0
        return weighted / total