from __future__ import annotations

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 1. оголошуємо нечіткі змінні та функції належності

_days = ctrl.Antecedent(np.arange(0, 31, 1), "days_left")
_stock = ctrl.Antecedent(np.arange(0, 3.1, 0.1), "stock_ratio")
_urgency = ctrl.Consequent(np.arange(0, 1.01, 0.01), "urgency")

# days_left
_days["critical"] = fuzz.trapmf(_days.universe, [0, 0, 2, 5])
_days["soon"] = fuzz.trimf(_days.universe, [2, 7, 14])
_days["safe"] = fuzz.trapmf(_days.universe, [10, 15, 30, 30])

# stock_ratio
_stock["low"] = fuzz.trapmf(_stock.universe, [0, 0, 0.2, 0.6])
_stock["ok"] = fuzz.trimf(_stock.universe, [0.4, 1.0, 1.6])
_stock["high"] = fuzz.trapmf(_stock.universe, [1.2, 2.0, 3.0, 3.0])

# urgency
_urgency["low"] = fuzz.trimf(_urgency.universe, [0.0, 0.1, 0.3])
_urgency["medium"] = fuzz.trimf(_urgency.universe, [0.2, 0.5, 0.8])
_urgency["high"] = fuzz.trapmf(_urgency.universe, [0.6, 0.8, 1.0, 1.0])

# 2. база правил

RULES = [
    ctrl.Rule(_days["critical"] | _stock["low"], _urgency["high"]),
    ctrl.Rule(_days["soon"] & _stock["ok"], _urgency["medium"]),
    ctrl.Rule(_days["safe"] & _stock["high"], _urgency["low"]),
    ctrl.Rule(_days["soon"] & _stock["high"], _urgency["low"]),
    ctrl.Rule(_days["safe"] & _stock["low"], _urgency["medium"]),
]

_system = ctrl.ControlSystem(RULES)
_simulator = ctrl.ControlSystemSimulation(_system)



# 3. API-функції

def evaluate(days_left: float, stock_ratio: float) -> float:

    sim = ctrl.ControlSystemSimulation(_system)  # локальний, потокобезпечний
    sim.input["days_left"] = max(0.0, min(30.0, float(days_left)))
    sim.input["stock_ratio"] = max(0.0, min(3.0, float(stock_ratio)))
    sim.compute()
    return float(sim.output.get("urgency", 0.0))


def produce_now(days_left: float, stock_ratio: float, thr: float = 0.6) -> bool:
    #повертає True, якщо urgency ≥ thr (за замовчуванням 0.6).
    return evaluate(days_left, stock_ratio) >= thr


__all__ = ["evaluate", "produce_now"]  