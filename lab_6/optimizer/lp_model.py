from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

try:
    import pulp
except ModuleNotFoundError as e:  # pragma: no cover
    raise ImportError(
        "lp_model потребує встановленого pulp. "
        "pip install pulp"
    ) from e


def optimize_lp(
    demand_df: pd.DataFrame,
    products_df: pd.DataFrame,
    materials_df: pd.DataFrame,
    bom_df: pd.DataFrame,
    *,
    shift_minutes: int = 8 * 60,
) -> Tuple[pd.DataFrame, float]:
    """Повертає (plan_df, obj_val)."""

    skus = products_df.sku.tolist()
    need_map = demand_df.set_index("sku")["need_to_cover"]

    # моделюємо
    prob = pulp.LpProblem("Daily_Production_Plan", pulp.LpMaximize)

    qty = {s: pulp.LpVariable(f"make_{s}", lowBound=0, cat="Integer") for s in skus}

    # ціль: прибуток
    base_profit = (
        products_df.set_index("sku")[["unit_price", "unit_cost"]]
        .eval("unit_price - unit_cost")
    )
    urgency = demand_df.set_index("sku")["urgency"]
    profit_coeff = ((base_profit * (1 + urgency)).fillna(base_profit)).to_dict()
    prob += pulp.lpSum(profit_coeff[s] * qty[s] for s in skus)

    # попит
    for s in skus:
        prob += qty[s] <= int(need_map.get(s, 0))

    # час
    time_coeff = products_df.set_index("sku")["prod_time_min"].to_dict()
    prob += pulp.lpSum(time_coeff[s] * qty[s] for s in skus) <= shift_minutes, "Shift"

    # матеріали
    mat_ids = materials_df.material_id.tolist()
    for m_id in mat_ids:
        stock = float(
            materials_df.loc[materials_df.material_id == m_id, "stock_kg"].iloc[0]
        )
        mat_qty = {
            row.sku: row.qty_per_unit
            for row in bom_df[bom_df.material_id == m_id].itertuples()
        }
        prob += (
            pulp.lpSum(mat_qty.get(s, 0.0) * qty[s] for s in skus) <= stock,
            f"Mat_{m_id}",
        )

    # розв’язуємо
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    plan_qty = {s: int(v.value()) for s, v in qty.items() if v.value() and v.value() > 0.5}
    plan_df = (
        pd.DataFrame.from_dict(plan_qty, orient="index", columns=["qty_produce"])
        .reset_index()
        .rename(columns={"index": "sku"})
    )
    plan_df["unit_profit"] = plan_df["sku"].map(profit_coeff)
    plan_df["est_profit"] = plan_df["unit_profit"] * plan_df["qty_produce"]

    return plan_df, pulp.value(prob.objective)