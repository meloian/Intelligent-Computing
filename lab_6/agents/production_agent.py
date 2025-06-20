from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from rules.fuzzy_rules import evaluate as fuzzy_urgency
from optimizer.ga_optimizer import optimize_ga

try:
    from optimizer.lp_model import optimize_lp

    _HAS_LP = True
except ImportError:
    _HAS_LP = False



# RESULT TYPE

@dataclass(slots=True)
class ProductionPlan:
    date: str
    df_plan: pd.DataFrame  # sku, qty_produce, est_profit
    objective_value: float



# AGENT

class ProductionPlannerAgent:
    SHIFT_MINUTES = 8 * 60          # довжина виробничої зміни
    DEMAND_FACTOR = 1.10            # множник «звичний попит × 1.1»
    DEMAND_FLOOR  = 25              # попит, коли магазини зовсім порожній

    def __init__(
        self,
        data_dir: str | Path = "data",
        *,
        algo: Literal["lp", "ga", "auto"] = "auto",
    ):

        data_dir = Path(data_dir)
        self.products = pd.read_csv(data_dir / "products.csv")
        self.materials = pd.read_csv(data_dir / "materials.csv")
        self.bom = pd.read_csv(data_dir / "bom.csv")

        self.mat_lookup = self.bom.pivot_table(
            index="sku", columns="material_id", values="qty_per_unit", fill_value=0.0
        )

        # алгоритм
        if algo == "auto":
            self.algo = "lp" if _HAS_LP else "ga"
        else:
            self.algo = algo

    def handle_sales_report(self, report) -> ProductionPlan:
        demand_df = self._forecast_demand(report.df)

        if self.algo == "lp" and _HAS_LP:
            plan_df, obj = optimize_lp(
                demand_df,
                self.products,
                self.materials,
                self.bom,
                shift_minutes=self.SHIFT_MINUTES,
            )

        elif self.algo == "ga":
            plan_df, obj = optimize_ga(
                demand_df,
                self.products,
                self.materials,
                self.bom,
                shift_minutes=self.SHIFT_MINUTES,
            )

        else:  # greedy fallback
            plan_df, obj = self._heuristic_plan(demand_df.copy(deep=True))

        return ProductionPlan(date=report.date, df_plan=plan_df, objective_value=obj)

    # 1. Прогноз попиту  +  нечітка «urgency»

    def _forecast_demand(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        df = sales_df[["sku", "quantity_sold", "stock_left"]].copy()
        sold = df["quantity_sold"]
        df["pred_demand"] = np.where(
            (sold == 0) & (df["stock_left"] == 0),
            self.DEMAND_FLOOR,
            sold * self.DEMAND_FACTOR,
        ).round()

        # оцінка «днів до закінчення» ~ пропорційно запасу / попиту
        prod_shelf = self.products.set_index("sku")["shelf_life_days"]
        df["days_left_est"] = (
            prod_shelf.reindex(df.sku).to_numpy()
            * df["stock_left"]
            / (df["pred_demand"].replace(0, 1))
        ).clip(0, prod_shelf.max())

        # ratio запасу до прогноз. попиту
        df["stock_ratio"] = df["stock_left"] / df["pred_demand"].replace(0, 1)

        # нечітка терміновість
        df["urgency"] = df.apply(
            lambda r: fuzzy_urgency(r.days_left_est, r.stock_ratio), axis=1
        )

        # базова потреба
        need = np.maximum(df["pred_demand"] - df["stock_left"], 0).astype(int)

        # підсилюємо потребу пропорційно urgency (до +100 %)
        df["need_to_cover"] = (need * (1 + df["urgency"])).round().astype(int)

        return df[["sku", "need_to_cover", "urgency"]]

    # 2. Heuristic (fallback)
    
    def _heuristic_plan(self, demand_df: pd.DataFrame):
        # працюємо з копією materials, щоб не змінювати state агента
        materials = self.materials.copy(deep=True)

        prod = (
            self.products.copy()
            .merge(demand_df[["sku", "need_to_cover", "urgency"]], on="sku", how="left")
            .fillna({"need_to_cover": 0, "urgency": 0.0})
        )

        prod["unit_profit"] = prod.unit_price - prod.unit_cost
        prod["score"] = (
            prod["unit_profit"] / prod["prod_time_min"] * (1 + prod["urgency"])
        )

        total_time = 0
        plan_rows = []

        for row in prod.sort_values("score", ascending=False).itertuples():
            max_by_time = (self.SHIFT_MINUTES - total_time) // row.prod_time_min
            max_by_demand = int(row.need_to_cover)
            if max_by_time <= 0 or max_by_demand <= 0:
                continue

            # обмеження матеріалів
            max_by_mat = np.inf
            for m_id, qty_per in self.mat_lookup.loc[row.sku].items():
                if qty_per == 0:
                    continue
                stock = float(
                    materials.loc[
                        materials.material_id == m_id, "stock_kg"
                    ].iloc[0]
                )
                max_by_mat = min(max_by_mat, stock // qty_per)

            qty_make = int(min(max_by_time, max_by_demand, max_by_mat))
            if qty_make == 0:
                continue

            plan_rows.append(
                dict(
                    sku=row.sku,
                    qty_produce=qty_make,
                    est_profit=qty_make * row.unit_profit,
                )
            )
            total_time += qty_make * row.prod_time_min

            # списуємо сировину
            for m_id, qty_per in self.mat_lookup.loc[row.sku].items():
                if qty_per == 0:
                    continue
                idx = materials.index[materials.material_id == m_id][0]
                materials.at[idx, "stock_kg"] -= qty_per * qty_make

        # результати
        plan_df = pd.DataFrame(plan_rows)

        if plan_df.empty:
            # немає, що виробляти → прибуток 0
            return plan_df, 0.0

        # якщо є рядки – дораховуємо est_profit
        plan_df["est_profit"] = plan_df["qty_produce"] * plan_df["sku"].map(
            prod.set_index("sku")["unit_profit"]
        ).fillna(0)
        return plan_df, float(plan_df["est_profit"].sum()) 