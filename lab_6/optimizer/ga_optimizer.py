from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Tuple

import numpy as np
import pandas as pd


# Головний API

def optimize_ga(
    demand_df: pd.DataFrame,
    products_df: pd.DataFrame,
    materials_df: pd.DataFrame,
    bom_df: pd.DataFrame,
    *,
    shift_minutes: int = 8 * 60,
    pop_size: int = 40,
    generations: int = 150,
    cx_prob: float = 0.6,
    mut_prob: float = 0.3,
    seed: int | None = None,
) -> Tuple[pd.DataFrame, float]:

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    skus: Sequence[str] = products_df.sku.tolist()
    n = len(skus)

    # попит, час, прибуток за одиницю
    need_map = demand_df.set_index("sku")["need_to_cover"].reindex(skus).fillna(0).astype(
        int
    )
    time_map = products_df.set_index("sku")["prod_time_min"].reindex(skus)
    base_profit = (
        products_df.set_index("sku")[["unit_price", "unit_cost"]]
        .eval("unit_price - unit_cost")
        .reindex(skus)
    )
    # коефіцієнт  (1 + urgency) - прямо з demand_df
    urgency = demand_df.set_index("sku")["urgency"].reindex(skus).fillna(0)
    profit_map = base_profit * (1 + urgency)

    # запаси матеріалів → матриця A (m × n)
    mat_ids = materials_df.material_id.tolist()
    A = np.zeros((len(mat_ids), n), dtype=float)
    for i, sku in enumerate(skus):
        rows = bom_df[bom_df.sku == sku]
        for _, r in rows.iterrows():
            j = mat_ids.index(r.material_id)
            A[j, i] = r.qty_per_unit

    mat_stock = materials_df.set_index("material_id")["stock_kg"].to_numpy()

    # допоміжні
    def feasible(q: np.ndarray) -> bool:
        return (
            (q <= need_map.to_numpy()).all()
            and (q @ time_map.to_numpy() <= shift_minutes)
            and ((A @ q) <= mat_stock + 1e-6).all()
        )

    def fitness(q: np.ndarray) -> float:
        if not feasible(q):
            # штраф: віднімаємо велике число за перевищення
            pen = 0.0
            pen += max(0, q @ time_map.to_numpy() - shift_minutes) * 1000
            over_mat = np.maximum(A @ q - mat_stock, 0).sum()
            pen += over_mat * 1000
            over_demand = np.maximum(q - need_map.to_numpy(), 0).sum()
            pen += over_demand * 500
            return -pen
        return float(q @ profit_map.to_numpy())

    # 1. ініціалізація
    def random_individual() -> np.ndarray:
        q = np.zeros(n, dtype=int)
        for i in rng.sample(range(n), k=n):  # випадковий порядок
            hi = need_map.iloc[i]
            if hi == 0:
                continue
            q[i] = rng.randint(0, hi)
            # якщо вийшли за час/матеріали — відкотимо
            if not feasible(q):
                q[i] = 0
        return q

    population = [random_individual() for _ in range(pop_size)]
    best = max(population, key=fitness)

    # 2. основний цикл
    for _ in range(generations):
        new_pop = [best.copy()]  # еліта
        while len(new_pop) < pop_size:
            # турнір
            p1, p2 = rng.sample(population, 2)
            parent1 = p1 if fitness(p1) > fitness(p2) else p2
            p3, p4 = rng.sample(population, 2)
            parent2 = p3 if fitness(p3) > fitness(p4) else p4

            # кросовер
            child = parent1.copy()
            if rng.random() < cx_prob:
                mask = np_rng.integers(0, 2, size=n).astype(bool)
                child[mask] = parent2[mask]

            # мутація
            if rng.random() < mut_prob:
                idx = rng.randrange(n)
                delta = rng.choice([-1, 1]) * rng.randint(1, max(1, need_map.iloc[idx] // 4))
                child[idx] = max(0, min(child[idx] + delta, need_map.iloc[idx]))

            # відновлення допустимості простим урізанням
            while not feasible(child) and child.sum() > 0:
                bad_idx = rng.randrange(n)
                child[bad_idx] = max(0, child[bad_idx] - 1)

            new_pop.append(child)

        population = new_pop
        cand_best = max(population, key=fitness)
        if fitness(cand_best) > fitness(best):
            best = cand_best

    # 3. результати
    qty = best.astype(int)
    plan_df = (
        pd.DataFrame({"sku": skus, "qty_produce": qty})
        .query("qty_produce > 0")
        .assign(unit_profit=lambda df: df["sku"].map(profit_map))
)
    plan_df["est_profit"] = plan_df["unit_profit"] * plan_df["qty_produce"]
    return plan_df, fitness(best) 