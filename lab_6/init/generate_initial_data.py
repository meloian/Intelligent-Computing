import argparse
import random
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd


# ПАРАМЕТРИ ЗА ДОВІЛЬНЮ

N_PRODUCTS = 10          # кількість SKU
N_MATERIALS = 5          # видів сировини
MAX_BOM_COMPONENTS = 3   # максимум різних матеріалів у рецептурі
BASE_STOCK_UNITS = 30    # стартовий залишок товару
BASE_STOCK_KG = 1_000    # стартовий залишок сировини



# ДОПОМІЖНІ ФУНКЦІЇ

def make_products(n=N_PRODUCTS, *, rng: random.Random) -> pd.DataFrame:
    records = []
    for i in range(1, n + 1):
        shelf = rng.randint(5, 30)                         # дні придатності
        unit_cost = rng.uniform(1.5, 5.0)                  # $
        margin = rng.uniform(1.4, 2.0)                     # коеф. націнки
        prod_time = rng.randint(2, 10)                     # хв/од.
        records.append(
            dict(
                sku=f"P{i:03}",
                name=f"Product {i}",
                shelf_life_days=shelf,
                unit_cost=round(unit_cost, 2),
                unit_price=round(unit_cost * margin, 2),
                prod_time_min=prod_time,
                stock_units=BASE_STOCK_UNITS,
            )
        )
    return pd.DataFrame(records)


def make_materials(m=N_MATERIALS, *, rng: random.Random) -> pd.DataFrame:
    recs = []
    for i in range(1, m + 1):
        cost = rng.uniform(0.8, 2.5)
        recs.append(
            dict(
                material_id=f"M{i:02}",
                name=f"Material {i}",
                stock_kg=BASE_STOCK_KG,
                unit_cost=round(cost, 2),
            )
        )
    return pd.DataFrame(recs)


def make_bom(products: pd.DataFrame, materials: pd.DataFrame,
             *, rng: random.Random) -> pd.DataFrame:
    rows = []
    for _, p in products.iterrows():
        mats = rng.sample(list(materials.material_id), rng.randint(1, MAX_BOM_COMPONENTS))
        for m_id in mats:
            qty = rng.uniform(0.1, 1.0)                    # кг на одиницю
            rows.append(dict(sku=p.sku, material_id=m_id, qty_per_unit=round(qty, 2)))
    return pd.DataFrame(rows)


def make_sales_history(products: pd.DataFrame, start: dt.date,
                       days: int, *, rng: np.random.Generator) -> list[pd.DataFrame]:

    stock_remaining = products["stock_units"].to_numpy().astype(int)
    lambdas = rng.uniform(20, 60, size=len(products))  # середній попит
    histories = []

    for d in range(days):
        date = start + dt.timedelta(days=d)
        # продано: не більше ніж залишок
        q_sold = np.minimum(rng.poisson(lam=lambdas).astype(int), stock_remaining)
        stock_remaining = np.maximum(stock_remaining - q_sold, 0)

        df_day = pd.DataFrame({
            "date": date.strftime("%Y-%m-%d"),
            "sku": products["sku"],
            "quantity_sold": q_sold,
            "stock_left": stock_remaining,
        })
        histories.append(df_day)

    return histories


# ГОЛОВНА ФУНКЦІЯ

def main(args: argparse.Namespace) -> None:
    base_dir = Path(__file__).resolve().parents[1]  # production_planner/
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True, parents=True)

    rng_py = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)

    # 1) базові довідники
    products = make_products(rng=rng_py)
    materials = make_materials(rng=rng_py)
    bom = make_bom(products, materials, rng=rng_py)

    products.to_csv(data_dir / "products.csv", index=False)
    materials.to_csv(data_dir / "materials.csv", index=False)
    bom.to_csv(data_dir / "bom.csv", index=False)


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate initial CSV data.")
    parser.add_argument("--days", type=int, default=30, help="Кількість днів історії продажів")
    parser.add_argument("--seed", type=int, default=42, help="Seed для відтворюваності")
    main(parser.parse_args()) 