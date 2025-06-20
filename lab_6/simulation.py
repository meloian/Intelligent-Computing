from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import trange

from agents.store_agent import SalesReport          # тільки dataclass
from agents.production_agent import ProductionPlannerAgent
from optimizer.ga_optimizer import optimize_ga

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("LIVE daily simulation")
    p.add_argument("--days", type=int, default=30, help="Кількість днів симуляції")
    p.add_argument(
        "--algo",
        choices=["lp", "ga"],
        default="lp",
        help="lp → PuLP / ga → генетичний алгоритм",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data")
    return p.parse_args()


def main() -> None:
    args = cli()
    rng = np.random.default_rng(args.seed)

    base_dir = Path(__file__).resolve().parents[0]
    data_dir = base_dir / args.data_dir
    plans_dir = data_dir / "plans"
    sales_dir = data_dir / "sales_live"
    plans_dir.mkdir(parents=True, exist_ok=True)
    sales_dir.mkdir(parents=True, exist_ok=True)

    # довідники
    products = pd.read_csv(data_dir / "products.csv")
    materials = pd.read_csv(data_dir / "materials.csv")
    bom = pd.read_csv(data_dir / "bom.csv")

    # стартові склади магазинів
    stock_store = products.set_index("sku")["stock_units"].to_numpy().astype(int)

    # стартове λ для кожного SKU
    lambdas = rng.uniform(20, 60, size=len(products))

    # planner-агент
    planner = ProductionPlannerAgent(data_dir, algo=args.algo)

    # забезпечуємо єдиний "живий" об’єкт DataFrame,
    # щоб обидві сторони дивилися на ті самі числа
    planner.materials = materials            # спільний об’єкт
    planner.products  = products             

    history: list[dict] = []
    start_date = dt.date.today()

    print(f"▶ LIVE-симуляція {args.days} дн., алгоритм={args.algo.upper()}")

    for day in trange(args.days, unit="day"):
        date = start_date + dt.timedelta(days=day)
        date_iso = date.isoformat()

        # 1. попит магазину
        sold = rng.poisson(lambdas).astype(int)
        sold = np.minimum(sold, stock_store)          # не більше, ніж у наявності
        stock_store -= sold

        sales_df = pd.DataFrame(
            {
                "date": date_iso,
                "sku": products.sku,
                "quantity_sold": sold,
                "stock_left": stock_store,
            }
        )
        # лишимо історію
        sales_df.to_csv(sales_dir / f"sales_{date:%Y%m%d}.csv", index=False)

        report = SalesReport(date=date_iso, df=sales_df)

        # 2. планування
        if args.algo == "lp":
            plan = planner.handle_sales_report(report)
            plan_df, profit = plan.df_plan, plan.objective_value
        else:
            demand = planner._forecast_demand(sales_df)
            plan_df, profit = optimize_ga(
                demand,
                planner.products,
                planner.materials,
                planner.bom,
                shift_minutes=planner.SHIFT_MINUTES,
            )
        
        # 2a. якщо план порожній — пропускаємо день
        if plan_df.empty:
            profit = 0.0  # жодних витрат, зміна скасована
            history.append(dict(
                date=date_iso,
                skus=0,
                units_produced=0,
                profit=profit,
            ))
            continue  # до наступного дня

        # 3. оновити склади фабрики + списати матеріали
        #    (функція вже додає qty_produce до products.stock_units)
        from simulation_utils import apply_plan_to_stocks  # new helper
        apply_plan_to_stocks(plan_df, products, materials, bom)

        # 4. доставити вироблене у магазин (на завтра)
        if not plan_df.empty:
            produced_vec = plan_df.set_index("sku").reindex(products.sku)["qty_produce"].fillna(0).to_numpy().astype(
                int
            )
            stock_store += produced_vec

        # 5. оновити λ (повільніше згладжування)
        lambdas = 0.95 * lambdas + 0.05 * sold

        # 6. журнал
        history.append(
            dict(
                date=date_iso,
                skus=len(plan_df),
                units_produced=int(plan_df.qty_produce.sum() if not plan_df.empty else 0),
                profit=profit,
            )
        )

        # зберігаємо денний план
        plan_df.to_csv(plans_dir / f"plan_{date:%Y%m%d}.csv", index=False)

    # фінальний звіт
    hist_df = pd.DataFrame(history)

    # history_lp.csv  або  history_ga.csv
    hist_path = plans_dir / f"history_{args.algo}.csv"
    hist_df.to_csv(hist_path, index=False)
    print("\n✔ DONE.  Підсумок:")
    print(hist_df.to_string(index=False, float_format="%.2f"))
    print(f"\n• Плани:  {plans_dir.resolve()}\n• Продажі: {sales_dir.resolve()}")


if __name__ == "__main__":
    main()