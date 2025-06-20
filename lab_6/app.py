from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from agents.store_agent import StoreAgent, SalesReport
from agents.production_agent import ProductionPlannerAgent
from optimizer.ga_optimizer import optimize_ga

DATA_DIR = Path(__file__).resolve().parents[0] / "data"
PLANS_DIR = DATA_DIR / "plans"
LIVE_DIR = DATA_DIR / "sales_live" 



# HELPERS

@st.cache_data(show_spinner=False)
def list_available_dates() -> list[str]:
    files = list(DATA_DIR.glob("sales_*.csv")) + list(LIVE_DIR.glob("sales_*.csv"))
    return sorted(f.stem.split("_")[1] for f in files)


def human_date(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"


@st.cache_data(show_spinner=False, ttl=60)
def get_sales(date_iso: str) -> SalesReport:
    # спроба 1 – live-каталог
    ymd = date_iso.replace("-", "")
    p_live = LIVE_DIR / f"sales_{ymd}.csv"
    if p_live.exists():
        df = pd.read_csv(p_live)
        return SalesReport(date=date_iso, df=df)

    # спроба 2 – legacy CSV через StoreAgent
    return StoreAgent(DATA_DIR).get_sales_report(date_iso)

@st.cache_data(show_spinner=False)
def material_stock_on(date_iso: str) -> pd.DataFrame:
    """Повертає DataFrame stock_kg для заданої дати, віднімаючи усі плани ≤ d."""
    materials = pd.read_csv(DATA_DIR / "materials.csv").set_index("material_id")
    bom = pd.read_csv(DATA_DIR / "bom.csv")                  # qty_per_unit
    # план-файли до обраної дати
    ymd_limit = date_iso.replace("-", "")
    for p in PLANS_DIR.glob("plan_*.csv"):
        if p.stem.split("_")[1] > ymd_limit:
            continue
        plan = pd.read_csv(p)                                # sku, qty_produce
        if plan.empty:
            continue
        # витрати
        use = (
            bom.merge(plan, on="sku")
               .assign(total=lambda d: d.qty_per_unit * d.qty_produce)
               .groupby("material_id")["total"].sum()
        )
        materials.loc[use.index, "stock_kg"] -= use
        # не даємо піти нижче нуля (для відображення)
        materials["stock_kg"] = materials["stock_kg"].clip(lower=0)
    return materials.reset_index()


# SIDEBAR

st.set_page_config(page_title="Production Planner", layout="wide")
st.sidebar.title("🗂 Дані та параметри")

# вибір дати
dates = list_available_dates()
if not dates:
    st.sidebar.warning(
        "Немає жодних продажів. Запустіть симуляцію "
        "(python simulation.py) або завантажте CSV вручну 👆"
    )
    st.stop()

default_idx = len(dates) - 1
sel_date = st.sidebar.selectbox(
    "Дата продажів", options=dates, format_func=human_date, index=default_idx
)
date_iso = human_date(sel_date)

# власний CSV
uploaded = st.sidebar.file_uploader(
    "↗️ Завантажити свій sales CSV",
    type="csv",
    help="Очікуються колонки: sku, quantity_sold, stock_left",
)

# оптимізатор
algo = st.sidebar.selectbox("Оптимізатор", ["LP (PuLP)", "GA", "Greedy"])
algo_key = "lp" if algo.startswith("LP") else ("ga" if algo == "GA" else "heur")

# GA-параметри
if algo_key == "ga":
    st.sidebar.markdown("**GA параметри**")
    ga_pop = st.sidebar.slider("Pop size", 20, 100, 40, 10)
    ga_gen = st.sidebar.slider("Generations", 50, 300, 150, 50)
else:
    ga_pop = ga_gen = None

run_btn = st.sidebar.button("🔄 Згенерувати план", use_container_width=True)

# MAIN

st.title("🏭 Production Planner")

if run_btn or "plan_df" not in st.session_state:
    # 1) завантажуємо продажі
    if uploaded is not None:
        sales_df = pd.read_csv(uploaded)
        sales_df.insert(0, "date", date_iso)
        report = SalesReport(date=date_iso, df=sales_df)
    else:
        report = get_sales(date_iso)

    st.subheader("Продажі та залишки")
    st.dataframe(report.df, use_container_width=True)

    # 2) агент-планувальник
    planner = ProductionPlannerAgent(
        DATA_DIR,
        algo="auto" if algo_key == "heur" else algo_key,
    )

    demand_df = planner._forecast_demand(report.df)  # noqa: SLA-private
    if algo_key == "ga":
        plan_df, obj_val = optimize_ga(
            demand_df,
            planner.products,
            planner.materials,
            planner.bom,
            shift_minutes=planner.SHIFT_MINUTES,
            pop_size=ga_pop,
            generations=ga_gen,
        )
    elif algo_key == "lp":
        plan = planner.handle_sales_report(report)
        plan_df, obj_val = plan.df_plan, plan.objective_value
    else:
        plan_df, obj_val = planner._heuristic_plan(demand_df)  # noqa: SLA-private

    st.session_state.plan_df = plan_df
    st.session_state.obj_val = obj_val
    st.session_state.demand_df = demand_df

# OUTPUT SECTION

plan_df: pd.DataFrame = st.session_state.get("plan_df", pd.DataFrame())
obj_val: float = st.session_state.get("obj_val", 0.0)
demand_df: pd.DataFrame = st.session_state.get("demand_df", pd.DataFrame())

col1, col2, col3 = st.columns(3)
col1.metric("📦 SKU у плані", len(plan_df))
col2.metric("🔨 Одиниць вироблено", int(plan_df.qty_produce.sum() if not plan_df.empty else 0))
col3.metric("💰 Очік. прибуток", f"${obj_val:,.0f}")

# об'єднана таблиця (demand vs plan)
if not plan_df.empty:
    merged = demand_df.merge(plan_df, on="sku", how="left").fillna(0)
    merged = merged.rename(columns={"need_to_cover": "need", "qty_produce": "plan"})
    st.subheader("Порівняння потреби та плану")
    st.dataframe(merged[["sku", "need", "plan", "urgency"]], use_container_width=True)

    st.subheader("Графік: Need vs Plan")
    chart_df = merged.set_index("sku")[["need", "plan"]]
    st.bar_chart(chart_df)
else:
    st.info("План порожній — недостатньо попиту або суворі обмеження.") 

# HISTORY

@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    dfs = []
    for f in PLANS_DIR.glob("history_*.csv"):
        algo = f.stem.split("_")[1]           # lp  /  ga
        df = pd.read_csv(f, parse_dates=["date"])
        df["algo"] = algo.upper()
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

hist_df = load_history()
if not hist_df.empty:
    # прибуток за днями для всіх алгоритмів
    st.subheader("📈 Динаміка симуляції")
    st.line_chart(
        hist_df.pivot(index="date", columns="algo", values="profit"),
        use_container_width=True,
    )

    # випущені одиниці
    st.subheader("🔨 Вироблено одиниць")
    st.line_chart(
        hist_df.pivot(index="date", columns="algo", values="units_produced"),
        use_container_width=True,
    )

    # кумулятивний прибуток
    st.subheader("💰 Кумулятивний прибуток")
    cum = (
        hist_df.sort_values("date")
               .groupby("algo", group_keys=False)
               .apply(lambda d: d.assign(cum_profit=d["profit"].cumsum()))
    )
    st.area_chart(
        cum.pivot(index="date", columns="algo", values="cum_profit"),
        use_container_width=True,
    )

#  ЗАЛИШКИ МАТЕРІАЛІВ та РЕЦЕПТИ

# 1) матеріали
st.subheader("📊 Залишок матеріалів на кінець дня")
mat_df = material_stock_on(date_iso)
st.dataframe(mat_df[["material_id", "name", "stock_kg"]],
             use_container_width=True)

# 2) рецептури (BOM)
st.subheader("📋 Кількість матеріалу на одиницю SKU)")
bom = pd.read_csv(DATA_DIR / "bom.csv").merge(
    pd.read_csv(DATA_DIR / "materials.csv")[["material_id", "name"]],
    on="material_id", how="left"
)
st.dataframe(bom.rename(columns={"name": "material_name"}),
             use_container_width=True) 


def apply_plan_to_stocks(plan_df: pd.DataFrame,
                         products_df: pd.DataFrame,
                         materials_df: pd.DataFrame,
                         bom_df: pd.DataFrame) -> None:
    if plan_df.empty:
        return

    # товар
    for row in plan_df.itertuples():
        idx = products_df.index[products_df.sku == row.sku][0]
        products_df.at[idx, "stock_units"] += int(row.qty_produce)

    # матеріали
    mat_lookup = bom_df.pivot_table(index="sku",
                                    columns="material_id",
                                    values="qty_per_unit",
                                    fill_value=0.0).loc[plan_df.sku]
    usage = (mat_lookup.T * plan_df.set_index("sku")["qty_produce"]).T.sum(axis=0)
    for m_id, used in usage.items():
        idx = materials_df.index[materials_df.material_id == m_id][0]
        materials_df.at[idx, "stock_kg"] = max(0.0, materials_df.at[idx, "stock_kg"] - used)

# довідка: час виготовлення товарів

st.subheader("⏱️ Час виготовлення кожного SKU")
prod_time_df = (
    pd.read_csv(DATA_DIR / "products.csv")[["sku", "name", "prod_time_min"]]
      .rename(columns={"prod_time_min": "minutes_per_unit"})
)
st.dataframe(prod_time_df, use_container_width=True)