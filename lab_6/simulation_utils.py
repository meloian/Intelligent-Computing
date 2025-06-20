import pandas as pd


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