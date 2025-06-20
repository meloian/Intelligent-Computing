from __future__ import annotations

import datetime as dt
from pathlib import Path
from dataclasses import dataclass

import pandas as pd


# МОДЕЛІ ПОВІДОМЛЕНЬ

@dataclass(slots=True)
class SalesReport:
    date: str            # YYYY-MM-DD
    df: pd.DataFrame     # колонки: sku, quantity_sold, stock_left


# StoreAgent

class StoreAgent:
    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)            # спершу зберігаємо
        self.live_dir = self.data_dir / "sales_live"

    # main API
    def get_sales_report(self, date: str | dt.date = None) -> SalesReport:
        if date is None:
            sales_files = sorted(self.data_dir.glob("sales_*.csv"))
            if not sales_files:
                raise FileNotFoundError("sales_*.csv не знайдено у data/.")
            file_path = sales_files[-1]
        else:
            date_obj = (
                dt.date.fromisoformat(date) if isinstance(date, str) else date
            )
            ymd = f"{date_obj:%Y%m%d}"
            file_path = (
                (self.live_dir / f"sales_{ymd}.csv")
                if (self.live_dir / f"sales_{ymd}.csv").exists()
                else self.data_dir / f"sales_{ymd}.csv"
            )
            if not file_path.exists():
                raise FileNotFoundError(file_path)

        df = pd.read_csv(file_path)
        return SalesReport(date=df.iloc[0]["date"], df=df)

    def publish(self, report: SalesReport, *, callback) -> None:
        callback(report)