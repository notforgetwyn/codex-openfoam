from __future__ import annotations

import csv
from pathlib import Path

from foamdesk.domain.models import SimulationProject


class ResidualPlotService:
    """Loads residual CSV data for in-client Python visualization."""

    def load_series(self, project: SimulationProject) -> dict[str, list[tuple[float, float]]]:
        csv_path = project.path / "results" / "residuals.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"未找到残差文件：{csv_path}")

        series: dict[str, list[tuple[float, float]]] = {}
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                time_text = row.get("time") or ""
                field = row.get("field") or ""
                final_text = row.get("final") or ""
                if not time_text or not field or not final_text:
                    continue
                series.setdefault(field, []).append((float(time_text), float(final_text)))

        if not series:
            raise ValueError("残差 CSV 中没有可绘制的数据。")
        return series
