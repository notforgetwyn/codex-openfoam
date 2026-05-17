from __future__ import annotations

import csv
from pathlib import Path

from foamdesk.domain.models import SimulationProject


class ResidualPlotService:
    """Loads residual CSV data for in-client Python visualization."""

    def load_series(self, project: SimulationProject) -> dict[str, list[tuple[float, float]]]:
        csv_path = project.case_dir / "foamdesk_results" / "residuals.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"未找到残差文件：{csv_path}")

        raw_rows: list[tuple[float, str, float]] = []
        duplicate_counts: dict[tuple[float, str], int] = {}
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                time_text = row.get("time") or ""
                field = row.get("field") or ""
                final_text = row.get("final") or ""
                if not time_text or not field or not final_text:
                    continue
                time_value = float(time_text)
                duplicate_key = (time_value, field)
                duplicate_counts[duplicate_key] = duplicate_counts.get(duplicate_key, 0) + 1
                raw_rows.append((time_value, field, float(final_text)))

        fields_with_corrections = {
            field
            for (_time_value, field), count in duplicate_counts.items()
            if count > 1
        }
        occurrence_by_time_field: dict[tuple[float, str], int] = {}
        series: dict[str, list[tuple[float, float]]] = {}
        for time_value, field, final_value in raw_rows:
            if field in fields_with_corrections:
                occurrence_key = (time_value, field)
                occurrence = occurrence_by_time_field.get(occurrence_key, 0) + 1
                occurrence_by_time_field[occurrence_key] = occurrence
                series_name = f"{field} corrector {occurrence}"
            else:
                series_name = field
            series.setdefault(series_name, []).append((time_value, final_value))

        if not series:
            raise ValueError("残差 CSV 中没有可绘制的数据。")
        return series
