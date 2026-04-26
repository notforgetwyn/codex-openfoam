from __future__ import annotations

import csv
import json
from pathlib import Path

from foamdesk.domain.models import SimulationProject, SolverMetrics


class MetricExportService:
    """Persists solver metrics for later plotting and external analysis."""

    def export(self, project: SimulationProject, metrics: SolverMetrics) -> tuple[Path, Path]:
        results_dir = project.path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        json_path = results_dir / "metrics.json"
        csv_path = results_dir / "residuals.csv"
        json_path.write_text(
            json.dumps(self._to_payload(metrics), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time", "field", "initial", "final", "iterations"])
            for residual in metrics.residuals:
                writer.writerow(
                    [
                        "" if residual.time is None else residual.time,
                        residual.field,
                        residual.initial,
                        residual.final,
                        residual.iterations,
                    ]
                )
        return json_path, csv_path

    def _to_payload(self, metrics: SolverMetrics) -> dict[str, object]:
        return {
            "times": metrics.times,
            "courant_max": metrics.courant_max,
            "latest_continuity_global": metrics.latest_continuity_global,
            "residuals": [
                {
                    "time": residual.time,
                    "field": residual.field,
                    "initial": residual.initial,
                    "final": residual.final,
                    "iterations": residual.iterations,
                }
                for residual in metrics.residuals
            ],
        }
