from __future__ import annotations

import csv
import json
from pathlib import Path

from foamdesk.domain.models import SolverMetrics, SolverResidual
from foamdesk.services.metric_export_service import MetricExportService
from foamdesk.services.project_service import ProjectService
from foamdesk.services.settings_service import AppSettingsService


def test_metric_export_service_writes_json_and_csv(tmp_path: Path) -> None:
    project = ProjectService(AppSettingsService(tmp_path)).create_project("demo")
    metrics = SolverMetrics(
        times=[0.005, 0.01],
        residuals=[
            SolverResidual(time=0.005, field="Ux", initial=1.0, final=1e-7, iterations=2),
            SolverResidual(time=0.01, field="p", initial=0.1, final=1e-6, iterations=8),
        ],
        courant_max=0.05,
        latest_continuity_global=1e-9,
    )

    json_path, csv_path = MetricExportService().export(project, metrics)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = list(csv.reader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert payload["times"] == [0.005, 0.01]
    assert payload["courant_max"] == 0.05
    assert payload["residuals"][0]["field"] == "Ux"
    assert rows[0] == ["time", "field", "initial", "final", "iterations"]
    assert rows[1] == ["0.005", "Ux", "1.0", "1e-07", "2"]
