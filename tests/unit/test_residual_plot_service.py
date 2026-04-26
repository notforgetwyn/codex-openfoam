from __future__ import annotations

from pathlib import Path

import pytest

from foamdesk.services.project_service import ProjectService
from foamdesk.services.residual_plot_service import ResidualPlotService
from foamdesk.services.settings_service import AppSettingsService


def test_residual_plot_service_loads_series_from_csv(tmp_path: Path) -> None:
    project = ProjectService(AppSettingsService(tmp_path)).create_project("demo")
    results_dir = project.path / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "residuals.csv").write_text(
        "time,field,initial,final,iterations\n"
        "0.005,Ux,1.0,1e-7,2\n"
        "0.010,Ux,0.5,5e-8,2\n"
        "0.005,p,1.0,1e-3,8\n",
        encoding="utf-8",
    )

    series = ResidualPlotService().load_series(project)

    assert series["Ux"] == [(0.005, 1e-7), (0.01, 5e-8)]
    assert series["p"] == [(0.005, 1e-3)]


def test_residual_plot_service_requires_csv(tmp_path: Path) -> None:
    project = ProjectService(AppSettingsService(tmp_path)).create_project("demo")

    with pytest.raises(FileNotFoundError):
        ResidualPlotService().load_series(project)
