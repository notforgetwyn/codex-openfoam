from __future__ import annotations

from pathlib import Path

from foamdesk.services.openfoam_vtk_service import OpenFoamVtkService
from foamdesk.services.project_service import ProjectService
from foamdesk.services.settings_service import AppSettingsService


def test_openfoam_vtk_service_creates_marker_file(tmp_path: Path) -> None:
    project = ProjectService(AppSettingsService(tmp_path)).create_project("demo")
    service = OpenFoamVtkService()

    marker_file = service.ensure_marker_file(project)

    assert marker_file.name == "foamdesk.foam"
    assert marker_file.exists()
    assert marker_file.parent == project.case_dir
