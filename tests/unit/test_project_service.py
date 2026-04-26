from __future__ import annotations

from pathlib import Path

import pytest

from foamdesk.services.project_service import ProjectService
from foamdesk.services.settings_service import AppSettingsService


def test_project_service_creates_project_with_case_template(tmp_path: Path) -> None:
    service = ProjectService(AppSettingsService(tmp_path))

    project = service.create_project("demo")

    assert project.name == "demo"
    assert (project.path / "project.json").exists()
    assert (project.case_dir / "system" / "blockMeshDict").exists()
    assert (project.case_dir / "system" / "controlDict").exists()
    assert (project.case_dir / "system" / "fvSchemes").exists()
    assert (project.case_dir / "system" / "fvSolution").exists()
    assert (project.case_dir / "0" / "U").exists()
    assert (project.case_dir / "0" / "p").exists()
    assert (project.case_dir / "constant" / "transportProperties").exists()
    assert (project.case_dir / "constant" / "physicalProperties").exists()
    assert (project.case_dir / "0").exists()
    assert (project.case_dir / "constant").exists()

    control_dict = (project.case_dir / "system" / "controlDict").read_text(encoding="utf-8")
    fv_solution = (project.case_dir / "system" / "fvSolution").read_text(encoding="utf-8")
    velocity_field = (project.case_dir / "0" / "U").read_text(encoding="utf-8")
    assert "application     icoFoam;" in control_dict
    assert "pRefCell" in fv_solution
    assert "movingWall" in velocity_field


def test_project_service_rejects_duplicate_project(tmp_path: Path) -> None:
    service = ProjectService(AppSettingsService(tmp_path))

    service.create_project("demo")

    with pytest.raises(ValueError, match="项目已存在"):
        service.create_project("demo")
