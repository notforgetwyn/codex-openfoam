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


def test_project_service_backfills_missing_minimal_case_files(tmp_path: Path) -> None:
    service = ProjectService(AppSettingsService(tmp_path))
    project = service.create_project("legacy")
    velocity_field = project.case_dir / "0" / "U"
    velocity_field.write_text("custom velocity", encoding="utf-8")
    missing_control_dict = project.case_dir / "system" / "controlDict"
    missing_physical_properties = project.case_dir / "constant" / "physicalProperties"
    missing_control_dict.unlink()
    missing_physical_properties.unlink()

    repaired_files = service.ensure_minimal_case_template(project)

    assert missing_control_dict in repaired_files
    assert missing_physical_properties in repaired_files
    assert velocity_field in repaired_files
    assert missing_control_dict.exists()
    assert missing_physical_properties.exists()
    assert "movingWall" in velocity_field.read_text(encoding="utf-8")


def test_project_service_syncs_fields_with_legacy_boundary_names(tmp_path: Path) -> None:
    service = ProjectService(AppSettingsService(tmp_path))
    project = service.create_project("legacy_boundary")
    (project.case_dir / "system" / "blockMeshDict").write_text(
        """
boundary
(
    inlet
    {
        type patch;
        faces ((0 1 2 3));
    }
    outlet
    {
        type patch;
        faces ((4 5 6 7));
    }
    walls
    {
        type wall;
        faces ((0 4 7 3));
    }
);
""",
        encoding="utf-8",
    )

    repaired_files = service.ensure_minimal_case_template(project)

    velocity_field = (project.case_dir / "0" / "U").read_text(encoding="utf-8")
    pressure_field = (project.case_dir / "0" / "p").read_text(encoding="utf-8")
    assert project.case_dir / "0" / "U" in repaired_files
    assert project.case_dir / "0" / "p" in repaired_files
    assert "inlet" in velocity_field
    assert "outlet" in velocity_field
    assert "walls" in velocity_field
    assert "value           uniform 0;" in pressure_field
