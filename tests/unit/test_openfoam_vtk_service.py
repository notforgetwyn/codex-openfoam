from __future__ import annotations

from pathlib import Path

from vtkmodules.vtkCommonCore import vtkDoubleArray
from vtkmodules.vtkCommonDataModel import vtkPolyData

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


def test_openfoam_vtk_service_reads_vtk_array_names() -> None:
    service = OpenFoamVtkService()
    poly_data = vtkPolyData()
    pressure = vtkDoubleArray()
    pressure.SetName("p")
    velocity = vtkDoubleArray()
    velocity.SetName("U")

    poly_data.GetPointData().AddArray(pressure)
    poly_data.GetCellData().AddArray(velocity)

    assert service._array_names(poly_data.GetPointData()) == ["p"]
    assert service._array_names(poly_data.GetCellData()) == ["U"]


def test_openfoam_vtk_service_builds_geometry_filter_with_time_value(tmp_path: Path) -> None:
    project = ProjectService(AppSettingsService(tmp_path)).create_project("demo")
    service = OpenFoamVtkService()

    geometry = service.build_geometry_filter(project, time_value=0.0)

    assert geometry is not None
    assert service.ensure_marker_file(project).exists()
