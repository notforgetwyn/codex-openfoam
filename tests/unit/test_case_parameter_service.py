from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from foamdesk.services.case_parameter_service import OpenFoamCaseParameterService
from foamdesk.services.project_service import ProjectService
from foamdesk.services.settings_service import AppSettingsService


def test_case_parameter_service_loads_defaults_from_generated_case(tmp_path: Path) -> None:
    project = ProjectService(AppSettingsService(tmp_path)).create_project("demo")
    service = OpenFoamCaseParameterService()

    parameters = service.load(project)

    assert parameters == replace(
        service.defaults(),
        end_time=0.5,
        delta_t=0.001,
        write_interval=100,
        viscosity=0.01,
    )


def test_case_parameter_service_saves_control_and_viscosity(tmp_path: Path) -> None:
    project = ProjectService(AppSettingsService(tmp_path)).create_project("demo")
    service = OpenFoamCaseParameterService()

    service.save(
        project,
        replace(
            service.defaults(),
            end_time=1.25,
            delta_t=0.01,
            write_interval=10,
            viscosity=0.02,
        ),
    )

    control_dict = (project.case_dir / "system" / "controlDict").read_text(encoding="utf-8")
    physical_properties = (project.case_dir / "constant" / "physicalProperties").read_text(
        encoding="utf-8"
    )
    transport_properties = (project.case_dir / "constant" / "transportProperties").read_text(
        encoding="utf-8"
    )
    fv_solution = (project.case_dir / "system" / "fvSolution").read_text(encoding="utf-8")
    assert "endTime         1.25;" in control_dict
    assert "deltaT          0.01;" in control_dict
    assert "writeInterval   10;" in control_dict
    assert "nu              [0 2 -1 0 0 0 0] 0.02;" in physical_properties
    assert "nu              [0 2 -1 0 0 0 0] 0.02;" in transport_properties
    assert "pFinal" in fv_solution
    assert "pRefCell        0;" in fv_solution
    assert "pRefValue       0;" in fv_solution


def test_case_parameter_service_rejects_invalid_values(tmp_path: Path) -> None:
    project = ProjectService(AppSettingsService(tmp_path)).create_project("demo")
    service = OpenFoamCaseParameterService()

    with pytest.raises(ValueError, match="时间步长不能大于结束时间"):
        service.save(
            project,
            replace(
                service.defaults(),
                end_time=0.1,
                delta_t=1.0,
                write_interval=1,
                viscosity=0.01,
            ),
        )
