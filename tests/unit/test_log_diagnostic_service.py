from __future__ import annotations

from foamdesk.services.log_diagnostic_service import OpenFoamLogDiagnosticService


def test_log_diagnostic_service_detects_missing_file() -> None:
    service = OpenFoamLogDiagnosticService()

    diagnostics = service.diagnose('cannot find file "/case/system/controlDict"')

    assert diagnostics[0].title == "缺少 OpenFOAM 必需文件"
    assert "/case/system/controlDict" in diagnostics[0].detail


def test_log_diagnostic_service_detects_patch_field_mismatch() -> None:
    service = OpenFoamLogDiagnosticService()

    diagnostics = service.diagnose("Cannot find patchField entry for inlet")

    assert diagnostics[0].title == "边界条件名称不匹配"
    assert "inlet" in diagnostics[0].detail


def test_log_diagnostic_service_detects_pressure_reference_error() -> None:
    service = OpenFoamLogDiagnosticService()

    diagnostics = service.diagnose("Unable to set reference cell for field p")

    assert diagnostics[0].title == "压力参考值缺失"


def test_log_diagnostic_service_includes_fatal_error_fallback() -> None:
    service = OpenFoamLogDiagnosticService()

    diagnostics = service.diagnose("FOAM FATAL ERROR: unknown problem")
    formatted = service.format_diagnostics(diagnostics)

    assert diagnostics[0].title == "OpenFOAM 致命错误"
    assert "诊断 1" in formatted


def test_log_diagnostic_service_extracts_fatal_error_block() -> None:
    service = OpenFoamLogDiagnosticService()
    output = """
Create time

--> FOAM FATAL ERROR:
Cannot find patchField entry for importedGeometry

From function Foam::GeometricBoundaryField
in file fields/GeometricFields/GeometricBoundaryField.C at line 158.

FOAM exiting
"""

    block = service.extract_fatal_error_block(output)
    diagnostics = service.diagnose(output)

    assert block is not None
    assert "Cannot find patchField entry for importedGeometry" in block
    assert "FOAM exiting" in block
    assert "Cannot find patchField entry for importedGeometry" in diagnostics[-1].detail


def test_log_diagnostic_service_detects_numerical_instability() -> None:
    service = OpenFoamLogDiagnosticService()
    output = """
Courant Number mean: 0.0299153 max: 3.08557
smoothSolver:  Solving for Ux, Initial residual = 0.222643, Final residual = 4.54257e+302, No Iterations 1000
#1  Foam::sigFpe::sigHandler(int) at ??:?
"""

    diagnostics = service.diagnose(output)
    formatted = service.format_diagnostics(diagnostics)

    assert diagnostics[0].title == "数值发散或时间步过大"
    assert "3.08557" in diagnostics[0].detail
    assert "deltaT" in formatted
