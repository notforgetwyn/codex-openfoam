from __future__ import annotations

from foamdesk.services.log_metric_service import OpenFoamLogMetricService


def test_log_metric_service_extracts_solver_metrics() -> None:
    log_text = """
Time = 0.005s

Courant Number mean: 0 max: 0
smoothSolver:  Solving for Ux, Initial residual = 1, Final residual = 2.48578e-08, No Iterations 2
DICPCG:  Solving for p, Initial residual = 1, Final residual = 0.0288917, No Iterations 8
time step continuity errors : sum local = 3.38063e-07, global = -3.99694e-23, cumulative = -3.99694e-23

Time = 0.01s

Courant Number mean: 8.94971e-05 max: 0.000393631
smoothSolver:  Solving for Ux, Initial residual = 0.317574, Final residual = 6.38137e-09, No Iterations 2
time step continuity errors : sum local = 7.00147e-08, global = -1.30231e-22, cumulative = -4.47269e-22
"""
    service = OpenFoamLogMetricService()

    metrics = service.parse(log_text)

    assert metrics.times == [0.005, 0.01]
    assert len(metrics.residuals) == 3
    assert metrics.residuals[0].time == 0.005
    assert metrics.residuals[0].field == "Ux"
    assert metrics.residuals[-1].time == 0.01
    assert metrics.courant_max == 0.000393631
    assert metrics.latest_continuity_global == -1.30231e-22


def test_log_metric_service_formats_empty_summary() -> None:
    service = OpenFoamLogMetricService()

    summary = service.format_summary(service.parse(""))

    assert "最新时间：未识别" in summary
    assert "最新残差：未识别" in summary
