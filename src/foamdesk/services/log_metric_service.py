from __future__ import annotations

import re

from foamdesk.domain.models import SolverMetrics, SolverResidual


class OpenFoamLogMetricService:
    """Extracts first structured metrics from OpenFOAM solver logs."""

    TIME_PATTERN = re.compile(r"^Time = ([0-9.eE+-]+)s?$", flags=re.MULTILINE)
    COURANT_PATTERN = re.compile(
        r"Courant Number mean:\s*([0-9.eE+-]+)\s+max:\s*([0-9.eE+-]+)"
    )
    RESIDUAL_PATTERN = re.compile(
        r"Solving for (\w+), Initial residual = ([0-9.eE+-]+), "
        r"Final residual = ([0-9.eE+-]+), No Iterations (\d+)"
    )
    CONTINUITY_PATTERN = re.compile(
        r"time step continuity errors : .* global = ([0-9.eE+-]+),"
    )

    def parse(self, output: str) -> SolverMetrics:
        time_matches = list(self.TIME_PATTERN.finditer(output))
        times = [float(match.group(1)) for match in time_matches]
        residuals = self._parse_residuals(output, time_matches)
        courant_matches = list(self.COURANT_PATTERN.finditer(output))
        continuity_matches = list(self.CONTINUITY_PATTERN.finditer(output))
        return SolverMetrics(
            times=times,
            residuals=residuals,
            courant_max=float(courant_matches[-1].group(2)) if courant_matches else None,
            latest_continuity_global=(
                float(continuity_matches[-1].group(1)) if continuity_matches else None
            ),
        )

    def format_summary(self, metrics: SolverMetrics) -> str:
        latest_time = metrics.times[-1] if metrics.times else None
        latest_residual = metrics.residuals[-1] if metrics.residuals else None
        lines = ["关键指标摘要："]
        lines.append(f"- 时间步数量：{len(metrics.times)}")
        lines.append(f"- 最新时间：{latest_time:g}" if latest_time is not None else "- 最新时间：未识别")
        if latest_residual:
            lines.append(
                f"- 最新残差：{latest_residual.field} "
                f"{latest_residual.initial:.3g} -> {latest_residual.final:.3g}"
            )
        else:
            lines.append("- 最新残差：未识别")
        if metrics.courant_max is not None:
            lines.append(f"- 最新最大 Courant Number：{metrics.courant_max:.6g}")
        else:
            lines.append("- 最新最大 Courant Number：未识别")
        if metrics.latest_continuity_global is not None:
            lines.append(f"- 最新 continuity global：{metrics.latest_continuity_global:.6g}")
        else:
            lines.append("- 最新 continuity global：未识别")
        return "\n".join(lines)

    def _parse_residuals(
        self,
        output: str,
        time_matches: list[re.Match[str]],
    ) -> list[SolverResidual]:
        residuals: list[SolverResidual] = []
        for match in self.RESIDUAL_PATTERN.finditer(output):
            residuals.append(
                SolverResidual(
                    time=self._time_for_position(match.start(), time_matches),
                    field=match.group(1),
                    initial=float(match.group(2)),
                    final=float(match.group(3)),
                    iterations=int(match.group(4)),
                )
            )
        return residuals

    def _time_for_position(
        self,
        position: int,
        time_matches: list[re.Match[str]],
    ) -> float | None:
        current_time: float | None = None
        for match in time_matches:
            if match.start() > position:
                break
            current_time = float(match.group(1))
        return current_time
