from __future__ import annotations

import re
from pathlib import Path

from foamdesk.domain.models import SimulationParameters, SimulationProject


class OpenFoamCaseParameterService:
    """Reads and writes the first editable OpenFOAM case parameters."""

    DEFAULT_PARAMETERS = SimulationParameters(
        end_time=0.5,
        delta_t=0.005,
        write_interval=20,
        viscosity=0.01,
    )

    def load(self, project: SimulationProject) -> SimulationParameters:
        control_dict = (project.case_dir / "system" / "controlDict").read_text(encoding="utf-8")
        physical_properties = self._physical_properties_path(project).read_text(encoding="utf-8")
        return SimulationParameters(
            end_time=self._read_float(control_dict, "endTime", self.DEFAULT_PARAMETERS.end_time),
            delta_t=self._read_float(control_dict, "deltaT", self.DEFAULT_PARAMETERS.delta_t),
            write_interval=self._read_int(
                control_dict,
                "writeInterval",
                self.DEFAULT_PARAMETERS.write_interval,
            ),
            viscosity=self._read_viscosity(physical_properties, self.DEFAULT_PARAMETERS.viscosity),
        )

    def save(self, project: SimulationProject, parameters: SimulationParameters) -> None:
        self._validate(parameters)
        control_path = project.case_dir / "system" / "controlDict"
        control_text = control_path.read_text(encoding="utf-8")
        control_text = self._replace_assignment(control_text, "endTime", self._format_float(parameters.end_time))
        control_text = self._replace_assignment(control_text, "deltaT", self._format_float(parameters.delta_t))
        control_text = self._replace_assignment(control_text, "writeInterval", str(parameters.write_interval))
        control_path.write_text(control_text, encoding="utf-8")

        viscosity_value = self._format_float(parameters.viscosity)
        for path in (
            project.case_dir / "constant" / "physicalProperties",
            project.case_dir / "constant" / "transportProperties",
        ):
            if path.exists():
                text = path.read_text(encoding="utf-8")
                text = self._replace_viscosity(text, viscosity_value)
            else:
                text = self._default_properties_text(path.stem, viscosity_value)
            path.write_text(text, encoding="utf-8")

    def defaults(self) -> SimulationParameters:
        return self.DEFAULT_PARAMETERS

    def _physical_properties_path(self, project: SimulationProject) -> Path:
        physical_properties = project.case_dir / "constant" / "physicalProperties"
        if physical_properties.exists():
            return physical_properties
        return project.case_dir / "constant" / "transportProperties"

    def _read_float(self, text: str, key: str, default: float) -> float:
        match = re.search(rf"^\s*{re.escape(key)}\s+([^;]+);", text, flags=re.MULTILINE)
        if not match:
            return default
        return float(match.group(1).strip())

    def _read_int(self, text: str, key: str, default: int) -> int:
        match = re.search(rf"^\s*{re.escape(key)}\s+([^;]+);", text, flags=re.MULTILINE)
        if not match:
            return default
        return int(float(match.group(1).strip()))

    def _read_viscosity(self, text: str, default: float) -> float:
        match = re.search(r"^\s*nu\s+\[[^\]]+\]\s+([^;]+);", text, flags=re.MULTILINE)
        if not match:
            return default
        return float(match.group(1).strip())

    def _replace_assignment(self, text: str, key: str, value: str) -> str:
        pattern = rf"(^\s*{re.escape(key)}\s+)([^;]+)(;)"
        replacement = rf"\g<1>{value}\g<3>"
        replaced, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if count:
            return replaced
        return f"{text.rstrip()}\n{key:<16}{value};\n"

    def _replace_viscosity(self, text: str, value: str) -> str:
        pattern = r"(^\s*nu\s+\[[^\]]+\]\s+)([^;]+)(;)"
        replacement = rf"\g<1>{value}\g<3>"
        replaced, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if count:
            return replaced
        return f"{text.rstrip()}\nnu              [0 2 -1 0 0 0 0] {value};\n"

    def _validate(self, parameters: SimulationParameters) -> None:
        if parameters.end_time <= 0:
            raise ValueError("结束时间必须大于 0。")
        if parameters.delta_t <= 0:
            raise ValueError("时间步长必须大于 0。")
        if parameters.delta_t > parameters.end_time:
            raise ValueError("时间步长不能大于结束时间。")
        if parameters.write_interval <= 0:
            raise ValueError("写出间隔必须大于 0。")
        if parameters.viscosity <= 0:
            raise ValueError("运动粘度必须大于 0。")

    def _format_float(self, value: float) -> str:
        return f"{value:.12g}"

    def _default_properties_text(self, object_name: str, viscosity: str) -> str:
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      {object_name};
}}

nu              [0 2 -1 0 0 0 0] {viscosity};
"""
