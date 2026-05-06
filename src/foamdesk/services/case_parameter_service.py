from __future__ import annotations

import re
import json
from dataclasses import replace
from pathlib import Path

from foamdesk.domain.models import SimulationParameters, SimulationProject


class OpenFoamCaseParameterService:
    """Reads and writes the first editable OpenFOAM case parameters."""

    MATERIAL_PRESETS = {
        "air": {
            "label": "空气",
            "density": 1.225,
            "viscosity": 1.5e-5,
        },
        "water": {
            "label": "水",
            "density": 998.2,
            "viscosity": 1.004e-6,
        },
        "oil": {
            "label": "机油",
            "density": 850.0,
            "viscosity": 1.0e-4,
        },
        "custom": {
            "label": "自定义流体",
            "density": 1.0,
            "viscosity": 0.01,
        },
    }

    DEFAULT_PARAMETERS = SimulationParameters(
        solver_name="icoFoam",
        end_time=0.5,
        delta_t=0.001,
        write_interval=100,
        max_iterations=100,
        residual_tolerance=1e-6,
        material_name="custom",
        density=1.0,
        viscosity=0.01,
        dynamic_viscosity=0.01,
        turbulence_model="laminar",
        numeric_scheme="stable",
        fv_solution_preset="default",
    )

    SOLVERS = {
        "icoFoam": "入门不可压瞬态流，适合先跑通流程。",
        "simpleFoam": "稳态不可压流，适合风洞/绕流的工程入门场景。",
        "pisoFoam": "瞬态不可压流，适合观察随时间变化的流动。",
    }

    def load(self, project: SimulationProject) -> SimulationParameters:
        control_dict = (project.case_dir / "system" / "controlDict").read_text(encoding="utf-8")
        physical_properties = self._physical_properties_path(project).read_text(encoding="utf-8")
        config = self._read_config(project)
        default = self.DEFAULT_PARAMETERS
        return SimulationParameters(
            solver_name=str(config.get("solver_name", self._read_word(control_dict, "application", default.solver_name))),
            end_time=self._read_float(control_dict, "endTime", default.end_time),
            delta_t=self._read_float(control_dict, "deltaT", default.delta_t),
            write_interval=self._read_int(
                control_dict,
                "writeInterval",
                default.write_interval,
            ),
            max_iterations=int(config.get("max_iterations", default.max_iterations)),
            residual_tolerance=float(config.get("residual_tolerance", default.residual_tolerance)),
            material_name=str(config.get("material_name", default.material_name)),
            density=float(config.get("density", self._read_density(physical_properties, default.density))),
            viscosity=self._read_viscosity(physical_properties, default.viscosity),
            dynamic_viscosity=float(
                config.get(
                    "dynamic_viscosity",
                    self._read_dynamic_viscosity(
                        physical_properties,
                        self._read_density(physical_properties, default.density)
                        * self._read_viscosity(physical_properties, default.viscosity),
                    ),
                )
            ),
            turbulence_model=str(config.get("turbulence_model", default.turbulence_model)),
            numeric_scheme=str(config.get("numeric_scheme", default.numeric_scheme)),
            fv_solution_preset=str(config.get("fv_solution_preset", default.fv_solution_preset)),
        )

    def save(self, project: SimulationProject, parameters: SimulationParameters) -> None:
        self._validate(parameters)
        control_path = project.case_dir / "system" / "controlDict"
        control_text = control_path.read_text(encoding="utf-8")
        control_text = self._replace_assignment(control_text, "application", parameters.solver_name)
        control_text = self._replace_assignment(control_text, "endTime", self._format_float(parameters.end_time))
        control_text = self._replace_assignment(control_text, "deltaT", self._format_float(parameters.delta_t))
        control_text = self._replace_assignment(control_text, "writeInterval", str(parameters.write_interval))
        control_path.write_text(control_text, encoding="utf-8")

        viscosity_value = self._format_float(parameters.viscosity)
        density_value = self._format_float(parameters.density)
        dynamic_viscosity_value = self._format_float(parameters.dynamic_viscosity)
        for path in (
            project.case_dir / "constant" / "physicalProperties",
            project.case_dir / "constant" / "transportProperties",
        ):
            if path.exists():
                text = path.read_text(encoding="utf-8")
                text = self._replace_viscosity(text, viscosity_value)
                text = self._replace_density(text, density_value)
                text = self._replace_dynamic_viscosity(text, dynamic_viscosity_value)
            else:
                text = self._default_properties_text(
                    path.stem,
                    viscosity_value,
                    density_value,
                    dynamic_viscosity_value,
                )
            path.write_text(text, encoding="utf-8")
        (project.case_dir / "system" / "fvSchemes").write_text(
            self._fv_schemes_text(parameters.numeric_scheme),
            encoding="utf-8",
        )
        (project.case_dir / "system" / "fvSolution").write_text(
            self._fv_solution_text(parameters),
            encoding="utf-8",
        )
        self._config_path(project).write_text(
            json.dumps(self._to_config(parameters), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def defaults(self) -> SimulationParameters:
        return self.DEFAULT_PARAMETERS

    def material_preset(self, material_name: str) -> SimulationParameters:
        preset = self.MATERIAL_PRESETS.get(material_name)
        if preset is None:
            raise ValueError("物性预设必须是 air、water、oil 或 custom。")
        density = float(preset["density"])
        viscosity = float(preset["viscosity"])
        return replace(
            self.DEFAULT_PARAMETERS,
            material_name=material_name,
            density=density,
            viscosity=viscosity,
            dynamic_viscosity=density * viscosity,
        )

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

    def _read_word(self, text: str, key: str, default: str) -> str:
        match = re.search(rf"^\s*{re.escape(key)}\s+([^;]+);", text, flags=re.MULTILINE)
        if not match:
            return default
        return match.group(1).strip()

    def _read_viscosity(self, text: str, default: float) -> float:
        match = re.search(r"^\s*nu\s+\[[^\]]+\]\s+([^;]+);", text, flags=re.MULTILINE)
        if not match:
            return default
        return float(match.group(1).strip())

    def _read_density(self, text: str, default: float) -> float:
        match = re.search(r"^\s*rho\s+\[[^\]]+\]\s+([^;]+);", text, flags=re.MULTILINE)
        if not match:
            return default
        return float(match.group(1).strip())

    def _read_dynamic_viscosity(self, text: str, default: float) -> float:
        match = re.search(r"^\s*mu\s+\[[^\]]+\]\s+([^;]+);", text, flags=re.MULTILINE)
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

    def _replace_density(self, text: str, value: str) -> str:
        pattern = r"(^\s*rho\s+\[[^\]]+\]\s+)([^;]+)(;)"
        replacement = rf"\g<1>{value}\g<3>"
        replaced, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if count:
            return replaced
        return f"{text.rstrip()}\nrho             [1 -3 0 0 0 0 0] {value};\n"

    def _replace_dynamic_viscosity(self, text: str, value: str) -> str:
        pattern = r"(^\s*mu\s+\[[^\]]+\]\s+)([^;]+)(;)"
        replacement = rf"\g<1>{value}\g<3>"
        replaced, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if count:
            return replaced
        return f"{text.rstrip()}\nmu              [1 -1 -1 0 0 0 0] {value};\n"

    def _validate(self, parameters: SimulationParameters) -> None:
        if parameters.solver_name not in self.SOLVERS:
            raise ValueError("求解器必须是 icoFoam、simpleFoam 或 pisoFoam。")
        if parameters.end_time <= 0:
            raise ValueError("结束时间必须大于 0。")
        if parameters.delta_t <= 0:
            raise ValueError("时间步长必须大于 0。")
        if parameters.delta_t > parameters.end_time:
            raise ValueError("时间步长不能大于结束时间。")
        if parameters.write_interval <= 0:
            raise ValueError("写出间隔必须大于 0。")
        if parameters.max_iterations <= 0:
            raise ValueError("最大迭代次数必须大于 0。")
        if parameters.residual_tolerance <= 0:
            raise ValueError("收敛残差必须大于 0。")
        if parameters.material_name not in self.MATERIAL_PRESETS:
            raise ValueError("流体类型必须是空气、水、机油或自定义流体。")
        if parameters.density <= 0:
            raise ValueError("流体密度必须大于 0。")
        if parameters.viscosity <= 0:
            raise ValueError("运动粘度必须大于 0。")
        if parameters.dynamic_viscosity <= 0:
            raise ValueError("动力粘度必须大于 0。")
        if parameters.turbulence_model not in {"laminar", "RAS kEpsilon"}:
            raise ValueError("湍流模型当前只支持 laminar 或 RAS kEpsilon。")
        if parameters.numeric_scheme not in {"stable", "balanced", "accurate"}:
            raise ValueError("数值格式当前只支持 stable、balanced、accurate。")
        if parameters.fv_solution_preset not in {"default", "strict", "fast"}:
            raise ValueError("fvSolution 预设当前只支持 default、strict、fast。")

    def _format_float(self, value: float) -> str:
        return f"{value:.12g}"

    def _config_path(self, project: SimulationProject) -> Path:
        return project.case_dir / "system" / "foamdesk_simulation_config.json"

    def _read_config(self, project: SimulationProject) -> dict:
        path = self._config_path(project)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return {}

    def _to_config(self, parameters: SimulationParameters) -> dict:
        return {
            "solver_name": parameters.solver_name,
            "max_iterations": parameters.max_iterations,
            "residual_tolerance": parameters.residual_tolerance,
            "material_name": parameters.material_name,
            "density": parameters.density,
            "dynamic_viscosity": parameters.dynamic_viscosity,
            "turbulence_model": parameters.turbulence_model,
            "numeric_scheme": parameters.numeric_scheme,
            "fv_solution_preset": parameters.fv_solution_preset,
        }

    def _default_properties_text(
        self,
        object_name: str,
        viscosity: str,
        density: str,
        dynamic_viscosity: str,
    ) -> str:
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      {object_name};
}}

nu              [0 2 -1 0 0 0 0] {viscosity};
rho             [1 -3 0 0 0 0 0] {density};
mu              [1 -1 -1 0 0 0 0] {dynamic_viscosity};
"""

    def _fv_schemes_text(self, preset: str) -> str:
        div_scheme = {
            "stable": "Gauss upwind",
            "balanced": "Gauss linearUpwind grad(U)",
            "accurate": "Gauss linear",
        }.get(preset, "Gauss upwind")
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}}

ddtSchemes
{{
    default         Euler;
}}

gradSchemes
{{
    default         Gauss linear;
}}

divSchemes
{{
    default         none;
    div(phi,U)      {div_scheme};
}}

laplacianSchemes
{{
    default         Gauss linear corrected;
}}

interpolationSchemes
{{
    default         linear;
}}

snGradSchemes
{{
    default         corrected;
}}
"""

    def _fv_solution_text(self, parameters: SimulationParameters) -> str:
        rel_tol = "0" if parameters.fv_solution_preset == "strict" else "0.05"
        smoother = "symGaussSeidel" if parameters.fv_solution_preset != "fast" else "GaussSeidel"
        tolerance = self._format_float(parameters.residual_tolerance)
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}}

solvers
{{
    p
    {{
        solver          PCG;
        preconditioner  DIC;
        tolerance       {tolerance};
        relTol          {rel_tol};
    }}

    pFinal
    {{
        $p;
        relTol          0;
    }}

    U
    {{
        solver          smoothSolver;
        smoother        {smoother};
        tolerance       {tolerance};
        relTol          {rel_tol};
    }}
}}

PISO
{{
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}}

SIMPLE
{{
    nNonOrthogonalCorrectors 0;
    residualControl
    {{
        p               {tolerance};
        U               {tolerance};
    }}
}}

relaxationFactors
{{
    fields
    {{
        p               0.3;
    }}
    equations
    {{
        U               0.7;
    }}
}}
"""
