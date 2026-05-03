from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

from foamdesk.domain.models import SimulationProject
from foamdesk.services.settings_service import AppSettingsService


@dataclass(frozen=True, slots=True)
class ComputationDomainTemplate:
    key: str
    name: str
    level: str
    size: tuple[float, float, float]
    cells: tuple[int, int, int]
    suggested_location_in_mesh: tuple[float, float, float]
    description: str
    shape: str = "box"


@dataclass(frozen=True, slots=True)
class BoundaryConditionSettings:
    inlet_velocity: tuple[float, float, float] = (1.0, 0.0, 0.0)
    outlet_pressure: float = 0.0
    wall_type: str = "noSlip"


class ProjectService:
    """Creates and discovers local simulation projects in the configured workspace."""

    def __init__(self, settings_service: AppSettingsService) -> None:
        self._settings_service = settings_service

    def domain_templates(self) -> tuple[ComputationDomainTemplate, ...]:
        return (
            ComputationDomainTemplate(
                key="simple_unit_box",
                name="简单：单位立方体 1 x 1 x 1",
                level="简单",
                size=(1.0, 1.0, 1.0),
                cells=(10, 10, 10),
                suggested_location_in_mesh=(0.2, 0.2, 0.2),
                description="用于最快验证流程，小 STL 测试最合适。",
            ),
            ComputationDomainTemplate(
                key="medium_wind_tunnel",
                name="中等：小型风洞 4 x 2 x 2",
                level="中等",
                size=(4.0, 2.0, 2.0),
                cells=(40, 20, 20),
                suggested_location_in_mesh=(0.5, 1.0, 1.0),
                description="推荐默认模板，用于流体从左到右流过一个小物体。",
            ),
            ComputationDomainTemplate(
                key="advanced_long_wind_tunnel",
                name="高等：长风洞 10 x 4 x 4",
                level="高等",
                size=(10.0, 4.0, 4.0),
                cells=(80, 32, 32),
                suggested_location_in_mesh=(1.0, 2.0, 2.0),
                description="用于较大外流场景，能给物体前后留出更长流动空间。",
            ),
            ComputationDomainTemplate(
                key="medium_tapered_wind_tunnel",
                name="中等：渐扩风洞 6 x 2.5 x 2.5",
                level="中等",
                size=(6.0, 2.5, 2.5),
                cells=(60, 24, 24),
                suggested_location_in_mesh=(1.0, 1.25, 1.25),
                description="入口截面小、出口截面大，用于观察流体从窄口进入后扩散的场景。",
                shape="tapered",
            ),
            ComputationDomainTemplate(
                key="advanced_ramp_channel",
                name="高等：斜坡通道 8 x 3 x 2.5",
                level="高等",
                size=(8.0, 3.0, 2.5),
                cells=(70, 28, 24),
                suggested_location_in_mesh=(1.0, 1.5, 1.0),
                description="底部带斜坡的通道，用于观察流体经过地形/坡面时的变化。",
                shape="ramp",
            ),
        )

    def load_domain_template_key(self, project: SimulationProject) -> str:
        config_path = self._domain_config_path(project.case_dir)
        if not config_path.exists():
            return "simple_unit_box"
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            key = str(payload.get("key", "simple_unit_box"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return "simple_unit_box"
        if key == "custom_domain":
            return key
        if any(template.key == key for template in self.domain_templates()):
            return key
        return "simple_unit_box"

    def apply_domain_template(self, project: SimulationProject, template_key: str) -> ComputationDomainTemplate:
        template = self._domain_template_by_key(template_key)
        self._write_domain(project, template)
        return template

    def apply_custom_domain(
        self,
        project: SimulationProject,
        size: tuple[float, float, float],
        cells: tuple[int, int, int],
    ) -> ComputationDomainTemplate:
        if any(value <= 0 for value in size):
            raise ValueError("计算域长宽高必须大于 0。")
        if any(value <= 0 for value in cells):
            raise ValueError("网格数必须大于 0。")
        template = ComputationDomainTemplate(
            key="custom_domain",
            name=f"自定义：{size[0]:g} x {size[1]:g} x {size[2]:g}",
            level="自定义",
            size=size,
            cells=cells,
            suggested_location_in_mesh=(size[0] * 0.1, size[1] * 0.5, size[2] * 0.5),
            description="用户手动输入的计算域。",
        )
        self._write_domain(project, template)
        return template

    def load_boundary_conditions(self, project: SimulationProject) -> BoundaryConditionSettings:
        config_path = self._boundary_config_path(project.case_dir)
        if not config_path.exists():
            return BoundaryConditionSettings()
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            velocity = payload.get("inlet_velocity", [1.0, 0.0, 0.0])
            if not isinstance(velocity, list | tuple) or len(velocity) != 3:
                velocity = [1.0, 0.0, 0.0]
            wall_type = str(payload.get("wall_type", "noSlip"))
            if wall_type not in {"noSlip", "slip"}:
                wall_type = "noSlip"
            return BoundaryConditionSettings(
                inlet_velocity=(float(velocity[0]), float(velocity[1]), float(velocity[2])),
                outlet_pressure=float(payload.get("outlet_pressure", 0.0)),
                wall_type=wall_type,
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return BoundaryConditionSettings()

    def apply_boundary_conditions(
        self,
        project: SimulationProject,
        settings: BoundaryConditionSettings,
    ) -> list[Path]:
        boundary_names = self._extract_boundary_names(project.case_dir / "system" / "blockMeshDict")
        if not boundary_names:
            boundary_names = ("inlet", "outlet", "fixedWalls")
        written_files: list[Path] = []
        velocity_field = project.case_dir / "0" / "U"
        pressure_field = project.case_dir / "0" / "p"
        velocity_field.write_text(self._default_velocity_field(boundary_names, settings), encoding="utf-8")
        pressure_field.write_text(self._default_pressure_field(boundary_names, settings), encoding="utf-8")
        written_files.extend([velocity_field, pressure_field])
        config_path = self._boundary_config_path(project.case_dir)
        config_path.write_text(
            json.dumps(
                {
                    "inlet_velocity": list(settings.inlet_velocity),
                    "outlet_pressure": settings.outlet_pressure,
                    "wall_type": settings.wall_type,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        written_files.append(config_path)
        return written_files

    def _write_domain(self, project: SimulationProject, template: ComputationDomainTemplate) -> None:
        block_mesh_dict = project.case_dir / "system" / "blockMeshDict"
        block_mesh_dict.parent.mkdir(parents=True, exist_ok=True)
        block_mesh_dict.write_text(self._block_mesh_dict_for_template(template), encoding="utf-8")
        self._sync_field_boundaries_for_names(project.case_dir, ("inlet", "outlet", "fixedWalls"))
        config_path = self._domain_config_path(project.case_dir)
        config_path.write_text(
            json.dumps(
                {
                    "key": template.key,
                    "name": template.name,
                    "size": list(template.size),
                    "cells": list(template.cells),
                    "suggested_location_in_mesh": list(template.suggested_location_in_mesh),
                    "shape": template.shape,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def list_projects(self) -> list[SimulationProject]:
        workspace = self._projects_dir()
        workspace.mkdir(parents=True, exist_ok=True)
        projects: list[SimulationProject] = []
        for metadata_file in sorted(workspace.glob("*/project.json")):
            try:
                projects.append(self.open_project(metadata_file.parent))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return projects

    def create_project(self, name: str) -> SimulationProject:
        clean_name = self._normalize_project_name(name)
        project_dir = self._projects_dir() / clean_name
        if project_dir.exists():
            raise ValueError(f"项目已存在：{clean_name}")

        case_dir = project_dir / "case"
        for path in (
            case_dir / "0",
            case_dir / "constant",
            case_dir / "system",
            project_dir / "logs",
            project_dir / "results",
        ):
            path.mkdir(parents=True, exist_ok=True)

        self._write_minimal_case_template(case_dir, overwrite=True)
        metadata = {
            "name": clean_name,
            "case_dir": "case",
            "current_case": "case",
        }
        (project_dir / "project.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return SimulationProject(name=clean_name, path=project_dir, case_dir=case_dir)

    def open_project(self, path: Path) -> SimulationProject:
        project_dir = path.expanduser().resolve()
        metadata_file = project_dir / "project.json"
        if not metadata_file.exists():
            raise ValueError("所选目录不是 FoamDesk 项目，缺少 project.json。")

        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        name = str(payload["name"])
        case_name = str(payload.get("current_case") or payload.get("case_dir", "case"))
        case_dir = project_dir / case_name
        if not case_dir.exists():
            raise ValueError("项目 case 目录不存在。")
        return SimulationProject(name=name, path=project_dir, case_dir=case_dir, case_name=case_dir.name)

    def list_cases(self, project: SimulationProject) -> list[str]:
        case_names: list[str] = []
        for path in sorted(project.path.iterdir()):
            if path.is_dir() and (path / "system").exists() and (path / "constant").exists():
                case_names.append(path.name)
        return case_names

    def create_case(self, project: SimulationProject, name: str) -> SimulationProject:
        clean_name = self._normalize_project_name(name)
        case_dir = project.path / clean_name
        if case_dir.exists():
            raise ValueError(f"Case 已存在：{clean_name}")
        for path in (
            case_dir / "0",
            case_dir / "constant",
            case_dir / "system",
        ):
            path.mkdir(parents=True, exist_ok=True)
        self._write_minimal_case_template(case_dir, overwrite=True)
        new_project = SimulationProject(
            name=project.name,
            path=project.path,
            case_dir=case_dir,
            case_name=clean_name,
        )
        self._write_current_case(new_project)
        return new_project

    def switch_case(self, project: SimulationProject, case_name: str) -> SimulationProject:
        clean_name = self._normalize_project_name(case_name)
        case_dir = project.path / clean_name
        if not case_dir.exists():
            raise ValueError(f"Case 不存在：{clean_name}")
        if not (case_dir / "system").exists() or not (case_dir / "constant").exists():
            raise ValueError(f"所选目录不是有效 Case：{clean_name}")
        switched_project = SimulationProject(
            name=project.name,
            path=project.path,
            case_dir=case_dir,
            case_name=clean_name,
        )
        self._write_current_case(switched_project)
        return switched_project

    def remember_project(self, project: SimulationProject) -> None:
        self._write_current_case(project)
        settings = self._settings_service.load()
        self._settings_service.save(replace(settings, last_project_path=project.path))

    def open_last_project(self) -> SimulationProject | None:
        last_project_path = self._settings_service.load().last_project_path
        if last_project_path is None:
            return None
        try:
            return self.open_project(last_project_path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def ensure_minimal_case_template(self, project: SimulationProject) -> list[Path]:
        """Backfill missing files for projects created by older templates."""
        written_files = self._write_minimal_case_template(project.case_dir, overwrite=False)
        written_files.extend(self._sync_field_boundaries(project.case_dir))
        return written_files

    def ensure_field_boundaries(
        self,
        project: SimulationProject,
        boundary_names: tuple[str, ...],
    ) -> list[Path]:
        """Ensure U and p contain all boundaries required by the current mesh workflow."""
        clean_names = tuple(dict.fromkeys(name for name in boundary_names if name.strip()))
        if not clean_names:
            return []
        return self._sync_field_boundaries_for_names(project.case_dir, clean_names)

    def _projects_dir(self) -> Path:
        return self._settings_service.load().workspace_dir / "projects"

    def _write_current_case(self, project: SimulationProject) -> None:
        metadata_file = project.path / "project.json"
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        payload["current_case"] = project.case_name
        payload.setdefault("case_dir", "case")
        metadata_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _normalize_project_name(self, name: str) -> str:
        clean_name = name.strip().replace(" ", "_")
        if not clean_name:
            raise ValueError("项目名称不能为空。")
        invalid_chars = set('/\\:*?"<>|')
        if any(char in invalid_chars for char in clean_name):
            raise ValueError("项目名称包含非法路径字符。")
        return clean_name

    def _write_minimal_case_template(self, case_dir: Path, *, overwrite: bool) -> list[Path]:
        files = {
            case_dir / "system" / "blockMeshDict": self._default_block_mesh_dict(),
            case_dir / "0" / "U": self._default_velocity_field(("inlet", "outlet", "fixedWalls")),
            case_dir / "0" / "p": self._default_pressure_field(("inlet", "outlet", "fixedWalls")),
            case_dir / "constant" / "transportProperties": self._default_transport_properties(),
            case_dir / "constant" / "physicalProperties": self._default_physical_properties(),
            case_dir / "system" / "controlDict": self._default_control_dict(),
            case_dir / "system" / "fvSchemes": self._default_fv_schemes(),
            case_dir / "system" / "fvSolution": self._default_fv_solution(),
        }
        written_files: list[Path] = []
        for path, content in files.items():
            if path.exists() and not overwrite:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            written_files.append(path)
        return written_files

    def _domain_config_path(self, case_dir: Path) -> Path:
        return case_dir / "system" / "domain_config.json"

    def _boundary_config_path(self, case_dir: Path) -> Path:
        return case_dir / "system" / "boundary_config.json"

    def _domain_template_by_key(self, template_key: str) -> ComputationDomainTemplate:
        for template in self.domain_templates():
            if template.key == template_key:
                return template
        raise ValueError(f"未知计算域模板：{template_key}")

    def _sync_field_boundaries(self, case_dir: Path) -> list[Path]:
        boundary_names = self._extract_boundary_names(case_dir / "system" / "blockMeshDict")
        if not boundary_names:
            return []
        return self._sync_field_boundaries_for_names(case_dir, boundary_names)

    def _sync_field_boundaries_for_names(self, case_dir: Path, boundary_names: tuple[str, ...]) -> list[Path]:
        written_files: list[Path] = []
        velocity_field = case_dir / "0" / "U"
        pressure_field = case_dir / "0" / "p"
        if not self._field_has_boundaries(velocity_field, boundary_names):
            velocity_field.write_text(self._default_velocity_field(boundary_names), encoding="utf-8")
            written_files.append(velocity_field)
        if not self._field_has_boundaries(pressure_field, boundary_names):
            pressure_field.write_text(self._default_pressure_field(boundary_names), encoding="utf-8")
            written_files.append(pressure_field)
        return written_files

    def _extract_boundary_names(self, block_mesh_dict: Path) -> tuple[str, ...]:
        if not block_mesh_dict.exists():
            return ()
        content = block_mesh_dict.read_text(encoding="utf-8")
        boundary_match = re.search(r"\bboundary\s*\(", content)
        if not boundary_match:
            return ()

        start_index = boundary_match.end() - 1
        depth = 0
        end_index: int | None = None
        for index in range(start_index, len(content)):
            char = content[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end_index = index
                    break
        if end_index is None:
            return ()

        names: list[str] = []
        lines = [line.strip() for line in content[start_index + 1 : end_index].splitlines()]
        for index, line in enumerate(lines[:-1]):
            if not line or line.startswith("//") or line in {"(", ")", "{", "}"}:
                continue
            if line.endswith(";") or line in {"type", "faces"}:
                continue
            if lines[index + 1] == "{":
                names.append(line)
        return tuple(names)

    def _field_has_boundaries(self, field_path: Path, boundary_names: tuple[str, ...]) -> bool:
        if not field_path.exists():
            return False
        content = field_path.read_text(encoding="utf-8")
        return all(re.search(rf"\b{re.escape(name)}\s*\{{", content) for name in boundary_names)

    def _block_mesh_dict_for_template(self, template: ComputationDomainTemplate) -> str:
        cell_x, cell_y, cell_z = template.cells
        vertices = self.domain_vertices(template)
        vertex_lines = "\n".join(f"    ({x:g} {y:g} {z:g})" for x, y, z in vertices)
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

convertToMeters 1;

vertices
(
{vertex_lines}
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({cell_x} {cell_y} {cell_z}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    outlet
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    fixedWalls
    {{
        type wall;
        faces
        (
            (0 1 5 4)
            (0 3 2 1)
            (4 5 6 7)
            (3 7 6 2)
        );
    }}
);

mergePatchPairs
(
);
"""

    def domain_vertices(self, template: ComputationDomainTemplate) -> tuple[tuple[float, float, float], ...]:
        length_x, length_y, length_z = template.size
        if template.shape == "tapered":
            y_margin = length_y * 0.24
            z_margin = length_z * 0.24
            return (
                (0.0, y_margin, z_margin),
                (length_x, 0.0, 0.0),
                (length_x, length_y, 0.0),
                (0.0, length_y - y_margin, z_margin),
                (0.0, y_margin, length_z - z_margin),
                (length_x, 0.0, length_z),
                (length_x, length_y, length_z),
                (0.0, length_y - y_margin, length_z - z_margin),
            )
        if template.shape == "ramp":
            ramp_height = length_z * 0.24
            return (
                (0.0, 0.0, 0.0),
                (length_x, 0.0, ramp_height),
                (length_x, length_y, ramp_height),
                (0.0, length_y, 0.0),
                (0.0, 0.0, length_z),
                (length_x, 0.0, length_z),
                (length_x, length_y, length_z),
                (0.0, length_y, length_z),
            )
        return (
            (0.0, 0.0, 0.0),
            (length_x, 0.0, 0.0),
            (length_x, length_y, 0.0),
            (0.0, length_y, 0.0),
            (0.0, 0.0, length_z),
            (length_x, 0.0, length_z),
            (length_x, length_y, length_z),
            (0.0, length_y, length_z),
        )

    def _default_block_mesh_dict(self) -> str:
        return self._block_mesh_dict_for_template(self.domain_templates()[0])

    def _default_velocity_field(
        self,
        boundary_names: tuple[str, ...],
        settings: BoundaryConditionSettings | None = None,
    ) -> str:
        resolved_settings = settings or BoundaryConditionSettings()
        boundary_field = "\n".join(self._velocity_boundary_entry(name, resolved_settings) for name in boundary_names)
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}

dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);

boundaryField
{
__BOUNDARY_FIELD__
}
""".replace("__BOUNDARY_FIELD__", boundary_field)

    def _default_pressure_field(
        self,
        boundary_names: tuple[str, ...],
        settings: BoundaryConditionSettings | None = None,
    ) -> str:
        resolved_settings = settings or BoundaryConditionSettings()
        boundary_field = "\n".join(self._pressure_boundary_entry(name, resolved_settings) for name in boundary_names)
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}

dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{
__BOUNDARY_FIELD__
}
""".replace("__BOUNDARY_FIELD__", boundary_field)

    def _velocity_boundary_entry(self, name: str, settings: BoundaryConditionSettings) -> str:
        normalized_name = name.lower()
        if "movingwall" in normalized_name or "inlet" in normalized_name:
            ux, uy, uz = settings.inlet_velocity
            return f"""    {name}
    {{
        type            fixedValue;
        value           uniform ({ux:g} {uy:g} {uz:g});
    }}
"""
        if "outlet" in normalized_name:
            return f"""    {name}
    {{
        type            zeroGradient;
    }}
"""
        return f"""    {name}
    {{
        type            {settings.wall_type};
    }}
"""

    def _pressure_boundary_entry(self, name: str, settings: BoundaryConditionSettings) -> str:
        normalized_name = name.lower()
        if "outlet" in normalized_name:
            return f"""    {name}
    {{
        type            fixedValue;
        value           uniform {settings.outlet_pressure:g};
    }}
"""
        return f"""    {name}
    {{
        type            zeroGradient;
    }}
"""

    def _default_transport_properties(self) -> str:
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}

nu              [0 2 -1 0 0 0 0] 0.01;
"""

    def _default_physical_properties(self) -> str:
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      physicalProperties;
}

nu              [0 2 -1 0 0 0 0] 0.01;
"""

    def _default_control_dict(self) -> str:
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application     icoFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         0.5;
deltaT          0.005;
writeControl    timeStep;
writeInterval   20;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""

    def _default_fv_schemes(self) -> str:
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
"""

    def _default_fv_solution(self) -> str:
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers
{
    p
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-06;
        relTol          0.05;
    }

    pFinal
    {
        $p;
        relTol          0;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0;
    }
}

PISO
{
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}
"""
