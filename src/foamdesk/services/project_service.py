from __future__ import annotations

import json
import re
from pathlib import Path

from foamdesk.domain.models import SimulationProject
from foamdesk.services.settings_service import AppSettingsService


class ProjectService:
    """Creates and discovers local simulation projects in the configured workspace."""

    def __init__(self, settings_service: AppSettingsService) -> None:
        self._settings_service = settings_service

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
        case_dir = project_dir / str(payload.get("case_dir", "case"))
        if not case_dir.exists():
            raise ValueError("项目 case 目录不存在。")
        return SimulationProject(name=name, path=project_dir, case_dir=case_dir)

    def ensure_minimal_case_template(self, project: SimulationProject) -> list[Path]:
        """Backfill missing files for projects created by older templates."""
        written_files = self._write_minimal_case_template(project.case_dir, overwrite=False)
        written_files.extend(self._sync_field_boundaries(project.case_dir))
        return written_files

    def _projects_dir(self) -> Path:
        return self._settings_service.load().workspace_dir / "projects"

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
            case_dir / "0" / "U": self._default_velocity_field(("movingWall", "fixedWalls")),
            case_dir / "0" / "p": self._default_pressure_field(("movingWall", "fixedWalls")),
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

    def _sync_field_boundaries(self, case_dir: Path) -> list[Path]:
        boundary_names = self._extract_boundary_names(case_dir / "system" / "blockMeshDict")
        if not boundary_names:
            return []

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

    def _default_block_mesh_dict(self) -> str:
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}

convertToMeters 1;

vertices
(
    (0 0 0)
    (1 0 0)
    (1 1 0)
    (0 1 0)
    (0 0 1)
    (1 0 1)
    (1 1 1)
    (0 1 1)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (10 10 10) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    movingWall
    {
        type wall;
        faces ((3 7 6 2));
    }
    fixedWalls
    {
        type wall;
        faces
        (
            (0 4 7 3)
            (1 2 6 5)
            (0 1 5 4)
            (0 3 2 1)
            (4 5 6 7)
        );
    }
);

mergePatchPairs
(
);
"""

    def _default_velocity_field(self, boundary_names: tuple[str, ...]) -> str:
        boundary_field = "\n".join(self._velocity_boundary_entry(name) for name in boundary_names)
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

    def _default_pressure_field(self, boundary_names: tuple[str, ...]) -> str:
        boundary_field = "\n".join(self._pressure_boundary_entry(name) for name in boundary_names)
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

    def _velocity_boundary_entry(self, name: str) -> str:
        normalized_name = name.lower()
        if "movingwall" in normalized_name or "inlet" in normalized_name:
            return f"""    {name}
    {{
        type            fixedValue;
        value           uniform (1 0 0);
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
        type            noSlip;
    }}
"""

    def _pressure_boundary_entry(self, name: str) -> str:
        normalized_name = name.lower()
        if "outlet" in normalized_name:
            return f"""    {name}
    {{
        type            fixedValue;
        value           uniform 0;
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
