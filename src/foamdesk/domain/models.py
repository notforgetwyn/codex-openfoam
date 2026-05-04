from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class OpenFOAMEnvironmentStatus:
    is_available: bool
    bash_path: str
    env_script_path: str | None
    foam_version: str | None
    detail: str


@dataclass(slots=True)
class AppSettings:
    workspace_dir: Path
    openfoam_env_script: str | None
    theme_name: str
    background_color: str
    font_family: str
    font_size: int
    last_project_path: Path | None


@dataclass(slots=True)
class SimulationProject:
    name: str
    path: Path
    case_dir: Path
    case_name: str = "case"


@dataclass(slots=True)
class SimulationParameters:
    solver_name: str
    end_time: float
    delta_t: float
    write_interval: int
    max_iterations: int
    residual_tolerance: float
    density: float
    viscosity: float
    turbulence_model: str
    numeric_scheme: str
    fv_solution_preset: str


@dataclass(slots=True)
class OpenFoamDiagnostic:
    title: str
    detail: str
    suggestion: str


@dataclass(slots=True)
class CaseResultIndex:
    time_directories: list[str]
    latest_time: str | None
    field_files: list[str]
    has_mesh: bool
    mesh_path: Path


@dataclass(slots=True)
class SolverResidual:
    time: float | None
    field: str
    initial: float
    final: float
    iterations: int


@dataclass(slots=True)
class SolverMetrics:
    times: list[float]
    residuals: list[SolverResidual]
    courant_max: float | None
    latest_continuity_global: float | None


@dataclass(slots=True)
class OpenFoamVtkCaseInfo:
    marker_file: Path
    time_values: list[float]
    block_count: int
    point_arrays: list[str]
    cell_arrays: list[str]
