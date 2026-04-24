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


@dataclass(slots=True)
class SimulationProject:
    name: str
    path: Path
    case_dir: Path
