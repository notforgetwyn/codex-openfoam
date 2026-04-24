from __future__ import annotations

import json
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

        (case_dir / "system" / "blockMeshDict").write_text(
            self._default_block_mesh_dict(),
            encoding="utf-8",
        )
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
    inlet
    {
        type patch;
        faces ((0 4 7 3));
    }
    outlet
    {
        type patch;
        faces ((1 2 6 5));
    }
    walls
    {
        type wall;
        faces
        (
            (0 1 5 4)
            (3 7 6 2)
            (0 3 2 1)
            (4 5 6 7)
        );
    }
);

mergePatchPairs
(
);
"""
