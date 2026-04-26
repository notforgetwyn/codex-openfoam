from __future__ import annotations

from pathlib import Path

from foamdesk.domain.models import CaseResultIndex, SimulationProject


class ResultIndexService:
    """Indexes the first OpenFOAM result artifacts FoamDesk can show."""

    FIELD_FILE_NAMES = {"U", "p", "T", "k", "epsilon", "omega", "nut", "phi"}

    def index(self, project: SimulationProject) -> CaseResultIndex:
        case_dir = project.case_dir
        time_directories = sorted(
            (path.name for path in case_dir.iterdir() if path.is_dir() and self._is_time_dir(path.name)),
            key=self._time_sort_key,
        )
        latest_time = time_directories[-1] if time_directories else None
        field_files = self._collect_field_files(case_dir, time_directories)
        mesh_path = case_dir / "constant" / "polyMesh"
        return CaseResultIndex(
            time_directories=time_directories,
            latest_time=latest_time,
            field_files=field_files,
            has_mesh=mesh_path.exists(),
            mesh_path=mesh_path,
        )

    def format_index(self, index: CaseResultIndex) -> str:
        time_dirs = ", ".join(index.time_directories) if index.time_directories else "未发现"
        fields = "\n".join(f"- {field}" for field in index.field_files) or "- 未发现字段文件"
        mesh_status = "已生成" if index.has_mesh else "未生成"
        latest_time = index.latest_time or "无"
        return (
            "结果索引\n\n"
            f"- 网格目录：{mesh_status}\n"
            f"- 网格路径：{index.mesh_path}\n"
            f"- 时间步目录：{time_dirs}\n"
            f"- 最新时间步：{latest_time}\n\n"
            "字段文件：\n"
            f"{fields}\n"
        )

    def _collect_field_files(self, case_dir: Path, time_directories: list[str]) -> list[str]:
        fields: list[str] = []
        for time_dir in time_directories:
            directory = case_dir / time_dir
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.name in self.FIELD_FILE_NAMES:
                    fields.append(str(path.relative_to(case_dir)))
        return fields

    def _is_time_dir(self, name: str) -> bool:
        try:
            float(name)
        except ValueError:
            return False
        return True

    def _time_sort_key(self, name: str) -> float:
        return float(name)
