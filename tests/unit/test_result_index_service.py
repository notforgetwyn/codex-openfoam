from __future__ import annotations

from pathlib import Path

from foamdesk.services.project_service import ProjectService
from foamdesk.services.result_index_service import ResultIndexService
from foamdesk.services.settings_service import AppSettingsService


def test_result_index_service_indexes_time_dirs_fields_and_mesh(tmp_path: Path) -> None:
    project = ProjectService(AppSettingsService(tmp_path)).create_project("demo")
    mesh_dir = project.case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "points").write_text("points", encoding="utf-8")
    for time_name in ("0.1", "0.02"):
        time_dir = project.case_dir / time_name
        time_dir.mkdir()
        (time_dir / "U").write_text("velocity", encoding="utf-8")
        (time_dir / "p").write_text("pressure", encoding="utf-8")
        (time_dir / "ignored.txt").write_text("ignored", encoding="utf-8")

    result_index = ResultIndexService().index(project)

    assert result_index.has_mesh is True
    assert result_index.time_directories == ["0", "0.02", "0.1"]
    assert result_index.latest_time == "0.1"
    assert "0.1/U" in result_index.field_files
    assert "0.1/p" in result_index.field_files
    assert all("ignored" not in field for field in result_index.field_files)


def test_result_index_service_formats_empty_index(tmp_path: Path) -> None:
    project = ProjectService(AppSettingsService(tmp_path)).create_project("demo")

    text = ResultIndexService().format_index(ResultIndexService().index(project))

    assert "时间步目录：0" in text
    assert "最新时间步：0" in text
