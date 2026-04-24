from __future__ import annotations

from pathlib import Path

from foamdesk.domain.models import AppSettings
from foamdesk.services.settings_service import AppSettingsService


def test_settings_service_creates_default_file(tmp_path: Path) -> None:
    service = AppSettingsService(tmp_path)

    settings = service.load()

    assert settings.workspace_dir == tmp_path / "workspace"
    assert settings.openfoam_env_script is None
    assert settings.theme_name == "vscode-dark"
    assert settings.background_color == "#1e1e1e"
    assert settings.font_family == "Noto Sans CJK SC"
    assert settings.font_size == 15
    assert (tmp_path / "config" / "settings.yaml").exists()


def test_settings_service_saves_roundtrip(tmp_path: Path) -> None:
    service = AppSettingsService(tmp_path)
    service.save(
        AppSettings(
            workspace_dir=tmp_path / "workspace-alt",
            openfoam_env_script="/opt/openfoam/etc/bashrc",
            theme_name="vscode-blue",
            background_color="#151b23",
            font_family="DejaVu Sans",
            font_size=18,
        )
    )

    settings = service.load()

    assert settings.workspace_dir == tmp_path / "workspace-alt"
    assert settings.openfoam_env_script == "/opt/openfoam/etc/bashrc"
    assert settings.theme_name == "vscode-blue"
    assert settings.background_color == "#151b23"
    assert settings.font_family == "DejaVu Sans"
    assert settings.font_size == 18
