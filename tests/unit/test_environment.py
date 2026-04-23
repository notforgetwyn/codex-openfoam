from __future__ import annotations

from pathlib import Path

from foamdesk.services.settings_service import AppSettingsService


def test_settings_service_creates_default_file(tmp_path: Path) -> None:
    service = AppSettingsService(tmp_path)

    settings = service.load()

    assert settings.workspace_dir == tmp_path / "workspace"
    assert settings.openfoam_env_script is None
    assert settings.theme_name == "vscode-dark"
    assert settings.background_color == "#1e1e1e"
    assert (tmp_path / "config" / "settings.yaml").exists()
