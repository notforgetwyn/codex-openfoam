from __future__ import annotations

from pathlib import Path

import yaml

from foamdesk.domain.models import AppSettings


class AppSettingsService:
    """Loads and stores local application settings."""

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root
        self._config_dir = project_root / "config"
        self._settings_file = self._config_dir / "settings.yaml"

    def load(self) -> AppSettings:
        self._ensure_defaults()
        payload = yaml.safe_load(self._settings_file.read_text(encoding="utf-8")) or {}
        workspace_dir = Path(payload["workspace_dir"])
        env_script = payload.get("openfoam_env_script")
        theme_name = payload.get("theme_name", "vscode-dark")
        background_color = payload.get("background_color", "#1e1e1e")
        font_family = payload.get("font_family", "Noto Sans CJK SC")
        font_size = int(payload.get("font_size", 15))
        return AppSettings(
            workspace_dir=workspace_dir,
            openfoam_env_script=env_script,
            theme_name=theme_name,
            background_color=background_color,
            font_family=font_family,
            font_size=font_size,
        )

    def save(self, settings: AppSettings) -> None:
        self._config_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "workspace_dir": str(settings.workspace_dir),
            "openfoam_env_script": settings.openfoam_env_script,
            "theme_name": settings.theme_name,
            "background_color": settings.background_color,
            "font_family": settings.font_family,
            "font_size": settings.font_size,
        }
        self._settings_file.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def _ensure_defaults(self) -> None:
        self._config_dir.mkdir(parents=True, exist_ok=True)
        if self._settings_file.exists():
            return

        default_settings = {
            "workspace_dir": str(self._project_root / "workspace"),
            "openfoam_env_script": None,
            "theme_name": "vscode-dark",
            "background_color": "#1e1e1e",
            "font_family": "Noto Sans CJK SC",
            "font_size": 15,
        }
        self._settings_file.write_text(
            yaml.safe_dump(default_settings, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
