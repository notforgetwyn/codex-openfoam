from __future__ import annotations

from pathlib import Path

from foamdesk.integrations.openfoam.environment import OpenFOAMEnvironmentDetector
from foamdesk.services.settings_service import AppSettingsService


class ApplicationContext:
    """Holds shared services for the desktop application."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.settings_service = AppSettingsService(project_root)
        self.environment_detector = OpenFOAMEnvironmentDetector(self.settings_service)

