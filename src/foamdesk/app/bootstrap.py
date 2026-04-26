from __future__ import annotations

from pathlib import Path

from foamdesk.integrations.openfoam.environment import OpenFOAMEnvironmentDetector
from foamdesk.services.case_parameter_service import OpenFoamCaseParameterService
from foamdesk.services.log_diagnostic_service import OpenFoamLogDiagnosticService
from foamdesk.services.project_service import ProjectService
from foamdesk.services.settings_service import AppSettingsService


class ApplicationContext:
    """Holds shared services for the desktop application."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.settings_service = AppSettingsService(project_root)
        self.project_service = ProjectService(self.settings_service)
        self.case_parameter_service = OpenFoamCaseParameterService()
        self.log_diagnostic_service = OpenFoamLogDiagnosticService()
        self.environment_detector = OpenFOAMEnvironmentDetector(self.settings_service)
