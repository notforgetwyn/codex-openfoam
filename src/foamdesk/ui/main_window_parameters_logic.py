from __future__ import annotations

from PySide6.QtWidgets import QComboBox

from foamdesk.domain.models import SimulationParameters


class ParametersLogicMixin:
    def _show_parameters(self, parameters: SimulationParameters) -> None:
        pass
