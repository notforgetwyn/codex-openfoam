from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from foamdesk.app.bootstrap import ApplicationContext
from foamdesk.ui.main_window import MainWindow
from foamdesk.ui.startup_window import StartupWindow


def main() -> int:
    app = QApplication(sys.argv)
    context = ApplicationContext(project_root=Path.cwd())
    initial_project = context.project_service.open_last_project()
    if initial_project is None:
        startup_window = StartupWindow(context)
        if startup_window.exec() != 1:
            return 0
        initial_project = startup_window.selected_project
        if initial_project is None:
            return 0

    window = MainWindow(context, initial_project=initial_project)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
