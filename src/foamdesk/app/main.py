from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from foamdesk.app.bootstrap import ApplicationContext
from foamdesk.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    context = ApplicationContext(project_root=Path.cwd())
    window = MainWindow(context)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

