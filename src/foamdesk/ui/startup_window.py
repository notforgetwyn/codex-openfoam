from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from foamdesk.app.bootstrap import ApplicationContext
from foamdesk.domain.models import SimulationProject
from foamdesk.ui.theme import build_stylesheet


class StartupWindow(QDialog):
    """Project-first startup screen shown when no valid remembered project exists."""

    def __init__(self, context: ApplicationContext) -> None:
        super().__init__()
        self._context = context
        self.selected_project: SimulationProject | None = None

        self.setWindowTitle("FoamDesk - 选择项目")
        self.resize(920, 620)
        self.setMinimumSize(760, 480)
        self._apply_theme()
        self._build_ui()
        self._load_projects()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QVBoxLayout()
        sidebar.setContentsMargins(28, 32, 24, 28)
        sidebar.setSpacing(14)
        product_title = QLabel("FoamDesk")
        product_title.setStyleSheet("font-size: 28px; font-weight: 700;")
        product_subtitle = QLabel("OpenFOAM 桌面仿真客户端")
        product_subtitle.setStyleSheet("color: #9da5b4;")
        projects_label = QLabel("Projects")
        projects_label.setStyleSheet(
            "margin-top: 28px; padding: 10px 14px; background: #094771; font-weight: 600;"
        )
        hint = QLabel("先选择或创建一个仿真项目，然后进入主工作台。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9da5b4;")
        sidebar.addWidget(product_title)
        sidebar.addWidget(product_subtitle)
        sidebar.addWidget(projects_label)
        sidebar.addWidget(hint)
        sidebar.addStretch(1)

        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar)
        sidebar_widget.setFixedWidth(270)
        sidebar_widget.setStyleSheet("background: #252526;")

        content = QVBoxLayout()
        content.setContentsMargins(32, 28, 32, 28)
        content.setSpacing(18)

        header = QHBoxLayout()
        title = QLabel("选择项目")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        new_button = QPushButton("新建项目")
        open_button = QPushButton("打开项目")
        refresh_button = QPushButton("刷新")
        new_button.clicked.connect(self._create_project)
        open_button.clicked.connect(self._open_project_from_disk)
        refresh_button.clicked.connect(self._load_projects)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(new_button)
        header.addWidget(open_button)
        header.addWidget(refresh_button)

        self._project_list = QListWidget()
        self._project_list.itemDoubleClicked.connect(self._accept_item)
        self._project_list.itemActivated.connect(self._accept_item)
        self._project_list.setStyleSheet("font-size: 16px;")

        footer = QLabel("双击项目进入主工作台。FoamDesk 会记住最后一次打开的项目。")
        footer.setStyleSheet("color: #9da5b4;")

        content.addLayout(header)
        content.addWidget(self._project_list, 1)
        content.addWidget(footer)

        content_widget = QWidget()
        content_widget.setLayout(content)
        content_widget.setStyleSheet("background: #1e1e1e;")

        root.addWidget(sidebar_widget)
        root.addWidget(content_widget, 1)

    def _apply_theme(self) -> None:
        settings = self._context.settings_service.load()
        self.setStyleSheet(
            build_stylesheet(
                settings.theme_name,
                settings.background_color,
                settings.font_family,
                settings.font_size,
            )
        )

    def _load_projects(self) -> None:
        self._project_list.clear()
        projects = self._context.project_service.list_projects()
        if not projects:
            item = QListWidgetItem("暂无项目，请点击右上角“新建项目”。")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self._project_list.addItem(item)
            return

        for project in projects:
            item = QListWidgetItem(f"{project.name}\n{project.path}")
            item.setData(Qt.ItemDataRole.UserRole, str(project.path))
            self._project_list.addItem(item)

    def _accept_item(self, item: QListWidgetItem) -> None:
        project_path = item.data(Qt.ItemDataRole.UserRole)
        if not project_path:
            return
        try:
            self._select_project(self._context.project_service.open_project(Path(project_path)))
        except ValueError as error:
            self._show_error(str(error))

    def _create_project(self) -> None:
        name, ok = QInputDialog.getText(self, "新建项目", "项目名称")
        if not ok:
            return
        try:
            project = self._context.project_service.create_project(name)
        except ValueError as error:
            self._show_error(str(error))
            return
        self._select_project(project)

    def _open_project_from_disk(self) -> None:
        settings = self._context.settings_service.load()
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "打开 FoamDesk 项目",
            str(settings.workspace_dir / "projects"),
        )
        if not selected_dir:
            return
        try:
            project = self._context.project_service.open_project(Path(selected_dir))
        except ValueError as error:
            self._show_error(str(error))
            return
        self._select_project(project)

    def _select_project(self, project: SimulationProject) -> None:
        self._context.project_service.remember_project(project)
        self.selected_project = project
        self.accept()

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "FoamDesk", message)
