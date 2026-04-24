from __future__ import annotations

import shlex
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtCore import QProcess
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QCheckBox,
    QFileDialog,
    QFontComboBox,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from foamdesk.app.bootstrap import ApplicationContext
from foamdesk.domain.models import SimulationProject
from foamdesk.ui.theme import THEMES, build_stylesheet


class WindowTitleBar(QFrame):
    def __init__(self, window: QMainWindow) -> None:
        super().__init__(window)
        self._window = window
        self.setObjectName("customTitleBar")
        self.setFixedHeight(36)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 4, 0)
        layout.setSpacing(6)

        title = QLabel("FoamDesk")
        title.setObjectName("windowTitleLabel")
        layout.addStretch(1)
        layout.addWidget(title)
        layout.addStretch(1)

        minimize_button = QPushButton("—")
        maximize_button = QPushButton("□")
        close_button = QPushButton("×")
        for button in (minimize_button, maximize_button, close_button):
            button.setObjectName("windowControlButton")
            button.setFixedSize(42, 30)
        close_button.setObjectName("windowCloseButton")

        minimize_button.clicked.connect(window.showMinimized)
        maximize_button.clicked.connect(self._toggle_maximized)
        close_button.clicked.connect(window.close)

        layout.addWidget(minimize_button)
        layout.addWidget(maximize_button)
        layout.addWidget(close_button)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._toggle_maximized()
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._window.windowHandle():
            self._window.windowHandle().startSystemMove()
        super().mousePressEvent(event)

    def _toggle_maximized(self) -> None:
        if self._window.isMaximized():
            self._window.showNormal()
        else:
            self._window.showMaximized()


class MainWindow(QMainWindow):
    def __init__(self, context: ApplicationContext) -> None:
        super().__init__()
        self._context = context
        self._theme_names = list(THEMES.keys())
        self._theme_index = 0
        self._current_project: SimulationProject | None = None
        self._foam_process: QProcess | None = None
        self.setWindowTitle("FoamDesk")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.resize(1400, 900)
        self._build_ui()
        self._apply_settings_theme()
        self._refresh_status_bar()
        if self._context.settings_service.load().show_tutorial_on_startup:
            self._show_tutorial()

    def _build_ui(self) -> None:
        self._build_status_bar()

        workbench = QSplitter(Qt.Orientation.Horizontal)
        workbench.addWidget(self._build_activity_bar())
        workbench.addWidget(self._build_sidebar())
        workbench.addWidget(self._build_editor_panel())
        workbench.setSizes([72, 300, 1028])
        workbench.setChildrenCollapsible(False)

        shell = QWidget()
        shell.setObjectName("appShell")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        shell_layout.addWidget(WindowTitleBar(self))
        shell_layout.addWidget(self._build_menu_bar())
        shell_layout.addWidget(self._build_toolbar())
        shell_layout.addWidget(workbench, 1)
        self.setCentralWidget(shell)

    def _build_menu_bar(self) -> QWidget:
        menu_bar = QMenuBar(self)
        menu_bar.setObjectName("topMenuBar")
        menu_bar.setFixedHeight(34)
        menu_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        file_menu = menu_bar.addMenu("文件")
        file_menu.addAction("新建项目", self._create_project)
        file_menu.addAction("打开项目", self._open_project)
        file_menu.addAction("保存设置", self._save_current_state)
        file_menu.addSeparator()
        file_menu.addAction("退出", self.close)

        project_menu = menu_bar.addMenu("项目")
        project_menu.addAction("刷新项目树", self._refresh_project_tree)
        project_menu.addAction("搜索项目", self._search_projects)

        case_menu = menu_bar.addMenu("Case")
        case_menu.addAction("打开当前 Case 目录", self._show_current_case_path)

        solver_menu = menu_bar.addMenu("求解器")
        solver_menu.addAction("运行 blockMesh", self._run_block_mesh)
        solver_menu.addAction("停止当前任务", self._stop_current_process)

        tools_menu = menu_bar.addMenu("工具")
        tools_menu.addAction("环境检查", self._open_environment_tab)
        tools_menu.addAction("设置", self._open_settings_tab)
        tools_menu.addAction("切换主题", self._cycle_theme)

        help_menu = menu_bar.addMenu("帮助")
        help_menu.addAction("新手教程", self._show_tutorial)
        help_menu.addAction("当前阶段说明", self._show_stage_summary)
        return menu_bar

    def _build_toolbar(self) -> QWidget:
        toolbar = QToolBar("主工具栏")
        toolbar.setObjectName("topToolBar")
        toolbar.setFixedHeight(44)
        toolbar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        toolbar.addAction("新建项目", self._create_project)
        toolbar.addAction("打开", self._open_project)
        toolbar.addAction("保存", self._save_current_state)
        toolbar.addSeparator()
        toolbar.addAction("运行", self._run_block_mesh)
        toolbar.addAction("停止", self._stop_current_process)
        toolbar.addSeparator()
        toolbar.addAction("设置", self._open_settings_tab)
        toolbar.addAction("切换主题", self._cycle_theme)
        return toolbar

    def _build_status_bar(self) -> None:
        status_bar = QStatusBar()
        self._case_label = QLabel("当前 Case: 未选择")
        self._task_label = QLabel("任务状态: 空闲")
        self._version_label = QLabel("OpenFOAM: 检测中")
        status_bar.addWidget(self._case_label)
        status_bar.addPermanentWidget(self._task_label)
        status_bar.addPermanentWidget(self._version_label)
        self.setStatusBar(status_bar)

    def _build_activity_bar(self) -> QWidget:
        container = QFrame()
        container.setObjectName("activityBar")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        activity_list = QListWidget()
        activity_list.setObjectName("activityList")
        activity_list.setFixedWidth(70)
        activity_list.setSpacing(4)
        for item_text in ("资源", "搜索", "任务", "结果"):
            activity_list.addItem(QListWidgetItem(item_text))
        activity_list.setCurrentRow(0)
        activity_list.currentRowChanged.connect(self._on_activity_changed)
        layout.addWidget(activity_list)
        return container

    def _build_sidebar(self) -> QWidget:
        container = QFrame()
        sidebar_layout = QVBoxLayout(container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        title = QLabel("资源管理器")
        title.setContentsMargins(12, 10, 12, 10)
        sidebar_layout.addWidget(title)
        sidebar_layout.addWidget(self._build_project_tree())
        return container

    def _build_project_tree(self) -> QWidget:
        self._project_tree = QTreeWidget()
        self._project_tree.setHeaderLabel("项目树")
        self._project_tree.itemClicked.connect(self._on_project_tree_item_clicked)
        self._refresh_project_tree()
        return self._project_tree

    def _build_editor_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        vertical_splitter.addWidget(self._build_workspace())
        vertical_splitter.addWidget(self._build_bottom_panel())
        vertical_splitter.setSizes([640, 220])
        vertical_splitter.setChildrenCollapsible(False)
        layout.addWidget(vertical_splitter)
        return container

    def _build_workspace(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._workspace_tabs = QTabWidget()
        self._workspace_tabs.setDocumentMode(True)
        self._workspace_tabs.setTabsClosable(False)
        self._workspace_tabs.addTab(self._build_project_home_tab(), "项目主页")
        self._workspace_tabs.addTab(self._make_text_panel("参数配置区"), "参数配置")
        self._workspace_tabs.addTab(self._make_text_panel("求解运行区"), "求解运行")
        self._workspace_tabs.addTab(self._build_environment_tab(), "环境检查")
        self._workspace_tabs.addTab(self._build_settings_tab(), "设置")
        self._workspace_tabs.addTab(self._make_text_panel("结果区"), "结果")
        layout.addWidget(self._workspace_tabs)
        return container

    def _build_bottom_panel(self) -> QWidget:
        self._bottom_tabs = QTabWidget()
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._task_text = QTextEdit()
        self._task_text.setReadOnly(True)
        self._problem_text = QTextEdit()
        self._problem_text.setReadOnly(True)
        self._bottom_tabs.addTab(self._log_text, "日志")
        self._bottom_tabs.addTab(self._task_text, "任务")
        self._bottom_tabs.addTab(self._problem_text, "问题")
        self._append_log("应用已启动。")
        self._task_text.setPlainText("任务状态：空闲")
        self._problem_text.setPlainText("暂无问题。")
        return self._bottom_tabs

    def _make_text_panel(self, title: str) -> QWidget:
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        editor = QTextEdit()
        editor.setReadOnly(True)
        editor.setPlainText(
            f"{title}\n\n"
            "VS Code 风格工作台布局：\n"
            "- 左侧活动栏\n"
            "- 左侧边栏\n"
            "- 中央标签工作区\n"
            "- 底部面板\n"
            "- 底部状态栏\n\n"
            "后续会继续接入参数表单、任务日志和结果视图。"
        )
        layout.addWidget(editor)
        return wrapper

    def _build_project_home_tab(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("项目主页")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        summary = QTextEdit()
        summary.setReadOnly(True)
        summary.setPlainText(
            "当前阶段：设置页 + 环境检查页 + 项目管理入口占位\n\n"
            "已具备：\n"
            "- VS Code 风格工作台\n"
            "- 基础主题切换\n"
            "- OpenFOAM 环境探测骨架\n"
            "- 项目树和结果/日志面板占位\n\n"
            "下一步将接入：\n"
            "- 项目新建流程\n"
            "- blockMesh 命令执行\n"
            "- 实时日志和任务状态"
        )
        layout.addWidget(title)
        layout.addWidget(summary)
        return wrapper

    def _build_environment_tab(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("环境检查")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        self._environment_text = QTextEdit()
        self._environment_text.setReadOnly(True)
        refresh_button = QPushButton("重新检测 OpenFOAM 环境")
        refresh_button.clicked.connect(self._refresh_environment_panels)

        layout.addWidget(title)
        layout.addWidget(refresh_button)
        layout.addWidget(self._environment_text)
        return wrapper

    def _build_settings_tab(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("设置")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        form = QFormLayout()

        settings = self._context.settings_service.load()
        self._theme_combo = QComboBox()
        self._theme_combo.addItems(self._theme_names)
        self._theme_combo.setCurrentText(settings.theme_name)

        self._background_color_input = QLineEdit(settings.background_color)
        self._workspace_input = QLineEdit(str(settings.workspace_dir))
        self._env_script_input = QLineEdit(settings.openfoam_env_script or "")
        self._font_combo = QFontComboBox()
        self._font_combo.setCurrentFont(QFont(settings.font_family))
        self._font_size_input = QSpinBox()
        self._font_size_input.setRange(11, 28)
        self._font_size_input.setValue(settings.font_size)
        self._show_tutorial_checkbox = QCheckBox("每次启动时显示新手教程")
        self._show_tutorial_checkbox.setChecked(settings.show_tutorial_on_startup)

        form.addRow("主题", self._theme_combo)
        form.addRow("主背景色", self._background_color_input)
        form.addRow("界面字体", self._font_combo)
        form.addRow("字体大小", self._font_size_input)
        form.addRow("新手教程", self._show_tutorial_checkbox)
        form.addRow("工作区路径", self._workspace_input)
        form.addRow("OpenFOAM 环境脚本", self._env_script_input)

        save_button = QPushButton("保存设置并应用")
        save_button.clicked.connect(self._save_settings)
        hint = QLabel("示例背景色：#1e1e1e、#151b23、#202020")

        layout.addWidget(title)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(save_button)
        layout.addStretch(1)
        return wrapper

    def _open_settings_tab(self) -> None:
        self._workspace_tabs.setCurrentIndex(4)
        self._set_status("已打开设置页。")

    def _open_environment_tab(self) -> None:
        self._workspace_tabs.setCurrentIndex(3)
        self._refresh_environment_panels()
        self._set_status("已打开环境检查页。")

    def _apply_settings_theme(self) -> None:
        settings = self._context.settings_service.load()
        self._theme_index = self._theme_names.index(settings.theme_name)
        self.setStyleSheet(
            build_stylesheet(
                settings.theme_name,
                settings.background_color,
                settings.font_family,
                settings.font_size,
            )
        )
        if hasattr(self, "_theme_combo"):
            self._theme_combo.setCurrentText(settings.theme_name)
            self._background_color_input.setText(settings.background_color)
            self._workspace_input.setText(str(settings.workspace_dir))
            self._env_script_input.setText(settings.openfoam_env_script or "")
            self._font_combo.setCurrentFont(QFont(settings.font_family))
            self._font_size_input.setValue(settings.font_size)
            self._show_tutorial_checkbox.setChecked(settings.show_tutorial_on_startup)

    def _cycle_theme(self) -> None:
        self._theme_index = (self._theme_index + 1) % len(self._theme_names)
        theme_name = self._theme_names[self._theme_index]
        palette = THEMES[theme_name]
        settings = self._context.settings_service.load()
        updated_settings = type(settings)(
            workspace_dir=settings.workspace_dir,
            openfoam_env_script=settings.openfoam_env_script,
            theme_name=theme_name,
            background_color=palette.window_bg,
            font_family=settings.font_family,
            font_size=settings.font_size,
            show_tutorial_on_startup=settings.show_tutorial_on_startup,
        )
        self._context.settings_service.save(updated_settings)
        self._apply_settings_theme()
        self._refresh_environment_panels()
        self._set_status(f"主题已切换为 {theme_name}。")

    def _save_settings(self) -> None:
        background_color = self._background_color_input.text().strip() or "#1e1e1e"
        env_script = self._env_script_input.text().strip() or None
        settings = self._context.settings_service.load()
        updated_settings = type(settings)(
            workspace_dir=settings.workspace_dir.__class__(self._workspace_input.text().strip()),
            openfoam_env_script=env_script,
            theme_name=self._theme_combo.currentText(),
            background_color=background_color,
            font_family=self._font_combo.currentFont().family(),
            font_size=self._font_size_input.value(),
            show_tutorial_on_startup=self._show_tutorial_checkbox.isChecked(),
        )
        self._context.settings_service.save(updated_settings)
        self._apply_settings_theme()
        self._refresh_environment_panels()
        self._set_status("设置已保存。")

    def _refresh_status_bar(self) -> None:
        status = self._context.environment_detector.detect()
        version_text = status.foam_version if status.is_available else "未就绪"
        self._version_label.setText(f"OpenFOAM: {version_text}")
        self._refresh_environment_panels(status)

    def _refresh_environment_panels(self, status=None) -> None:
        if status is None:
            status = self._context.environment_detector.detect()
        status_flag = "可用" if status.is_available else "不可用"
        self._environment_text.setPlainText(
            "OpenFOAM 环境检查结果\n\n"
            f"- 状态：{status_flag}\n"
            f"- bash 路径：{status.bash_path or '未找到'}\n"
            f"- 环境脚本：{status.env_script_path or '未配置'}\n"
            f"- OpenFOAM 版本：{status.foam_version or '未知'}\n"
            f"- 说明：{status.detail}\n"
        )
        self._append_log(f"环境检查完成：{status_flag}，OpenFOAM={status.foam_version or '未知'}")

    def _on_activity_changed(self, index: int) -> None:
        if index == 0:
            self._workspace_tabs.setCurrentIndex(0)
            self._set_status("已切换到资源视图。")
        elif index == 1:
            self._search_projects()
        elif index == 2:
            self._workspace_tabs.setCurrentIndex(2)
            self._bottom_tabs.setCurrentIndex(1)
            self._set_status("已切换到任务视图。")
        elif index == 3:
            self._workspace_tabs.setCurrentIndex(5)
            self._set_status("已切换到结果视图。")

    def _save_current_state(self) -> None:
        self._save_settings()
        self._append_log("保存：当前设置已写入本地配置。")

    def _create_project(self) -> None:
        name, ok = QInputDialog.getText(self, "新建项目", "项目名称")
        if not ok:
            return
        try:
            project = self._context.project_service.create_project(name)
        except ValueError as error:
            self._show_error(str(error))
            return

        self._current_project = project
        self._refresh_project_tree()
        self._workspace_tabs.setCurrentIndex(0)
        self._case_label.setText(f"当前 Case: {project.name}")
        self._append_log(f"已创建项目：{project.path}")
        self._set_status("项目创建完成。")

    def _open_project(self) -> None:
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

        self._current_project = project
        self._refresh_project_tree()
        self._workspace_tabs.setCurrentIndex(0)
        self._case_label.setText(f"当前 Case: {project.name}")
        self._append_log(f"已打开项目：{project.path}")
        self._set_status("项目打开完成。")

    def _refresh_project_tree(self) -> None:
        if not hasattr(self, "_project_tree"):
            return
        self._project_tree.clear()
        projects = self._context.project_service.list_projects()
        if not projects:
            empty_item = QTreeWidgetItem(["暂无项目"])
            empty_item.setDisabled(True)
            self._project_tree.addTopLevelItem(empty_item)
            return

        for project in projects:
            project_item = QTreeWidgetItem([project.name])
            project_item.setData(0, Qt.ItemDataRole.UserRole, str(project.path))
            case_item = QTreeWidgetItem(["case"])
            case_item.setData(0, Qt.ItemDataRole.UserRole, str(project.path))
            project_item.addChild(case_item)
            self._project_tree.addTopLevelItem(project_item)
        self._project_tree.expandAll()

    def _on_project_tree_item_clicked(self, item: QTreeWidgetItem) -> None:
        project_path = item.data(0, Qt.ItemDataRole.UserRole)
        if not project_path:
            return
        try:
            self._current_project = self._context.project_service.open_project(Path(project_path))
        except ValueError as error:
            self._show_error(str(error))
            return
        self._case_label.setText(f"当前 Case: {self._current_project.name}")
        self._append_log(f"当前项目：{self._current_project.path}")

    def _search_projects(self) -> None:
        keyword, ok = QInputDialog.getText(self, "搜索项目", "项目名称关键字")
        if not ok:
            return
        normalized = keyword.strip().lower()
        root_count = self._project_tree.topLevelItemCount()
        for index in range(root_count):
            item = self._project_tree.topLevelItem(index)
            visible = not normalized or normalized in item.text(0).lower()
            item.setHidden(not visible)
        self._set_status("项目搜索已应用。")

    def _run_block_mesh(self) -> None:
        if self._foam_process and self._foam_process.state() != QProcess.ProcessState.NotRunning:
            self._show_error("已有任务正在运行，请先停止当前任务。")
            return
        if self._current_project is None:
            self._show_error("请先新建或打开一个项目。")
            return

        status = self._context.environment_detector.detect()
        if not status.is_available or not status.env_script_path:
            self._show_error(f"OpenFOAM 环境不可用：{status.detail}")
            return

        block_mesh_dict = self._current_project.case_dir / "system" / "blockMeshDict"
        if not block_mesh_dict.exists():
            self._show_error("当前 case 缺少 system/blockMeshDict。")
            return

        self._workspace_tabs.setCurrentIndex(2)
        self._bottom_tabs.setCurrentIndex(0)
        self._task_text.setPlainText("任务状态：blockMesh 运行中")
        self._set_status("blockMesh 运行中。")

        command = (
            f"source {shlex.quote(status.env_script_path)} >/dev/null 2>&1 && "
            f"cd {shlex.quote(str(self._current_project.case_dir))} && blockMesh"
        )
        self._foam_process = QProcess(self)
        self._foam_process.setProgram("bash")
        self._foam_process.setArguments(["-lc", command])
        self._foam_process.readyReadStandardOutput.connect(self._read_process_stdout)
        self._foam_process.readyReadStandardError.connect(self._read_process_stderr)
        self._foam_process.finished.connect(self._on_process_finished)
        self._foam_process.start()
        self._append_log(f"启动 blockMesh：{self._current_project.case_dir}")

    def _stop_current_process(self) -> None:
        if not self._foam_process or self._foam_process.state() == QProcess.ProcessState.NotRunning:
            self._task_text.setPlainText("任务状态：空闲")
            self._set_status("当前没有正在运行的任务。")
            return
        self._foam_process.terminate()
        if not self._foam_process.waitForFinished(3000):
            self._foam_process.kill()
        self._task_text.setPlainText("任务状态：已停止")
        self._set_status("任务已停止。")

    def _read_process_stdout(self) -> None:
        if self._foam_process:
            self._append_log(bytes(self._foam_process.readAllStandardOutput()).decode(errors="replace"))

    def _read_process_stderr(self) -> None:
        if self._foam_process:
            self._append_log(bytes(self._foam_process.readAllStandardError()).decode(errors="replace"))

    def _on_process_finished(self, exit_code: int, _exit_status) -> None:
        if exit_code == 0:
            self._task_text.setPlainText("任务状态：blockMesh 完成")
            self._set_status("blockMesh 完成。")
        else:
            self._task_text.setPlainText(f"任务状态：blockMesh 失败，退出码 {exit_code}")
            self._set_status(f"blockMesh 失败，退出码 {exit_code}。")

    def _show_current_case_path(self) -> None:
        if self._current_project is None:
            self._show_error("当前没有打开的项目。")
            return
        self._workspace_tabs.setCurrentIndex(0)
        self._append_log(f"当前 Case 目录：{self._current_project.case_dir}")
        self._set_status("已输出当前 Case 目录。")

    def _show_stage_summary(self) -> None:
        self._append_log(
            "当前 Sprint：Sprint 2 UI 可用性与设置系统；下一阶段：Sprint 3 项目管理与 OpenFOAM 最小执行闭环。"
        )
        self._set_status("已输出当前阶段说明。")

    def _show_tutorial(self) -> None:
        QMessageBox.information(
            self,
            "FoamDesk 新手教程",
            "FoamDesk 当前可用流程：\n\n"
            "1. 点击“新建项目”，输入项目名称。\n"
            "2. 左侧项目树会显示真实项目，点击项目即可设为当前 Case。\n"
            "3. 打开“设置”，确认工作区路径、界面字体、字体大小和 OpenFOAM 环境脚本。\n"
            "4. 打开“环境检查”，确认 OpenFOAM 环境是否可用。\n"
            "5. 点击“运行”，当前阶段会尝试执行 blockMesh。\n"
            "6. 底部“日志 / 任务 / 问题”面板会显示运行信息。\n\n"
            "当前 Sprint：Sprint 2 UI 可用性与设置系统。\n"
            "下一阶段：Sprint 3 项目管理与 OpenFOAM 最小执行闭环。",
        )
        self._append_log("已显示新手教程。")

    def _show_error(self, message: str) -> None:
        self._problem_text.setPlainText(message)
        self._bottom_tabs.setCurrentIndex(2)
        self._append_log(f"错误：{message}")
        QMessageBox.warning(self, "FoamDesk", message)

    def _append_log(self, message: str) -> None:
        if hasattr(self, "_log_text"):
            self._log_text.append(message)

    def _set_status(self, message: str) -> None:
        self._task_label.setText(f"任务状态: {message}")
        self._append_log(message)
