from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFontComboBox,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QPushButton,
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
        self.setWindowTitle("FoamDesk")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.resize(1400, 900)
        self._build_ui()
        self._apply_settings_theme()
        self._refresh_status_bar()

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
        shell_layout.addWidget(workbench)
        self.setCentralWidget(shell)

    def _build_menu_bar(self) -> QWidget:
        menu_bar = self.menuBar()
        menu_bar.setObjectName("topMenuBar")
        for label in ("文件", "项目", "Case", "求解器", "工具", "帮助"):
            menu_bar.addMenu(QMenu(label, self))
        return menu_bar

    def _build_toolbar(self) -> QWidget:
        toolbar = QToolBar("主工具栏")
        toolbar.setObjectName("topToolBar")
        toolbar.addAction("新建项目", self._new_project_placeholder)
        toolbar.addAction("打开", self._open_project_placeholder)
        toolbar.addAction("保存", self._save_current_state)
        toolbar.addSeparator()
        toolbar.addAction("运行", self._run_placeholder)
        toolbar.addAction("停止", self._stop_placeholder)
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
        tree = QTreeWidget()
        tree.setHeaderLabel("项目树")
        project = QTreeWidgetItem(["示例项目"])
        project.addChild(QTreeWidgetItem(["Case1"]))
        project.addChild(QTreeWidgetItem(["Case2"]))
        tree.addTopLevelItem(project)
        tree.expandAll()
        return tree

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

        form.addRow("主题", self._theme_combo)
        form.addRow("主背景色", self._background_color_input)
        form.addRow("界面字体", self._font_combo)
        form.addRow("字体大小", self._font_size_input)
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
            self._workspace_tabs.setCurrentIndex(0)
            self._set_status("搜索功能将在项目管理阶段接入。")
        elif index == 2:
            self._workspace_tabs.setCurrentIndex(2)
            self._bottom_tabs.setCurrentIndex(1)
            self._set_status("已切换到任务视图。")
        elif index == 3:
            self._workspace_tabs.setCurrentIndex(5)
            self._set_status("已切换到结果视图。")

    def _new_project_placeholder(self) -> None:
        self._workspace_tabs.setCurrentIndex(0)
        self._append_log("新建项目：功能入口已响应，下一阶段接入真实项目创建流程。")
        self._set_status("新建项目入口已响应。")

    def _open_project_placeholder(self) -> None:
        self._workspace_tabs.setCurrentIndex(0)
        self._append_log("打开项目：功能入口已响应，下一阶段接入本地项目选择。")
        self._set_status("打开项目入口已响应。")

    def _save_current_state(self) -> None:
        self._save_settings()
        self._append_log("保存：当前设置已写入本地配置。")

    def _run_placeholder(self) -> None:
        self._workspace_tabs.setCurrentIndex(2)
        self._bottom_tabs.setCurrentIndex(1)
        self._task_text.setPlainText("任务状态：等待接入 blockMesh 执行链路")
        self._append_log("运行：按钮已响应，下一阶段接入 blockMesh 和实时日志。")
        self._set_status("运行入口已响应。")

    def _stop_placeholder(self) -> None:
        self._task_text.setPlainText("任务状态：空闲")
        self._append_log("停止：当前没有正在运行的任务。")
        self._set_status("当前没有正在运行的任务。")

    def _append_log(self, message: str) -> None:
        if hasattr(self, "_log_text"):
            self._log_text.append(message)

    def _set_status(self, message: str) -> None:
        self._task_label.setText(f"任务状态: {message}")
        self._append_log(message)
