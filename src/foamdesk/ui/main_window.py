from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtCore import QProcess
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFontComboBox,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from foamdesk.app.bootstrap import ApplicationContext
from foamdesk.domain.models import SimulationParameters, SimulationProject
from foamdesk.ui.theme import THEMES
from foamdesk.ui.visualization_widgets import NativeVtkPreviewWidget, NativeVtkViewerDialog, VtkViewerDialog
from foamdesk.ui.main_window_geometry_logic import GeometryLogicMixin
from foamdesk.ui.main_window_results_logic import ResultsLogicMixin
from foamdesk.ui.main_window_parameters_logic import ParametersLogicMixin
from foamdesk.ui.main_window_project_logic import ProjectProcessLogicMixin
from foamdesk.ui.main_window_settings_physics_logic import SettingsPhysicsLogicMixin
from foamdesk.ui.draw_geometry_tab import DrawGeometryLogicMixin


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


class MainWindow(GeometryLogicMixin, ResultsLogicMixin, ParametersLogicMixin, SettingsPhysicsLogicMixin, ProjectProcessLogicMixin, DrawGeometryLogicMixin, QMainWindow):
    TAB_PROJECT_HOME = 0
    TAB_DRAW_GEOMETRY = 1
    TAB_MESH_GENERATION = 2
    TAB_SOLVER_PREPARE = 3
    TAB_SOLVER_SELECT = 4
    TAB_PARAMETERS = 5
    TAB_SOLVER_RUN = 6
    TAB_ENVIRONMENT = 7
    TAB_SETTINGS = 8
    TAB_RESULTS = 9
    RESULT_FIELDS = [
        "U",
        "mag(U)",
        "p",
        "T",
        "k",
    ]
    RESULT_FIELD_UNITS = {
        "U": "m/s",
        "mag(U)": "m/s",
        "p": "m2/s2 或 Pa",
        "T": "K",
        "k": "m2/s2",
    }
    RESULT_DISPLAY_MODES = [
        "Surface 表面云图",
        "Slice 切片",
        "Contour 等值线",
        "Iso-surface 等值面",
        "Streamline 流线",
    ]
    RESULT_FIELD_DISPLAY_MODES = {
        "U": [
            "Surface 表面云图",
            "Slice 切片",
            "Streamline 流线",
        ],
        "mag(U)": [
            "Surface 表面云图",
            "Slice 切片",
            "Contour 等值线",
            "Iso-surface 等值面",
        ],
        "p": [
            "Surface 表面云图",
            "Slice 切片",
            "Contour 等值线",
            "Iso-surface 等值面",
        ],
        "T": [
            "Surface 表面云图",
            "Slice 切片",
            "Contour 等值线",
            "Iso-surface 等值面",
        ],
        "k": [
            "Surface 表面云图",
            "Slice 切片",
            "Contour 等值线",
            "Iso-surface 等值面",
        ],
    }

    def __init__(self, context: ApplicationContext, initial_project: SimulationProject | None = None) -> None:
        super().__init__()
        self._context = context
        self._theme_names = list(THEMES.keys())
        self._theme_index = 0
        self._current_project: SimulationProject | None = None
        self._foam_process: QProcess | None = None
        self._active_process_kind = "idle"
        self._current_process_output = ""
        self._last_diagnostic_summary = "暂无诊断。"
        self._suspend_draw_geometry_persist = False
        self._vtk_viewer: VtkViewerDialog | None = None
        self._native_vtk_viewer: NativeVtkViewerDialog | None = None
        self.setWindowTitle("FoamDesk")
        self.resize(1400, 900)
        self._init_modeling_state()
        self._build_ui()
        self._setup_modeling_axes()
        self._apply_settings_theme()
        self._refresh_status_bar()
        if initial_project is not None:
            self._activate_project(initial_project, "已恢复上次项目。")

    def closeEvent(self, event) -> None:  # noqa: N802
        super().closeEvent(event)

    def _build_ui(self) -> None:
        self._build_status_bar()

        workbench = QSplitter(Qt.Orientation.Horizontal)
        workbench.addWidget(self._build_sidebar())
        workbench.addWidget(self._build_editor_panel())
        workbench.setSizes([300, 1100])
        workbench.setChildrenCollapsible(False)

        shell = QWidget()
        shell.setObjectName("appShell")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        shell_layout.addWidget(self._build_menu_bar())
        shell_layout.addWidget(workbench, 1)
        self.setCentralWidget(shell)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)

    def _build_menu_bar(self) -> QWidget:
        menu_bar = QMenuBar(self)
        menu_bar.setObjectName("topMenuBar")
        menu_bar.setFixedHeight(34)
        menu_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        file_menu = menu_bar.addMenu("文件")
        file_menu.addAction("新建项目", self._create_project)
        file_menu.addAction("打开项目", self._open_project)
        file_menu.addAction("返回项目选择", self._return_to_project_selection)
        file_menu.addAction("保存设置", self._save_current_state)
        file_menu.addSeparator()
        file_menu.addAction("退出", self.close)

        project_menu = menu_bar.addMenu("项目")
        project_menu.addAction("返回项目选择", self._return_to_project_selection)
        project_menu.addAction("刷新 Case 树", self._refresh_project_tree)
        project_menu.addAction("搜索项目", self._search_projects)

        case_menu = menu_bar.addMenu("Case")
        case_menu.addAction("新增 Case", self._create_case)
        case_menu.addAction("删除 Case", self._delete_case)
        case_menu.addAction("打开当前 Case 目录", self._show_current_case_path)

        geometry_menu = menu_bar.addMenu("几何/CAD")
        geometry_menu.addAction("导入 STL", self._import_stl_geometry)
        geometry_menu.addAction("打开网格生成", self._open_mesh_generation_tab)
        geometry_menu.addSeparator()
        geometry_menu.addAction("生成 snappyHexMeshDict", self._generate_snappy_hex_mesh_dict)
        geometry_menu.addAction("运行 snappyHexMesh", self._run_snappy_hex_mesh)
        geometry_menu.addAction("运行 checkMesh", self._run_check_mesh)
        geometry_menu.addAction("一键前处理", self._run_preprocess_pipeline)

        solver_menu = menu_bar.addMenu("求解器")
        solver_menu.addAction("运行最小仿真", self._run_minimal_simulation)
        solver_menu.addAction("停止当前任务", self._stop_current_process)

        tools_menu = menu_bar.addMenu("工具")
        tools_menu.addAction("环境检查", self._open_environment_tab)
        tools_menu.addAction("设置", self._open_settings_tab)

        theme_menu = menu_bar.addMenu("主题")
        for theme_name in self._theme_names:
            theme_menu.addAction(theme_name, lambda _checked=False, name=theme_name: self._set_theme(name))
        theme_menu.addSeparator()
        theme_menu.addAction("循环切换主题", self._cycle_theme)

        help_menu = menu_bar.addMenu("帮助")
        help_menu.addAction("当前阶段说明", self._show_stage_summary)
        return menu_bar

    def _build_status_bar(self) -> None:
        status_bar = QStatusBar()
        self._case_label = QLabel("当前 Case: 未选择")
        self._task_label = QLabel("任务状态: 空闲")
        self._version_label = QLabel("OpenFOAM: 检测中")
        status_bar.addWidget(self._case_label)
        status_bar.addPermanentWidget(self._task_label)
        status_bar.addPermanentWidget(self._version_label)
        self.setStatusBar(status_bar)

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
        self._project_tree.setHeaderLabel("Case 树")
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
        self._workspace_tabs.addTab(self._build_draw_geometry_tab(), "绘制几何")
        self._workspace_tabs.addTab(self._build_mesh_generation_tab(), "网格生成")
        self._workspace_tabs.addTab(self._build_physics_prepare_tab(), "求解器准备")
        self._workspace_tabs.addTab(self._build_solver_select_tab(), "求解器选择")
        self._workspace_tabs.addTab(self._build_parameter_tab(), "仿真参数")
        self._workspace_tabs.addTab(self._build_solver_run_tab(), "求解运行")
        self._workspace_tabs.addTab(self._build_environment_tab(), "环境检查")
        self._workspace_tabs.addTab(self._build_settings_tab(), "设置")
        self._workspace_tabs.addTab(self._build_results_tab(), "结果")
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

    def _build_project_home_tab(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("项目主页")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        self._project_home_summary = QTextEdit()
        self._project_home_summary.setReadOnly(True)
        self._project_home_summary.setPlainText("请先选择或创建项目。")
        layout.addWidget(title)
        layout.addWidget(self._project_home_summary)
        return wrapper

    def _build_mesh_generation_tab(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("可视化网格生成")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        description = QLabel(
            "本页把 OpenFOAM 网格流程集中起来：先检查 blockMesh 背景网格和 STL，"
            "再生成 snappyHexMeshDict，最后执行 blockMesh、snappyHexMesh 和 checkMesh。"
        )
        description.setWordWrap(True)

        flow = QLabel("流程：绘制几何/导入 STL -> 生成 snappyHexMeshDict -> blockMesh -> snappyHexMesh -overwrite -> checkMesh")
        flow.setWordWrap(True)

        button_row = QHBoxLayout()
        actions = [
            ("刷新网格状态", self._refresh_mesh_generation_panel),
            ("打开绘制几何", self._open_draw_geometry_tab),
            ("导入 STL", self._import_stl_geometry),
            ("预览计算域/STL", self._open_domain_preview_dialog),
            ("生成 snappyHexMeshDict", self._generate_snappy_hex_mesh_dict),
            ("运行 snappyHexMesh", self._run_snappy_hex_mesh),
            ("运行 checkMesh", self._run_check_mesh),
            ("一键生成网格", self._run_preprocess_pipeline),
        ]
        for text, handler in actions:
            button = QPushButton(text)
            button.clicked.connect(lambda _checked=False, callback=handler: callback())
            button_row.addWidget(button)
        button_row.addStretch(1)

        self._mesh_generation_status = QLabel("网格状态：未刷新")
        self._mesh_generation_text = QTextEdit()
        self._mesh_generation_text.setReadOnly(True)
        self._mesh_generation_text.setMinimumHeight(360)
        self._mesh_generation_text.setPlainText("请选择项目和 Case 后点击“刷新网格状态”。")

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(flow)
        layout.addLayout(button_row)
        layout.addWidget(self._mesh_generation_status)
        layout.addWidget(self._mesh_generation_text, 1)
        return wrapper


    def _build_solver_select_tab(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("求解器选择")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        description = QLabel("选择当前 Case 要使用的 OpenFOAM 求解器。本阶段先支持 icoFoam、simpleFoam、pisoFoam。")
        description.setWordWrap(True)

        form = QFormLayout()
        self._solver_name_combo = QComboBox()
        self._solver_name_combo.addItem("icoFoam - 入门不可压瞬态流", "icoFoam")
        self._solver_name_combo.addItem("simpleFoam - 稳态不可压流/风洞绕流", "simpleFoam")
        self._solver_name_combo.addItem("pisoFoam - 瞬态不可压流", "pisoFoam")
        self._turbulence_model_combo = QComboBox()
        self._turbulence_model_combo.addItem("laminar - 层流/入门默认", "laminar")
        self._turbulence_model_combo.addItem("RAS kEpsilon - 湍流模型占位", "RAS kEpsilon")
        self._numeric_scheme_combo = QComboBox()
        self._numeric_scheme_combo.addItem("stable - 稳定优先 upwind", "stable")
        self._numeric_scheme_combo.addItem("balanced - 平衡 linearUpwind", "balanced")
        self._numeric_scheme_combo.addItem("accurate - 精度优先 linear", "accurate")
        self._fv_solution_preset_combo = QComboBox()
        self._fv_solution_preset_combo.addItem("default - 默认收敛设置", "default")
        self._fv_solution_preset_combo.addItem("strict - 更严格残差", "strict")
        self._fv_solution_preset_combo.addItem("fast - 更快但较粗", "fast")
        form.addRow("求解器", self._solver_name_combo)
        form.addRow("湍流模型", self._turbulence_model_combo)
        form.addRow("数值格式 fvSchemes", self._numeric_scheme_combo)
        form.addRow("求解设置 fvSolution", self._fv_solution_preset_combo)

        button_row = QHBoxLayout()
        load_button = QPushButton("加载当前 Case 求解器")
        save_button = QPushButton("保存求解器配置")
        load_button.clicked.connect(lambda _checked=False: self._load_case_parameters())
        save_button.clicked.connect(lambda _checked=False: self._save_case_parameters())
        button_row.addWidget(load_button)
        button_row.addWidget(save_button)
        button_row.addStretch(1)

        self._solver_select_status_label = QLabel("请先新建或打开项目。")
        self._solver_select_status_label.setWordWrap(True)
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setMaximumHeight(190)
        help_text.setPlainText(
            "求解器怎么选：\n"
            "- icoFoam：先跑通最小不可压瞬态流，适合学习和验证流程。\n"
            "- simpleFoam：稳态不可压流，后续做车/圆柱/风洞绕流更常用。\n"
            "- pisoFoam：瞬态不可压流，适合观察流动随时间变化。\n\n"
            "注意：当前页面先负责保存配置和生成核心字典，真正切换不同求解器的完整运行流水线后续继续增强。"
        )

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addLayout(form)
        layout.addLayout(button_row)
        layout.addWidget(self._solver_select_status_label)
        layout.addWidget(help_text)
        layout.addStretch(1)
        self._set_solver_inputs_enabled(False)
        return wrapper

    def _build_parameter_tab(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("仿真参数")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        description = QLabel("只配置当前 Case 的 system/controlDict 时间控制参数。")
        description.setWordWrap(True)

        form = QFormLayout()
        self._end_time_input = QLineEdit()
        self._delta_t_input = QLineEdit()
        self._write_interval_input = QSpinBox()
        self._write_interval_input.setRange(1, 1000000)
        form.addRow("仿真总时间 endTime", self._end_time_input)
        form.addRow("时间步长 deltaT", self._delta_t_input)
        form.addRow("写出间隔 writeInterval", self._write_interval_input)

        button_row = QHBoxLayout()
        load_button = QPushButton("加载当前项目参数")
        save_button = QPushButton("保存参数到 Case")
        default_button = QPushButton("恢复默认参数")
        load_button.clicked.connect(lambda _checked=False: self._load_case_parameters())
        save_button.clicked.connect(lambda _checked=False: self._save_control_dict_parameters())
        default_button.clicked.connect(lambda _checked=False: self._restore_default_parameters())
        button_row.addWidget(load_button)
        button_row.addWidget(save_button)
        button_row.addWidget(default_button)
        button_row.addStretch(1)

        self._parameter_status_label = QLabel("请先新建或打开项目。")
        self._parameter_status_label.setWordWrap(True)
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setMaximumHeight(150)
        help_text.setPlainText(
            "参数说明：\n"
            "- endTime：仿真总时间，越大运行越久。\n"
            "- deltaT：每一步的时间步长，越小越稳定但更慢。\n"
            "- writeInterval：每隔多少步写一次结果。\n\n"
            "注意：0/U、0/p、patch 边界和流体物性请到“求解器准备”页面配置。"
        )

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addLayout(form)
        layout.addLayout(button_row)
        layout.addWidget(self._parameter_status_label)
        layout.addWidget(help_text)
        layout.addStretch(1)
        self._set_parameter_inputs_enabled(False)
        return wrapper

    def _make_menu_button(self, title, actions):
        button = QPushButton(title)
        menu = QMenu(button)
        for label, callback in actions:
            action = menu.addAction(label)
            action.triggered.connect(lambda _checked=False, cb=callback: cb())
        button.setMenu(menu)
        return button




    def _build_draw_geometry_tab(self) -> QWidget:
        wrapper = QWidget()
        root = QVBoxLayout(wrapper)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- toolbar ---
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(8, 4, 8, 4)
        toolbar.setSpacing(4)

        primitives = [
            ("cube", "立方体"), ("sphere", "球体"),
            ("cylinder", "圆柱"), ("cone", "圆锥"),
        ]
        for kind, label in primitives:
            btn = QPushButton(label)
            btn.setFixedHeight(30)
            btn.clicked.connect(lambda _checked=False, k=kind: self._add_primitive(k))
            toolbar.addWidget(btn)

        toolbar.addSpacing(12)

        for text, handler, extra in [
            ("选择", self._activate_select_mode, ""),
            ("删除", self._delete_selected, "color: #f48771;"),
        ]:
            btn = QPushButton(text)
            btn.setFixedHeight(30)
            if extra:
                btn.setStyleSheet(extra)
            btn.clicked.connect(handler)
            toolbar.addWidget(btn)

        toolbar.addSpacing(12)

        reset_btn = QPushButton("重置视角")
        reset_btn.setFixedHeight(30)
        reset_btn.clicked.connect(self._reset_camera)
        toolbar.addWidget(reset_btn)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        # --- body: scene tree | viewport | property panel ---
        body = QSplitter(Qt.Orientation.Horizontal)

        # scene tree
        tree_wrapper = QWidget()
        tree_layout = QVBoxLayout(tree_wrapper)
        tree_layout.setContentsMargins(8, 8, 8, 8)
        tree_layout.setSpacing(6)
        tree_title = QLabel("模型树")
        tree_title.setStyleSheet("font-size: 15px; font-weight: 600;")
        tree_layout.addWidget(tree_title)
        self._modeling_tree = QTreeWidget()
        self._modeling_tree.setHeaderLabels(["模型", "类型"])
        self._modeling_tree.setColumnWidth(0, 100)
        self._modeling_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._modeling_tree.customContextMenuRequested.connect(self._tree_context_menu)
        self._modeling_tree.itemClicked.connect(self._on_tree_item_clicked)
        self._modeling_tree.itemChanged.connect(self._on_tree_item_changed)
        tree_layout.addWidget(self._modeling_tree, 1)
        body.addWidget(tree_wrapper)

        # viewport
        viewport_wrapper = QWidget()
        vp_layout = QVBoxLayout(viewport_wrapper)
        vp_layout.setContentsMargins(0, 0, 0, 0)
        self._modeling_viewport = NativeVtkPreviewWidget(viewport_wrapper, background=(0.94, 0.94, 0.94))
        vp_layout.addWidget(self._modeling_viewport, 1)
        body.addWidget(viewport_wrapper)

        # property panel
        prop_wrapper = QWidget()
        prop_wrapper.setStyleSheet("background: #252526;")
        prop_layout = QVBoxLayout(prop_wrapper)
        prop_layout.setContentsMargins(12, 12, 12, 12)
        prop_layout.setSpacing(10)
        prop_title = QLabel("属性")
        prop_title.setStyleSheet("font-size: 15px; font-weight: 600;")
        prop_layout.addWidget(prop_title)

        self._modeling_prop_name = QLineEdit()
        self._modeling_prop_name.setPlaceholderText("模型名称")
        self._modeling_prop_name.setEnabled(False)
        self._modeling_prop_name.editingFinished.connect(self._on_prop_name_changed)

        self._modeling_prop_kind = QLabel("—")
        self._modeling_prop_kind.setStyleSheet("color: #9da5b4;")

        pos_x = self._make_modeling_spinbox(-100, 100, 0.1, self._on_prop_transform_changed)
        pos_y = self._make_modeling_spinbox(-100, 100, 0.1, self._on_prop_transform_changed)
        pos_z = self._make_modeling_spinbox(-100, 100, 0.1, self._on_prop_transform_changed)
        rot_x = self._make_modeling_spinbox(-360, 360, 1.0, self._on_prop_transform_changed)
        rot_y = self._make_modeling_spinbox(-360, 360, 1.0, self._on_prop_transform_changed)
        rot_z = self._make_modeling_spinbox(-360, 360, 1.0, self._on_prop_transform_changed)
        scl_x = self._make_modeling_spinbox(0.01, 100, 0.1, self._on_prop_transform_changed)
        scl_y = self._make_modeling_spinbox(0.01, 100, 0.1, self._on_prop_transform_changed)
        scl_z = self._make_modeling_spinbox(0.01, 100, 0.1, self._on_prop_transform_changed)
        scl_x.setValue(1.0); scl_y.setValue(1.0); scl_z.setValue(1.0)
        self._modeling_prop_pos_x, self._modeling_prop_pos_y, self._modeling_prop_pos_z = pos_x, pos_y, pos_z
        self._modeling_prop_rot_x, self._modeling_prop_rot_y, self._modeling_prop_rot_z = rot_x, rot_y, rot_z
        self._modeling_prop_scl_x, self._modeling_prop_scl_y, self._modeling_prop_scl_z = scl_x, scl_y, scl_z

        self._modeling_color_btn = QPushButton("■")
        self._modeling_color_btn.setFixedSize(36, 36)
        self._modeling_color_btn.clicked.connect(self._on_color_pick)

        prop_layout.addWidget(QLabel("名称"))
        prop_layout.addWidget(self._modeling_prop_name)
        prop_layout.addWidget(QLabel("类型"))
        prop_layout.addWidget(self._modeling_prop_kind)
        prop_layout.addWidget(self._modeling_group("位置", pos_x, pos_y, pos_z))
        prop_layout.addWidget(self._modeling_group("旋转 (°)", rot_x, rot_y, rot_z))
        prop_layout.addWidget(self._modeling_group("缩放", scl_x, scl_y, scl_z))
        prop_layout.addWidget(QLabel("颜色"))
        prop_layout.addWidget(self._modeling_color_btn)
        prop_layout.addStretch(1)
        body.addWidget(prop_wrapper)

        body.setSizes([200, 520, 220])
        root.addWidget(body, 1)

        # --- status bar ---
        status_bar = QWidget()
        status_bar.setStyleSheet("background: #007acc;")
        sb_layout = QHBoxLayout(status_bar)
        sb_layout.setContentsMargins(12, 2, 12, 2)
        self._modeling_status_label = QLabel("就绪")
        self._modeling_status_label.setStyleSheet("color: #ffffff; background: transparent;")
        sb_layout.addWidget(self._modeling_status_label)
        sb_layout.addStretch(1)
        mode_label = QLabel("相机模式")
        mode_label.setStyleSheet("color: #ffffff; background: transparent;")
        sb_layout.addWidget(mode_label)
        root.addWidget(status_bar)

        return wrapper

    def _make_modeling_spinbox(self, min_val, max_val, step, callback):
        sb = QDoubleSpinBox()
        sb.setRange(min_val, max_val)
        sb.setDecimals(2)
        sb.setSingleStep(step)
        sb.valueChanged.connect(callback)
        return sb

    def _modeling_group(self, label_text, x, y, z) -> QGroupBox:
        gb = QGroupBox(label_text)
        form = QFormLayout(gb)
        form.setContentsMargins(8, 12, 8, 8)
        form.setSpacing(4)
        form.addRow("X", x)
        form.addRow("Y", y)
        form.addRow("Z", z)
        return gb

    def _build_physics_prepare_tab(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("求解器准备")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        description = QLabel(
            "这个页面负责启动仿真前的求解器输入准备：0/U、0/p、physicalProperties，"
            "以及按当前绘制几何自动衔接 snappyHexMeshDict。"
        )
        description.setWordWrap(True)

        action_row = QHBoxLayout()
        refresh_button = QPushButton("刷新求解器准备状态")
        prepare_button = QPushButton("补齐边界和物性")
        refresh_button.clicked.connect(lambda _checked=False: self._refresh_physics_prepare_panel())
        prepare_button.clicked.connect(lambda _checked=False: self._prepare_physics_files())
        for button in (refresh_button, prepare_button):
            action_row.addWidget(button)
        action_row.addStretch(1)

        self._physics_prepare_status = QTextEdit()
        self._physics_prepare_status.setReadOnly(True)
        self._physics_prepare_status.setMinimumHeight(150)
        self._physics_prepare_status.setPlainText("请先新建或打开项目。")

        material_form = QFormLayout()
        self._material_combo = QComboBox()
        self._material_combo.addItem("空气 air", "air")
        self._material_combo.addItem("水 water", "water")
        self._material_combo.addItem("机油 oil", "oil")
        self._material_combo.addItem("自定义流体 custom", "custom")
        self._material_combo.currentIndexChanged.connect(lambda _index: self._apply_material_preset())
        self._density_input = QLineEdit()
        self._viscosity_input = QLineEdit()
        self._dynamic_viscosity_input = QLineEdit()
        material_row = QHBoxLayout()
        calc_mu_button = QPushButton("按 rho × nu 计算 mu")
        calc_mu_button.clicked.connect(lambda _checked=False: self._calculate_dynamic_viscosity())
        material_row.addWidget(self._material_combo)
        material_row.addWidget(calc_mu_button)
        material_row.addStretch(1)
        material_form.addRow("流体类型", material_row)
        material_form.addRow("流体密度 rho", self._density_input)
        material_form.addRow("运动粘度 nu", self._viscosity_input)
        material_form.addRow("动力粘度 mu", self._dynamic_viscosity_input)

        self._boundary_table = QTableWidget(0, 6)
        self._boundary_table.setHorizontalHeaderLabels(["Patch", "角色", "U 类型", "U 值", "p 类型", "p 值"])
        self._boundary_table.horizontalHeader().setStretchLastSection(True)
        self._boundary_table.setMinimumHeight(220)

        self._physics_prepare_flow = QTextEdit()
        self._physics_prepare_flow.setReadOnly(True)
        self._physics_prepare_flow.setMaximumHeight(160)
        self._physics_prepare_flow.setPlainText(
            "本页负责：\n"
            "- 0/U：速度初始场和入口/壁面边界条件。\n"
            "- 0/p：压力初始场和出口/壁面边界条件。\n"
            "- constant/physicalProperties：密度、粘度等流体物性。\n\n"
            "推荐流程：绘制几何后到本页补齐边界和物性，然后进入求解器选择、仿真参数和求解运行。"
        )

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addLayout(action_row)
        layout.addWidget(self._physics_prepare_status)
        layout.addWidget(QLabel("流体物性文件"))
        layout.addLayout(material_form)
        layout.addWidget(QLabel("Patch 边界条件表"))
        layout.addWidget(self._boundary_table)
        layout.addWidget(self._physics_prepare_flow)
        layout.addStretch(1)
        return wrapper

    def _build_solver_run_tab(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("求解运行")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        description = QLabel("绘制几何 + 计算域 + 求解器 + 参数都配置好后，一键启动仿真。")
        description.setWordWrap(True)

        self._solver_status_label = QLabel("状态：空闲")
        self._solver_project_label = QLabel("当前项目：未选择")
        self._solver_case_path_label = QLabel("Case 路径：未选择")
        self._solver_command_label = QLabel("执行命令：blockMesh && icoFoam")

        action_row = QHBoxLayout()
        start_btn = QPushButton("一键启动仿真")

        start_btn.clicked.connect(lambda _checked=False: self._run_simulation_pipeline())
        stop_button = QPushButton("停止")
        stop_button.clicked.connect(lambda _checked=False: self._stop_current_process())
        refresh_button = QPushButton("刷新")
        refresh_button.clicked.connect(lambda _checked=False: self._refresh_solver_run_panel())
        action_row.addWidget(start_btn)
        action_row.addWidget(stop_button)
        action_row.addWidget(refresh_button)
        action_row.addStretch(1)

        self._solver_parameter_summary = QTextEdit()
        self._solver_parameter_summary.setReadOnly(True)
        self._solver_parameter_summary.setMaximumHeight(160)
        self._solver_parameter_summary.setPlainText("参数摘要：请先新建或打开项目。")

        self._solver_metric_summary = QTextEdit()
        self._solver_metric_summary.setReadOnly(True)
        self._solver_metric_summary.setMaximumHeight(150)
        self._solver_metric_summary.setPlainText("关键指标摘要：尚未运行。")

        self._solver_hint_text = QTextEdit()
        self._solver_hint_text.setReadOnly(True)
        self._solver_hint_text.setPlainText(
            "运行说明：\n"
            "1. 先在“参数配置”页确认参数。\n"
            "2. 点击“运行最小仿真”。\n"
            "3. 程序会先保存参数，再执行 blockMesh 和 icoFoam。\n"
            "4. 底部“日志”显示 OpenFOAM 实时输出。\n"
            "5. 底部“问题”显示失败原因。"
        )
        self._solver_diagnostic_text = QTextEdit()
        self._solver_diagnostic_text.setReadOnly(True)
        self._solver_diagnostic_text.setMaximumHeight(170)
        self._solver_diagnostic_text.setPlainText("最近诊断：暂无诊断。")

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(self._solver_status_label)
        layout.addWidget(self._solver_project_label)
        layout.addWidget(self._solver_case_path_label)
        layout.addWidget(self._solver_command_label)
        layout.addLayout(action_row)
        layout.addWidget(self._solver_parameter_summary)
        layout.addWidget(self._solver_metric_summary)
        layout.addWidget(self._solver_diagnostic_text)
        layout.addWidget(self._solver_hint_text, 1)
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
        refresh_button.clicked.connect(lambda _checked=False: self._refresh_environment_panels())

        layout.addWidget(title)
        layout.addWidget(refresh_button)
        layout.addWidget(self._environment_text)
        return wrapper

    def _build_results_tab(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("结果")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        description = QLabel("当前页面按“结果场选择 + 显示方式选择”组织后处理入口。先接入表面云图、切片、等值线、等值面和流线。")
        description.setWordWrap(True)

        result_data_button = self._make_menu_button(
            "结果数据",
            [
                ("刷新结果索引", self._refresh_results_panel),
                ("导出求解指标", self._export_solver_metrics),
                ("绘制残差曲线", self._plot_residual_curve),
                ("导出 Markdown 报告", self._export_markdown_report),
            ],
        )
        action_row = QHBoxLayout()
        action_row.addWidget(result_data_button)
        action_row.addStretch(1)
        action_bar = QWidget()
        action_bar.setLayout(action_row)
        action_scroll = QScrollArea()
        action_scroll.setWidget(action_bar)
        action_scroll.setWidgetResizable(True)
        action_scroll.setFrameShape(QFrame.Shape.NoFrame)
        action_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        action_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        action_scroll.setMaximumHeight(58)
        action_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        field_group = QFrame()
        field_group.setObjectName("sectionFrame")
        field_layout = QVBoxLayout(field_group)
        field_layout.setContentsMargins(12, 10, 12, 10)
        field_layout.setSpacing(8)
        field_title = QLabel("模块 1：结果场选择")
        field_title.setStyleSheet("font-size: 17px; font-weight: 600;")
        field_row = QHBoxLayout()
        self._result_field_combo = QComboBox()
        self._result_field_combo.addItems(self.RESULT_FIELDS)
        self._result_time_combo = QComboBox()
        self._result_color_min_input = QDoubleSpinBox()
        self._result_color_min_input.setRange(-1.0e12, 1.0e12)
        self._result_color_min_input.setDecimals(6)
        self._result_color_min_input.setValue(0.0)
        self._result_color_max_input = QDoubleSpinBox()
        self._result_color_max_input.setRange(-1.0e12, 1.0e12)
        self._result_color_max_input.setDecimals(6)
        self._result_color_max_input.setValue(1.0)
        self._result_unit_label = QLabel("单位：-")
        self._result_minmax_label = QLabel("最大/最小值：未刷新")
        refresh_fields_button = QPushButton("刷新字段/时间步")
        refresh_fields_button.clicked.connect(lambda _checked=False: self._refresh_result_field_panel())
        self._result_field_combo.currentTextChanged.connect(lambda _text: self._on_result_field_changed())
        field_row.addWidget(QLabel("变量"))
        field_row.addWidget(self._result_field_combo, 2)
        field_row.addWidget(QLabel("时间步"))
        field_row.addWidget(self._result_time_combo, 2)
        field_row.addWidget(QLabel("颜色最小"))
        field_row.addWidget(self._result_color_min_input, 1)
        field_row.addWidget(QLabel("颜色最大"))
        field_row.addWidget(self._result_color_max_input, 1)
        field_row.addWidget(refresh_fields_button)
        field_layout.addWidget(field_title)
        field_layout.addLayout(field_row)
        field_layout.addWidget(self._result_unit_label)
        field_layout.addWidget(self._result_minmax_label)

        display_group = QFrame()
        display_group.setObjectName("sectionFrame")
        display_layout = QVBoxLayout(display_group)
        display_layout.setContentsMargins(12, 10, 12, 10)
        display_layout.setSpacing(8)
        display_title = QLabel("模块 2：显示方式选择")
        display_title.setStyleSheet("font-size: 17px; font-weight: 600;")
        display_row = QHBoxLayout()
        self._result_display_combo = QComboBox()
        self._result_display_combo.addItems(self.RESULT_DISPLAY_MODES)
        self._result_slice_axis_combo = QComboBox()
        self._result_slice_axis_combo.addItems(["自动", "X", "Y", "Z"])
        self._result_slice_position_input = QDoubleSpinBox()
        self._result_slice_position_input.setRange(0.0, 1.0)
        self._result_slice_position_input.setSingleStep(0.05)
        self._result_slice_position_input.setDecimals(2)
        self._result_slice_position_input.setValue(0.5)
        load_display_button = QPushButton("加载显示")
        load_display_button.clicked.connect(lambda _checked=False: self._load_selected_result_display())
        display_row.addWidget(QLabel("显示方式"))
        display_row.addWidget(self._result_display_combo, 2)
        display_row.addWidget(QLabel("切面方向"))
        display_row.addWidget(self._result_slice_axis_combo)
        display_row.addWidget(QLabel("切面位置"))
        display_row.addWidget(self._result_slice_position_input)
        display_row.addWidget(load_display_button)
        display_hint = QLabel(
            "已接入：Surface、Slice、Contour、Iso-surface、Streamline。"
        )
        display_hint.setWordWrap(True)
        display_hint.setObjectName("sectionHint")
        display_layout.addWidget(display_title)
        display_layout.addLayout(display_row)
        display_layout.addWidget(display_hint)
        self._refresh_result_display_modes()

        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setPlainText("请先新建或打开项目，然后运行最小仿真。")
        self._results_text.setMaximumHeight(160)
        self._residual_figure = Figure(figsize=(6, 3), tight_layout=True)
        self._residual_canvas = FigureCanvas(self._residual_figure)
        self._residual_canvas.setMaximumHeight(220)
        self._vtk_hint_label = QLabel("三维结果仍在独立窗口中打开，避免 WSL 下 VTK 原生控件覆盖 Qt 主页面。")
        self._vtk_hint_label.setWordWrap(True)
        self._vtk_hint_label.setObjectName("sectionHint")

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(action_scroll)
        layout.addWidget(field_group)
        layout.addWidget(display_group)
        layout.addWidget(self._results_text, 1)
        layout.addWidget(self._residual_canvas, 2)
        layout.addWidget(self._vtk_hint_label)
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
        self._workspace_tabs.setCurrentIndex(self.TAB_SETTINGS)
        self._set_status("已打开设置页。")

    def _open_environment_tab(self) -> None:
        self._workspace_tabs.setCurrentIndex(self.TAB_ENVIRONMENT)
        self._refresh_environment_panels()
        self._set_status("已打开环境检查页。")

    def _open_draw_geometry_tab(self) -> None:
        self._workspace_tabs.setCurrentIndex(self.TAB_DRAW_GEOMETRY)
        self._set_status("已打开绘制几何页。")

    def _open_mesh_generation_tab(self) -> None:
        self._workspace_tabs.setCurrentIndex(self.TAB_MESH_GENERATION)
        self._refresh_mesh_generation_panel()
        self._set_status("已打开网格生成页。")


    def _set_parameter_inputs_enabled(self, enabled: bool) -> None:
        if not hasattr(self, "_end_time_input"):
            return
        self._end_time_input.setEnabled(enabled)
        self._delta_t_input.setEnabled(enabled)
        self._write_interval_input.setEnabled(enabled)
        self._set_physics_prepare_inputs_enabled(enabled)

    def _set_physics_prepare_inputs_enabled(self, enabled: bool) -> None:
        if not hasattr(self, "_material_combo"):
            return
        self._material_combo.setEnabled(enabled)
        self._density_input.setEnabled(enabled)
        self._viscosity_input.setEnabled(enabled)
        self._dynamic_viscosity_input.setEnabled(enabled)

    def _set_solver_inputs_enabled(self, enabled: bool) -> None:
        if not hasattr(self, "_solver_name_combo"):
            return
        self._solver_name_combo.setEnabled(enabled)
        self._turbulence_model_combo.setEnabled(enabled)
        self._numeric_scheme_combo.setEnabled(enabled)
        self._fv_solution_preset_combo.setEnabled(enabled)
