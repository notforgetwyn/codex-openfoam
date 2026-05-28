from __future__ import annotations

import shlex

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtCore import QProcess
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFontComboBox,
    QFrame,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QProgressBar,
    QMenuBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
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
from foamdesk.ui.main_window_geometry_logic import (
    BOUNDARY_TYPE_LABELS,
    GeometryLogicMixin,
)
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
    TAB_SIMULATION_CONFIG = 3
    TAB_SOLVER_RUN = 4
    TAB_ENVIRONMENT = 5
    TAB_SETTINGS = 6
    TAB_RESULTS = 7
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
        for obj in self._modeling_objects:
            self._modeling_viewport._renderer.AddActor(obj.actor)
            self._apply_transform(obj)
        self._rebuild_tree()
        self._apply_settings_theme()
        self._refresh_status_bar()
        if initial_project is not None:
            self._activate_project(initial_project, "已恢复上次项目。")

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_modeling_state()
        self._save_mesh_workflow_state()
        if self._foam_process and self._foam_process.state() != QProcess.ProcessState.NotRunning:
            self._foam_process.kill()
            self._foam_process.waitForFinished(1000)
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

        solver_menu = menu_bar.addMenu("求解器")
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
        workspace_widget = self._build_workspace()
        workspace_widget.setMinimumHeight(150)
        bottom_widget = self._build_bottom_panel()
        bottom_widget.setMinimumHeight(80)
        vertical_splitter.addWidget(workspace_widget)
        vertical_splitter.addWidget(bottom_widget)
        vertical_splitter.setSizes([640, 220])
        vertical_splitter.setChildrenCollapsible(False)
        vertical_splitter.setHandleWidth(6)
        vertical_splitter.setStretchFactor(0, 1)
        vertical_splitter.setStretchFactor(1, 0)
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
        self._workspace_tabs.addTab(self._build_simulation_config_tab(), "仿真参数配置")
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

        self._init_mesh_import_state()

        # ── bottom: VTK 3D view ──
        vtk_group = QGroupBox("几何预览")
        vtk_layout = QVBoxLayout(vtk_group)
        vtk_layout.setContentsMargins(0, 0, 0, 0)
        self._mesh_grid_vtk = NativeVtkPreviewWidget(wrapper, background=(0.12, 0.12, 0.12))
        vtk_layout.addWidget(self._mesh_grid_vtk)
        self._setup_boundary_picker()

        # ── top: scrollable parameter panel ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(16, 16, 16, 16)
        scroll_layout.setSpacing(12)

        # Group 1 — import & geometry management
        g1 = QGroupBox("导入与几何管理")
        g1_layout = QHBoxLayout(g1)
        import_btn = QPushButton("导入几何")
        import_btn.clicked.connect(self._import_geometry_file)
        self._mesh_import_combo = QComboBox()
        self._mesh_import_combo.setMinimumWidth(160)
        self._mesh_import_combo.currentIndexChanged.connect(
            self._on_mesh_import_selection_changed)
        self._mesh_import_visible_check = QCheckBox("显示几何")
        self._mesh_import_visible_check.setChecked(True)
        self._mesh_import_visible_check.toggled.connect(
            self._on_import_visibility_toggled)
        self._mesh_import_opacity_check = QCheckBox("半透明")
        self._mesh_import_opacity_check.toggled.connect(
            self._on_import_opacity_toggled)
        clear_btn = QPushButton("清空几何")
        clear_btn.clicked.connect(self._clear_mesh_imports)
        g1_layout.addWidget(import_btn)
        g1_layout.addWidget(QLabel("当前选中:"))
        g1_layout.addWidget(self._mesh_import_combo, 1)
        g1_layout.addWidget(self._mesh_import_visible_check)
        g1_layout.addWidget(self._mesh_import_opacity_check)
        g1_layout.addWidget(clear_btn)
        scroll_layout.addWidget(g1)

        # Group 2 — boundary face configuration
        g2 = QGroupBox("边界面配置")
        g2_layout = QHBoxLayout(g2)
        # left: pick controls
        left_widget = QWidget()
        left = QVBoxLayout(left_widget)
        left.setContentsMargins(0, 0, 0, 0)
        pick_mode_group = QButtonGroup(self)
        self._boundary_pick_point_rb = QRadioButton("点选面")
        self._boundary_pick_point_rb.setChecked(True)
        pick_mode_group.addButton(self._boundary_pick_point_rb)
        left.addWidget(self._boundary_pick_point_rb)
        self._boundary_pick_btn = QPushButton("拾取面")
        self._boundary_pick_btn.clicked.connect(self._toggle_boundary_pick)
        left.addWidget(self._boundary_pick_btn)
        cancel_pick_btn = QPushButton("取消选中")
        cancel_pick_btn.clicked.connect(
            lambda: self._boundary_pending_cells.clear() or self._redraw_mesh_grid_vtk())
        left.addWidget(cancel_pick_btn)
        clear_all_btn = QPushButton("清空所有边界")
        clear_all_btn.clicked.connect(self._clear_all_boundary_groups)
        left.addWidget(clear_all_btn)
        left.addStretch(1)
        g2_layout.addWidget(left_widget)
        # right: boundary properties + table
        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(0, 0, 0, 0)
        prop_row = QHBoxLayout()
        self._boundary_type_combo = QComboBox()
        for key, label in BOUNDARY_TYPE_LABELS.items():
            self._boundary_type_combo.addItem(label, key)
        self._boundary_type_combo.currentIndexChanged.connect(
            self._on_boundary_type_changed)
        self._boundary_name_input = QLineEdit()
        self._boundary_name_input.setPlaceholderText("边界名称（默认同类型）")
        apply_btn = QPushButton("应用到选中面")
        apply_btn.clicked.connect(self._apply_boundary_to_selected)
        prop_row.addWidget(QLabel("类型:"))
        prop_row.addWidget(self._boundary_type_combo)
        prop_row.addWidget(QLabel("名称:"))
        prop_row.addWidget(self._boundary_name_input, 1)
        prop_row.addWidget(apply_btn)
        right.addLayout(prop_row)
        self._boundary_table = QTableWidget(0, 4)
        self._boundary_table.setHorizontalHeaderLabels(
            ["边界名称", "边界类型", "面片数量", "操作"])
        self._boundary_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        right.addWidget(self._boundary_table)
        g2_layout.addWidget(right_widget, 1)
        scroll_layout.addWidget(g2)

        # Group 3 — background mesh / blockMeshDict
        g3 = QGroupBox("基础计算域网格（背景网格）")
        g3_layout = QVBoxLayout(g3)
        size_label = QLabel("计算域尺寸（坐标范围）")
        size_label.setStyleSheet("font-weight: 600; color: #cccccc;")
        g3_layout.addWidget(size_label)
        bounds_grid = QGridLayout()
        bounds_grid.setSpacing(6)
        axes = [
            ("X 最小", "X 最大"),
            ("Y 最小", "Y 最大"),
            ("Z 最小", "Z 最大"),
        ]
        self._domain_bounds_inputs: dict[str, QDoubleSpinBox] = {}
        for row, (min_label, max_label) in enumerate(axes):
            for col, (label_text, key) in enumerate(
                [(min_label, f"{'XYZ'[row].lower()}_min"),
                 (max_label, f"{'XYZ'[row].lower()}_max")]):
                lbl = QLabel(label_text)
                spin = QDoubleSpinBox()
                spin.setRange(-1e6, 1e6)
                spin.setDecimals(4)
                spin.setMinimumWidth(100)
                spin.valueChanged.connect(self._on_domain_manual_override)
                bounds_grid.addWidget(lbl, row, col * 2)
                bounds_grid.addWidget(spin, row, col * 2 + 1)
                self._domain_bounds_inputs[key] = spin
        g3_layout.addLayout(bounds_grid)

        resolution_label = QLabel("网格分辨率")
        resolution_label.setStyleSheet("font-weight: 600; color: #cccccc; margin-top: 8px;")
        g3_layout.addWidget(resolution_label)
        res_row = QHBoxLayout()
        self._domain_cells_inputs: dict[str, QSpinBox] = {}
        for axis in ("X", "Y", "Z"):
            lbl = QLabel(f"{axis}方向")
            spin = QSpinBox()
            spin.setRange(1, 9999)
            spin.setValue(20)
            spin.setToolTip(f"{axis}方向网格数量")
            res_row.addWidget(lbl)
            res_row.addWidget(spin)
            self._domain_cells_inputs[axis] = spin
        res_row.addStretch(1)
        g3_layout.addLayout(res_row)

        opt_row = QHBoxLayout()
        self._domain_orthogonal_check = QCheckBox("正交网格")
        opt_row.addWidget(self._domain_orthogonal_check)
        opt_row.addWidget(QLabel("网格单位:"))
        self._domain_unit_combo = QComboBox()
        self._domain_unit_combo.addItem("米 (m)", "m")
        self._domain_unit_combo.addItem("毫米 (mm)", "mm")
        opt_row.addWidget(self._domain_unit_combo)
        auto_btn = QPushButton("从几何自动计算")
        auto_btn.clicked.connect(self._auto_fill_domain_bounds)
        opt_row.addWidget(auto_btn)
        opt_row.addStretch(1)
        g3_layout.addLayout(opt_row)
        scroll_layout.addWidget(g3)

        # Group 4 — snappyHexMesh parameters
        g4 = QGroupBox("模型贴体网格加密")
        g4_layout = QVBoxLayout(g4)

        refine_row = QHBoxLayout()
        refine_row.addWidget(QLabel("全局加密等级:"))
        self._snappy_level_combo = QComboBox()
        for level in range(1, 6):
            self._snappy_level_combo.addItem(f"等级 {level}", level)
        self._snappy_level_combo.setCurrentIndex(2)  # default level 3
        refine_row.addWidget(self._snappy_level_combo)
        refine_row.addStretch(1)
        g4_layout.addLayout(refine_row)

        layer_label = QLabel("壁面边界层设置")
        layer_label.setStyleSheet("font-weight: 600; color: #cccccc; margin-top: 4px;")
        g4_layout.addWidget(layer_label)
        layer_row = QHBoxLayout()
        layer_row.addWidget(QLabel("边界层层数:"))
        self._snappy_n_layers = QSpinBox()
        self._snappy_n_layers.setRange(0, 20)
        self._snappy_n_layers.setValue(3)
        layer_row.addWidget(self._snappy_n_layers)
        layer_row.addWidget(QLabel("第一层高度:"))
        self._snappy_first_layer = QDoubleSpinBox()
        self._snappy_first_layer.setRange(0.0001, 10.0)
        self._snappy_first_layer.setDecimals(6)
        self._snappy_first_layer.setValue(0.001)
        layer_row.addWidget(self._snappy_first_layer)
        layer_row.addWidget(QLabel("增长比:"))
        self._snappy_expand_ratio = QDoubleSpinBox()
        self._snappy_expand_ratio.setRange(1.0, 2.0)
        self._snappy_expand_ratio.setDecimals(2)
        self._snappy_expand_ratio.setSingleStep(0.05)
        self._snappy_expand_ratio.setValue(1.2)
        layer_row.addWidget(self._snappy_expand_ratio)
        layer_row.addStretch(1)
        g4_layout.addLayout(layer_row)

        local_row = QHBoxLayout()
        self._snappy_local_enabled = QCheckBox("启用局部加密")
        self._snappy_local_enabled.toggled.connect(
            lambda checked: self._snappy_local_part_combo.setEnabled(checked))
        local_row.addWidget(self._snappy_local_enabled)
        local_row.addWidget(QLabel("选中部件:"))
        self._snappy_local_part_combo = QComboBox()
        self._snappy_local_part_combo.setMinimumWidth(140)
        self._snappy_local_part_combo.setEnabled(False)
        local_row.addWidget(self._snappy_local_part_combo)
        local_row.addWidget(QLabel("加密等级:"))
        self._snappy_local_level_combo = QComboBox()
        for level in range(1, 6):
            self._snappy_local_level_combo.addItem(str(level), level)
        self._snappy_local_level_combo.setEnabled(False)
        local_row.addWidget(self._snappy_local_level_combo)
        local_row.addStretch(1)
        g4_layout.addLayout(local_row)

        self._snappy_keep_outline = QCheckBox("保留原始边界轮廓")
        self._snappy_keep_outline.setChecked(True)
        g4_layout.addWidget(self._snappy_keep_outline)
        scroll_layout.addWidget(g4)

        # Group 5 — mesh quality constraints
        g5 = QGroupBox("网格质量约束")
        g5_layout = QHBoxLayout(g5)
        g5_layout.addWidget(QLabel("最大网格畸变率:"))
        self._quality_max_skew = QDoubleSpinBox()
        self._quality_max_skew.setRange(0.1, 1.0)
        self._quality_max_skew.setDecimals(2)
        self._quality_max_skew.setSingleStep(0.05)
        self._quality_max_skew.setValue(0.8)
        self._quality_max_skew.setToolTip("超过此畸变率的网格将在导出时剔除")
        g5_layout.addWidget(self._quality_max_skew)
        g5_layout.addWidget(QLabel("最小网格体积:"))
        self._quality_min_volume = QDoubleSpinBox()
        self._quality_min_volume.setRange(0.0, 1.0)
        self._quality_min_volume.setDecimals(8)
        self._quality_min_volume.setValue(1e-12)
        self._quality_min_volume.setToolTip("体积小于此值的网格视为无效")
        g5_layout.addWidget(self._quality_min_volume)
        self._quality_del_negative = QCheckBox("自动删除负体积网格")
        self._quality_del_negative.setChecked(True)
        g5_layout.addWidget(self._quality_del_negative)
        self._quality_smooth = QCheckBox("网格平滑处理")
        self._quality_smooth.setChecked(True)
        g5_layout.addWidget(self._quality_smooth)
        g5_layout.addStretch(1)
        scroll_layout.addWidget(g5)

        # Group 6 — action button bar
        btn_row = QHBoxLayout()
        btn_gen = QPushButton("生成字典并执行")
        btn_gen.clicked.connect(self._on_generate_and_execute)
        btn_gen.setStyleSheet("font-weight: bold;")
        btn_preview = QPushButton("预览网格")
        btn_preview.clicked.connect(self._on_preview_mesh)
        btn_check = QPushButton("检查网格质量")
        btn_check.clicked.connect(self._on_check_quality)
        btn_export_stl = QPushButton("导出STL(按边界拆分)")
        btn_export_stl.clicked.connect(self._on_export_boundary_stl)
        btn_reset = QPushButton("重置所有参数")
        btn_reset.clicked.connect(self._on_reset_all_params)
        for btn in (btn_gen, btn_preview, btn_check, btn_export_stl, btn_reset):
            btn.setMinimumHeight(32)
            btn_row.addWidget(btn)
        btn_row.addStretch(1)
        scroll_layout.addLayout(btn_row)

        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_widget)

        # ── splitter ──
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(scroll)
        splitter.addWidget(vtk_group)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        return wrapper


    def _build_simulation_config_tab(self) -> QWidget:
        wrapper = QWidget()
        root = QVBoxLayout(wrapper)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("仿真参数配置")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        root.addWidget(title)

        # top area: parameter groups in a 2x2 grid
        top = QHBoxLayout()
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()

        # === solver basics group ===
        solver_grp = QGroupBox("求解器基础")
        solver_form = QFormLayout(solver_grp)
        self._cfg_solver_combo = QComboBox()
        self._cfg_solver_combo.addItem("simpleFoam - 稳态不可压", "simpleFoam")
        self._cfg_solver_combo.addItem("pimpleFoam - 瞬态不可压", "pimpleFoam")
        self._cfg_solver_combo.addItem("icoFoam - 入门瞬态", "icoFoam")
        self._cfg_solver_combo.addItem("pisoFoam - 瞬态不可压", "pisoFoam")
        self._cfg_end_time = QLineEdit("1.0")
        self._cfg_delta_t = QLineEdit("0.001")
        self._cfg_write_interval = QSpinBox()
        self._cfg_write_interval.setRange(1, 1000000)
        self._cfg_write_interval.setValue(100)
        self._cfg_init_velocity = QLineEdit("(0 0 0)")
        self._cfg_init_pressure = QLineEdit("0")
        solver_form.addRow("求解类型", self._cfg_solver_combo)
        solver_form.addRow("总计算时长 (s)", self._cfg_end_time)
        solver_form.addRow("时间步长 (s)", self._cfg_delta_t)
        solver_form.addRow("输出间隔 (步)", self._cfg_write_interval)
        solver_form.addRow("初始速度 (m/s)", self._cfg_init_velocity)
        solver_form.addRow("初始压力 (Pa)", self._cfg_init_pressure)
        left_col.addWidget(solver_grp)

        # === fluid properties group ===
        fluid_grp = QGroupBox("流体物性")
        fluid_form = QFormLayout(fluid_grp)
        self._cfg_material_combo = QComboBox()
        self._cfg_material_combo.addItem("空气 air (rho=1.225, nu=1.48e-5)", "air")
        self._cfg_material_combo.addItem("水 water (rho=1000, nu=1e-6)", "water")
        self._cfg_material_combo.addItem("机油 oil (rho=880, nu=5e-5)", "oil")
        self._cfg_material_combo.addItem("自定义 custom", "custom")
        self._cfg_material_combo.currentIndexChanged.connect(self._on_material_preset_changed)
        self._cfg_density = QLineEdit("1.225")
        self._cfg_nu = QLineEdit("1.48e-5")
        self._cfg_mu = QLineEdit("1.812e-5")
        calc_mu_btn = QPushButton("计算 mu = rho * nu")
        calc_mu_btn.clicked.connect(self._calc_mu)
        mu_row = QHBoxLayout()
        mu_row.addWidget(self._cfg_mu)
        mu_row.addWidget(calc_mu_btn)
        fluid_form.addRow("材料预设", self._cfg_material_combo)
        fluid_form.addRow("密度 rho (kg/m3)", self._cfg_density)
        fluid_form.addRow("运动粘度 nu (m2/s)", self._cfg_nu)
        fluid_form.addRow("动力粘度 mu", mu_row)
        left_col.addWidget(fluid_grp)

        # === boundary conditions group ===
        bc_grp = QGroupBox("边界条件")
        bc_layout = QVBoxLayout(bc_grp)
        bc_top = QHBoxLayout()
        self._cfg_boundary_table = QTableWidget(0, 4)
        self._cfg_boundary_table.setHorizontalHeaderLabels(["边界名", "类型", "速度 U", "压力 p"])
        self._cfg_boundary_table.horizontalHeader().setStretchLastSection(True)
        self._cfg_boundary_table.setMinimumHeight(130)
        self._cfg_boundary_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._cfg_boundary_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._cfg_boundary_table.selectionModel().selectionChanged.connect(
            lambda sel, desel: self._on_boundary_row_selected(
                self._cfg_boundary_table.currentRow()
            )
        )
        bc_layout.addWidget(self._cfg_boundary_table)
        bc_form = QFormLayout()
        self._cfg_bc_name = QLabel("—")
        bc_form.addRow("选中边界", self._cfg_bc_name)
        self._cfg_bc_type = QComboBox()
        self._cfg_bc_type.addItems(["inlet", "outlet", "wall", "symmetry", "empty"])
        self._cfg_bc_type.currentIndexChanged.connect(self._on_boundary_type_changed)
        bc_form.addRow("边界类型", self._cfg_bc_type)
        self._cfg_bc_u_value = QLineEdit()
        self._cfg_bc_u_value.editingFinished.connect(
            lambda: self._sync_boundary_table_from_ui(self._cfg_boundary_table.currentRow())
        )
        bc_form.addRow("速度 U", self._cfg_bc_u_value)
        self._cfg_bc_p_value = QLineEdit()
        self._cfg_bc_p_value.editingFinished.connect(
            lambda: self._sync_boundary_table_from_ui(self._cfg_boundary_table.currentRow())
        )
        bc_form.addRow("压力 p", self._cfg_bc_p_value)
        bc_layout.addLayout(bc_form)
        right_col.addWidget(bc_grp)

        # === turbulence model group ===
        turb_grp = QGroupBox("湍流模型")
        turb_form = QFormLayout(turb_grp)
        self._cfg_turb_model = QComboBox()
        self._cfg_turb_model.addItem("laminar - 层流", "laminar")
        self._cfg_turb_model.addItem("kEpsilon - 标准 k-ε", "kEpsilon")
        self._cfg_turb_model.addItem("kOmega - k-ω SST", "kOmegaSST")
        self._cfg_turb_model.addItem("SpalartAllmaras - SA 模型", "SpalartAllmaras")
        self._cfg_turb_model.currentIndexChanged.connect(self._on_turb_model_changed)
        self._cfg_turb_intensity = QLineEdit("0.05")
        self._cfg_turb_length = QLineEdit("0.1")
        turb_form.addRow("湍流类型", self._cfg_turb_model)
        turb_form.addRow("湍流强度", self._cfg_turb_intensity)
        turb_form.addRow("混合长度 (m)", self._cfg_turb_length)
        right_col.addWidget(turb_grp)

        # === solver control group ===
        ctrl_grp = QGroupBox("求解控制")
        ctrl_form = QFormLayout(ctrl_grp)
        self._cfg_residual = QLineEdit("1e-6")
        self._cfg_max_iters = QSpinBox()
        self._cfg_max_iters.setRange(1, 100000)
        self._cfg_max_iters.setValue(1000)
        self._cfg_relaxation = QLineEdit("0.7")
        self._cfg_fv_schemes = QComboBox()
        self._cfg_fv_schemes.addItem("稳定 (upwind)", "stable")
        self._cfg_fv_schemes.addItem("平衡 (linearUpwind)", "balanced")
        self._cfg_fv_schemes.addItem("精度 (linear)", "accurate")
        self._cfg_fv_solution = QComboBox()
        self._cfg_fv_solution.addItem("默认收敛", "default")
        self._cfg_fv_solution.addItem("严格收敛", "strict")
        self._cfg_fv_solution.addItem("快速粗糙", "fast")
        ctrl_form.addRow("残差收敛阈值", self._cfg_residual)
        ctrl_form.addRow("最大迭代步数", self._cfg_max_iters)
        ctrl_form.addRow("松弛因子", self._cfg_relaxation)
        ctrl_form.addRow("数值格式 fvSchemes", self._cfg_fv_schemes)
        ctrl_form.addRow("求解设置 fvSolution", self._cfg_fv_solution)
        right_col.addWidget(ctrl_grp)

        top.addLayout(left_col)
        top.addLayout(right_col)
        root.addLayout(top)

        # === bottom: dictionary preview + export ===
        preview_label = QLabel("字典预览")
        preview_label.setStyleSheet("font-weight: 600; margin-top: 8px;")
        root.addWidget(preview_label)
        self._cfg_dict_preview = QTextEdit()
        self._cfg_dict_preview.setReadOnly(True)
        self._cfg_dict_preview.setMinimumHeight(150)
        self._cfg_dict_preview.setPlaceholderText("配置参数后点击 [刷新预览] 查看生成的 OpenFOAM 字典文件内容...")
        root.addWidget(self._cfg_dict_preview, 1)

        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("刷新预览")
        refresh_btn.clicked.connect(self._refresh_dict_preview)
        export_btn = QPushButton("导出全部仿真字典")
        export_btn.clicked.connect(self._export_sim_dicts)
        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(export_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        return wrapper

    def _load_boundaries_into_table(self) -> None:
        self._cfg_boundary_table.setRowCount(0)
        if self._current_project is None:
            return
        bmd = self._current_project.case_dir / "system" / "blockMeshDict"
        names = self._context.project_service._extract_boundary_names(bmd)
        if not names:
            names = ("inlet", "outlet", "fixedWalls")
        for i, name in enumerate(names):
            self._cfg_boundary_table.insertRow(i)
            self._cfg_boundary_table.setItem(i, 0, QTableWidgetItem(name))
            role = "入口" if "inlet" in name.lower() else "出口" if "outlet" in name.lower() else "壁面" if "wall" in name.lower() else "对称"
            self._cfg_boundary_table.setItem(i, 1, QTableWidgetItem(role))
            u_val = "(10 0 0)" if role == "入口" else "noSlip" if role == "壁面" else "zeroGradient"
            p_val = "0" if role == "出口" else "zeroGradient"
            self._cfg_boundary_table.setItem(i, 2, QTableWidgetItem(u_val))
            self._cfg_boundary_table.setItem(i, 3, QTableWidgetItem(p_val))
        if self._cfg_boundary_table.rowCount() > 0:
            self._cfg_boundary_table.selectRow(0)
            self._on_boundary_row_selected(0)

    def _on_boundary_row_selected(self, row: int) -> None:
        if row < 0 or row >= self._cfg_boundary_table.rowCount():
            return
        name_item = self._cfg_boundary_table.item(row, 0)
        type_item = self._cfg_boundary_table.item(row, 1)
        u_item = self._cfg_boundary_table.item(row, 2)
        p_item = self._cfg_boundary_table.item(row, 3)
        if name_item:
            self._cfg_bc_name.setText(name_item.text())
        if type_item:
            role = type_item.text()
            idx = self._cfg_bc_type.findText("inlet" if role == "入口" else "outlet" if role == "出口" else "wall")
            if idx >= 0:
                self._cfg_bc_type.setCurrentIndex(idx)
        if u_item:
            self._cfg_bc_u_value.setText(u_item.text())
        if p_item:
            self._cfg_bc_p_value.setText(p_item.text())

    def _sync_boundary_table_from_ui(self, row: int) -> None:
        if row < 0 or row >= self._cfg_boundary_table.rowCount():
            return
        role_text = {0: "入口", 1: "出口", 2: "壁面", 3: "对称", 4: "empty"}
        bc_type_idx = self._cfg_bc_type.currentIndex()
        role = role_text.get(bc_type_idx, "壁面")
        self._cfg_boundary_table.item(row, 1).setText(role)
        self._cfg_boundary_table.item(row, 2).setText(self._cfg_bc_u_value.text())
        self._cfg_boundary_table.item(row, 3).setText(self._cfg_bc_p_value.text())

    def _on_boundary_type_changed(self) -> None:
        row = self._cfg_boundary_table.currentRow()
        if row < 0:
            return
        role_text = {0: "入口", 1: "出口", 2: "壁面", 3: "对称", 4: "empty"}
        role = role_text.get(self._cfg_bc_type.currentIndex(), "壁面")
        if role == "入口":
            self._cfg_bc_u_value.setText("(10 0 0)")
            self._cfg_bc_p_value.setText("zeroGradient")
        elif role == "出口":
            self._cfg_bc_u_value.setText("zeroGradient")
            self._cfg_bc_p_value.setText("0")
        else:
            self._cfg_bc_u_value.setText("noSlip")
            self._cfg_bc_p_value.setText("zeroGradient")
        self._sync_boundary_table_from_ui(row)

    def _read_boundary_rows(self) -> list[dict]:
        rows = []
        for r in range(self._cfg_boundary_table.rowCount()):
            name = self._cfg_boundary_table.item(r, 0)
            role = self._cfg_boundary_table.item(r, 1)
            u_val = self._cfg_boundary_table.item(r, 2)
            p_val = self._cfg_boundary_table.item(r, 3)
            if name and role and u_val and p_val:
                rows.append({"name": name.text(), "role": role.text(),
                             "u_value": u_val.text(), "p_value": p_val.text()})
        return rows

    def _build_boundary_block(self, field: str) -> str:
        rows = self._read_boundary_rows()
        if not rows:
            return "    // 无边界"
        blocks = []
        for bc in rows:
            name = bc["name"]
            if field == "U":
                val = bc["u_value"]
                if val == "noSlip":
                    blocks.append(f"    {name}\n    {{\n        type            noSlip;\n    }}")
                elif val == "zeroGradient":
                    blocks.append(f"    {name}\n    {{\n        type            zeroGradient;\n    }}")
                elif val == "symmetry":
                    blocks.append(f"    {name}\n    {{\n        type            symmetry;\n    }}")
                else:
                    blocks.append(f"    {name}\n    {{\n        type            fixedValue;\n        value           uniform {val};\n    }}")
            else:  # p
                val = bc["p_value"]
                if val == "zeroGradient":
                    blocks.append(f"    {name}\n    {{\n        type            zeroGradient;\n    }}")
                elif val == "symmetry":
                    blocks.append(f"    {name}\n    {{\n        type            symmetry;\n    }}")
                else:
                    blocks.append(f"    {name}\n    {{\n        type            fixedValue;\n        value           uniform {val};\n    }}")
        return "\n".join(blocks)

    def _refresh_dict_preview(self) -> None:
        solver = self._cfg_solver_combo.currentData()
        end_time = self._cfg_end_time.text().strip()
        delta_t = self._cfg_delta_t.text().strip()
        write_interval = self._cfg_write_interval.value()
        init_u = self._cfg_init_velocity.text().strip()
        init_p = self._cfg_init_pressure.text().strip()
        residual = self._cfg_residual.text().strip()
        relaxation = self._cfg_relaxation.text().strip()
        fv_schemes_key = self._cfg_fv_schemes.currentData()
        fv_solution_key = self._cfg_fv_solution.currentData()
        fv_schemes_text = self._build_fv_schemes_text(fv_schemes_key)
        fv_solution_text = self._build_fv_solution_text(fv_solution_key, residual, relaxation)
        rho = self._cfg_density.text().strip()
        nu = self._cfg_nu.text().strip()
        mu = self._cfg_mu.text().strip()
        turb_model = self._cfg_turb_model.currentData()
        turb_intensity = self._cfg_turb_intensity.text().strip()
        turb_length = self._cfg_turb_length.text().strip()

        transport_props = (
            f"// constant/transportProperties\n"
            f"transportModel  Newtonian;\n"
            f"rho             rho [1 -3 0 0 0 0 0] {rho};\n"
            f"nu              nu [0 2 -1 0 0 0 0] {nu};\n"
            f"mu              mu [1 -1 -1 0 0 0 0] {mu};\n"
        )

        if turb_model == "laminar":
            turb_props = (
                f"// constant/turbulenceProperties\n"
                f"simulationType  laminar;\n"
            )
        else:
            ras_model = "kEpsilon" if turb_model == "kEpsilon" else "kOmegaSST" if turb_model == "kOmegaSST" else turb_model
            turb_props = (
                f"// constant/turbulenceProperties\n"
                f"simulationType  RAS;\n"
                f"RAS\n{{\n"
                f"    RASModel        {ras_model};\n"
                f"    turbulence      on;\n"
                f"    printCoeffs     on;\n"
                f"}}\n"
            )

        preview = (
            f"// system/controlDict\n"
            f"application     {solver};\n"
            f"startFrom       startTime;\n"
            f"startTime       0;\n"
            f"stopAt          endTime;\n"
            f"endTime         {end_time};\n"
            f"deltaT          {delta_t};\n"
            f"writeControl    timeStep;\n"
            f"writeInterval   {write_interval};\n\n"
            f"// 0/U\n"
            f"dimensions      [0 1 -1 0 0 0 0];\n"
            f"internalField   uniform {init_u};\n"
            f"boundaryField\n{{\n"
            f"{self._build_boundary_block('U')}\n"
            f"}}\n\n"
            f"// 0/p\n"
            f"dimensions      [0 2 -2 0 0 0 0];\n"
            f"internalField   uniform {init_p};\n"
            f"boundaryField\n{{\n"
            f"{self._build_boundary_block('p')}\n"
            f"}}\n\n"
            f"{transport_props}\n\n"
            f"{turb_props}\n\n"
            f"// system/fvSchemes\n{fv_schemes_text}\n\n"
            f"// system/fvSolution\n{fv_solution_text}\n"
        )
        self._cfg_dict_preview.setPlainText(preview)
        self._set_status("字典预览已刷新")

    def _build_fv_schemes_text(self, key: str) -> str:
        if key == "stable":
            return (
                "ddtSchemes   { default Euler; }\n"
                "gradSchemes  { default Gauss linear; }\n"
                "divSchemes   { default Gauss upwind; }\n"
                "laplacianSchemes { default Gauss linear corrected; }\n"
                "interpolationSchemes { default linear; }\n"
                "snGradSchemes { default corrected; }"
            )
        elif key == "accurate":
            return (
                "ddtSchemes   { default backward; }\n"
                "gradSchemes  { default Gauss linear; }\n"
                "divSchemes   { default Gauss linear; }\n"
                "laplacianSchemes { default Gauss linear corrected; }\n"
                "interpolationSchemes { default linear; }\n"
                "snGradSchemes { default corrected; }"
            )
        else:
            return (
                "ddtSchemes   { default Euler; }\n"
                "gradSchemes  { default Gauss linear; }\n"
                "divSchemes   { default Gauss linearUpwind grad(U); }\n"
                "laplacianSchemes { default Gauss linear corrected; }\n"
                "interpolationSchemes { default linear; }\n"
                "snGradSchemes { default corrected; }"
            )

    def _build_fv_solution_text(self, key: str, residual: str, relaxation: str) -> str:
        ncorrectors = "1" if key == "fast" else "2"
        reltol = "0.1" if key == "fast" else "0.01" if key == "default" else "0.001"
        return (
            f"solvers\n{{\n"
            f"  p {{ solver PCG; preconditioner DIC; tolerance {residual}; relTol {reltol}; }}\n"
            f"  U {{ solver smoothSolver; smoother symGaussSeidel; tolerance {residual}; relTol {reltol}; }}\n"
            f"}}\n"
            f"PIMPLE {{ nCorrectors {ncorrectors}; nNonOrthogonalCorrectors 0; }}\n"
            f"relaxationFactors {{ U {relaxation}; }}"
        )

    def _export_sim_dicts(self) -> None:
        if self._current_project is None:
            self._set_status("请先新建或打开项目")
            return
        case_dir = self._current_project.case_dir
        for d in ("system", "0"):
            (case_dir / d).mkdir(parents=True, exist_ok=True)

        solver = self._cfg_solver_combo.currentData()
        end_time = self._cfg_end_time.text().strip()
        delta_t = self._cfg_delta_t.text().strip()
        write_interval = self._cfg_write_interval.value()
        init_u = self._cfg_init_velocity.text().strip()
        init_p = self._cfg_init_pressure.text().strip()
        residual = self._cfg_residual.text().strip()
        relaxation = self._cfg_relaxation.text().strip()
        fv_schemes_key = self._cfg_fv_schemes.currentData()
        fv_solution_key = self._cfg_fv_solution.currentData()
        rho = self._cfg_density.text().strip()
        nu = self._cfg_nu.text().strip()
        mu = self._cfg_mu.text().strip()
        turb_model = self._cfg_turb_model.currentData()

        control_dict = (
            f"application     {solver};\n"
            f"startFrom       startTime;\n"
            f"startTime       0;\n"
            f"stopAt          endTime;\n"
            f"endTime         {end_time};\n"
            f"deltaT          {delta_t};\n"
            f"writeControl    timeStep;\n"
            f"writeInterval   {write_interval};\n"
        )
        (case_dir / "system" / "controlDict").write_text(control_dict, encoding="utf-8")

        u_field = (
            f"dimensions      [0 1 -1 0 0 0 0];\n"
            f"internalField   uniform {init_u};\n"
            f"boundaryField\n{{\n"
            f"{self._build_boundary_block('U')}\n"
            f"}}\n"
        )
        (case_dir / "0" / "U").write_text(u_field, encoding="utf-8")

        p_field = (
            f"dimensions      [0 2 -2 0 0 0 0];\n"
            f"internalField   uniform {init_p};\n"
            f"boundaryField\n{{\n"
            f"{self._build_boundary_block('p')}\n"
            f"}}\n"
        )
        (case_dir / "0" / "p").write_text(p_field, encoding="utf-8")

        (case_dir / "constant").mkdir(parents=True, exist_ok=True)
        transport = (
            f"transportModel  Newtonian;\n"
            f"rho             rho [1 -3 0 0 0 0 0] {rho};\n"
            f"nu              nu [0 2 -1 0 0 0 0] {nu};\n"
            f"mu              mu [1 -1 -1 0 0 0 0] {mu};\n"
        )
        (case_dir / "constant" / "transportProperties").write_text(transport, encoding="utf-8")

        if turb_model == "laminar":
            turb = "simulationType  laminar;\n"
        else:
            ras_model = "kEpsilon" if turb_model == "kEpsilon" else "kOmegaSST" if turb_model == "kOmegaSST" else turb_model
            turb = (
                f"simulationType  RAS;\n"
                f"RAS\n{{\n"
                f"    RASModel        {ras_model};\n"
                f"    turbulence      on;\n"
                f"    printCoeffs     on;\n"
                f"}}\n"
            )
        (case_dir / "constant" / "turbulenceProperties").write_text(turb, encoding="utf-8")

        (case_dir / "system" / "fvSchemes").write_text(
            self._build_fv_schemes_text(fv_schemes_key), encoding="utf-8"
        )
        (case_dir / "system" / "fvSolution").write_text(
            self._build_fv_solution_text(fv_solution_key, residual, relaxation), encoding="utf-8"
        )

        self._set_status(f"字典已导出到 {case_dir}")
        self._workspace_tabs.setCurrentIndex(self.TAB_SOLVER_RUN)

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

        import_btn = QPushButton("导入 STL")
        import_btn.setFixedHeight(30)
        import_btn.clicked.connect(self._import_stl_file)
        toolbar.addWidget(import_btn)

        export_btn = QPushButton("导出 STL")
        export_btn.setFixedHeight(30)
        export_btn.clicked.connect(self._export_stl_file)
        toolbar.addWidget(export_btn)

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
        prop_layout.addWidget(QLabel("透明度"))
        self._modeling_prop_opacity = QDoubleSpinBox()
        self._modeling_prop_opacity.setRange(0.1, 1.0)
        self._modeling_prop_opacity.setSingleStep(0.1)
        self._modeling_prop_opacity.setValue(1.0)
        self._modeling_prop_opacity.valueChanged.connect(self._on_prop_opacity_changed)
        prop_layout.addWidget(self._modeling_prop_opacity)
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

    def _build_solver_run_tab(self) -> QWidget:
        wrapper = QWidget()
        root = QVBoxLayout(wrapper)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        title = QLabel("仿真运行监控")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        root.addWidget(title)

        control_row = QHBoxLayout()
        self._cfg_start_btn = QPushButton("启动计算")
        self._cfg_start_btn.setFixedHeight(34)
        self._cfg_start_btn.setStyleSheet("background: #0e639c; color: #fff; font-weight: 600;")
        self._cfg_start_btn.clicked.connect(self._start_simulation)
        self._cfg_stop_btn = QPushButton("终止计算")
        self._cfg_stop_btn.setFixedHeight(34)
        self._cfg_stop_btn.setStyleSheet("color: #f48771;")
        self._cfg_stop_btn.clicked.connect(self._stop_current_process)
        self._cfg_stop_btn.setEnabled(False)
        self._cfg_progress = QProgressBar()
        self._cfg_progress.setRange(0, 0)
        self._cfg_progress.setFixedWidth(200)
        self._cfg_progress.setFixedHeight(20)
        self._cfg_progress.setTextVisible(False)
        self._cfg_progress.setVisible(False)
        self._cfg_status_label = QLabel("状态：空闲")
        self._cfg_status_label.setStyleSheet("font-weight: 600;")
        control_row.addWidget(self._cfg_start_btn)
        control_row.addWidget(self._cfg_stop_btn)
        control_row.addSpacing(12)
        control_row.addWidget(self._cfg_status_label)
        control_row.addWidget(self._cfg_progress)
        control_row.addStretch(1)
        root.addLayout(control_row)

        residual_label = QLabel("残差收敛曲线")
        residual_label.setStyleSheet("font-weight: 600; margin-top: 8px;")
        root.addWidget(residual_label)
        self._cfg_residual_figure = Figure(figsize=(8, 2.5), facecolor="#1e1e1e")
        self._cfg_residual_axes = self._cfg_residual_figure.add_subplot(111)
        self._cfg_residual_axes.set_facecolor("#1e1e1e")
        self._cfg_residual_axes.tick_params(colors="#cccccc", labelsize=9)
        self._cfg_residual_axes.spines["bottom"].set_color("#2d2d30")
        self._cfg_residual_axes.spines["top"].set_color("#2d2d30")
        self._cfg_residual_axes.spines["left"].set_color("#2d2d30")
        self._cfg_residual_axes.spines["right"].set_color("#2d2d30")
        self._cfg_residual_axes.set_title("Residuals", color="#cccccc", fontsize=11)
        self._cfg_residual_axes.set_xlabel("Iteration", color="#9d9d9d", fontsize=9)
        self._cfg_residual_axes.set_ylabel("Residual", color="#9d9d9d", fontsize=9)
        self._cfg_residual_axes.set_yscale("log")
        self._cfg_residual_axes.grid(True, alpha=0.2, color="#2d2d30")
        self._cfg_residual_canvas = FigureCanvas(self._cfg_residual_figure)
        self._cfg_residual_canvas.setMinimumHeight(180)
        root.addWidget(self._cfg_residual_canvas)

        log_label = QLabel("实时日志输出")
        log_label.setStyleSheet("font-weight: 600; margin-top: 4px;")
        root.addWidget(log_label)
        self._cfg_run_log = QTextEdit()
        self._cfg_run_log.setReadOnly(True)
        self._cfg_run_log.setStyleSheet("font-family: Consolas, monospace; font-size: 13px;")
        self._cfg_run_log.setPlaceholderText("点击 [启动计算] 后，OpenFOAM 终端输出将实时显示在这里...")
        root.addWidget(self._cfg_run_log, 1)

        return wrapper

    def _sim_config_path(self) -> str:
        from pathlib import Path
        p = Path(__file__).parent.parent.parent.parent / "config" / "sim_config.json"
        return str(p)

    def _save_sim_config_state(self) -> None:
        import json
        data = {
            "solver": self._cfg_solver_combo.currentData(),
            "end_time": self._cfg_end_time.text(),
            "delta_t": self._cfg_delta_t.text(),
            "write_interval": self._cfg_write_interval.value(),
            "init_velocity": self._cfg_init_velocity.text(),
            "init_pressure": self._cfg_init_pressure.text(),
            "material": self._cfg_material_combo.currentData(),
            "density": self._cfg_density.text(),
            "nu": self._cfg_nu.text(),
            "mu": self._cfg_mu.text(),
            "turb_model": self._cfg_turb_model.currentData(),
            "turb_intensity": self._cfg_turb_intensity.text(),
            "turb_length": self._cfg_turb_length.text(),
            "residual": self._cfg_residual.text(),
            "max_iters": self._cfg_max_iters.value(),
            "relaxation": self._cfg_relaxation.text(),
            "fv_schemes": self._cfg_fv_schemes.currentData(),
            "fv_solution": self._cfg_fv_solution.currentData(),
            "boundaries": self._read_boundary_rows(),
        }
        with open(self._sim_config_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_sim_config_state(self) -> None:
        import json
        from pathlib import Path
        path = self._sim_config_path()
        if not Path(path).exists():
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self._cfg_solver_combo.setCurrentIndex(
            max(self._cfg_solver_combo.findData(data.get("solver", "simpleFoam")), 0))
        self._cfg_end_time.setText(data.get("end_time", "1.0"))
        self._cfg_delta_t.setText(data.get("delta_t", "0.001"))
        self._cfg_write_interval.setValue(data.get("write_interval", 100))
        self._cfg_init_velocity.setText(data.get("init_velocity", "(0 0 0)"))
        self._cfg_init_pressure.setText(data.get("init_pressure", "0"))
        self._cfg_material_combo.setCurrentIndex(
            max(self._cfg_material_combo.findData(data.get("material", "air")), 0))
        self._cfg_density.setText(data.get("density", "1.225"))
        self._cfg_nu.setText(data.get("nu", "1.48e-5"))
        self._cfg_mu.setText(data.get("mu", "1.812e-5"))
        self._cfg_turb_model.setCurrentIndex(
            max(self._cfg_turb_model.findData(data.get("turb_model", "laminar")), 0))
        self._cfg_turb_intensity.setText(data.get("turb_intensity", "0.05"))
        self._cfg_turb_length.setText(data.get("turb_length", "0.1"))
        self._cfg_residual.setText(data.get("residual", "1e-6"))
        self._cfg_max_iters.setValue(data.get("max_iters", 1000))
        self._cfg_relaxation.setText(data.get("relaxation", "0.7"))
        self._cfg_fv_schemes.setCurrentIndex(
            max(self._cfg_fv_schemes.findData(data.get("fv_schemes", "stable")), 0))
        self._cfg_fv_solution.setCurrentIndex(
            max(self._cfg_fv_solution.findData(data.get("fv_solution", "default")), 0))
        boundaries = data.get("boundaries", [])
        for r, bc in enumerate(boundaries):
            if r < self._cfg_boundary_table.rowCount():
                if bc.get("name"):
                    self._cfg_boundary_table.item(r, 0).setText(bc["name"])
                if bc.get("role"):
                    self._cfg_boundary_table.item(r, 1).setText(bc["role"])
                if bc.get("u_value"):
                    self._cfg_boundary_table.item(r, 2).setText(bc["u_value"])
                if bc.get("p_value"):
                    self._cfg_boundary_table.item(r, 3).setText(bc["p_value"])

    def _on_material_preset_changed(self) -> None:
        presets = {
            "air": ("1.225", "1.48e-5", "1.812e-5"),
            "water": ("1000", "1e-6", "0.001"),
            "oil": ("880", "5e-5", "0.044"),
        }
        key = self._cfg_material_combo.currentData()
        if key in presets:
            rho, nu, mu = presets[key]
            self._cfg_density.setText(rho)
            self._cfg_nu.setText(nu)
            self._cfg_mu.setText(mu)

    def _calc_mu(self) -> None:
        try:
            rho = float(self._cfg_density.text().strip())
            nu = float(self._cfg_nu.text().strip())
            self._cfg_mu.setText(f"{rho * nu:.6g}")
        except ValueError:
            pass

    def _on_turb_model_changed(self) -> None:
        key = self._cfg_turb_model.currentData()
        if key == "laminar":
            self._cfg_turb_intensity.setEnabled(False)
            self._cfg_turb_length.setEnabled(False)
        else:
            self._cfg_turb_intensity.setEnabled(True)
            self._cfg_turb_length.setEnabled(True)

    def _start_simulation(self) -> None:
        if self._current_project is None:
            self._set_status("请先新建或打开项目")
            return
        if self._foam_process and self._foam_process.state() != QProcess.ProcessState.NotRunning:
            self._set_status("已有任务正在运行")
            return
        status = self._context.environment_detector.detect()
        if not status.is_available or not status.env_script_path:
            self._set_status(f"OpenFOAM 环境不可用：{status.detail}")
            return
        case_dir = self._current_project.case_dir
        solver = self._cfg_solver_combo.currentData()
        command = (
            f"source {shlex.quote(status.env_script_path)} >/dev/null 2>&1 && "
            f"cd {shlex.quote(str(case_dir))} && "
            f"blockMesh && {shlex.quote(solver)}"
        )
        self._cfg_start_btn.setEnabled(False)
        self._cfg_stop_btn.setEnabled(True)
        self._cfg_status_label.setText("状态：计算中")
        self._cfg_progress.setRange(0, 0)
        self._cfg_progress.setVisible(True)
        self._cfg_residual_axes.clear()
        self._cfg_residual_axes.set_facecolor("#1e1e1e")
        self._cfg_residual_axes.set_yscale("log")
        self._cfg_residual_axes.grid(True, alpha=0.2, color="#2d2d30")
        self._cfg_residual_canvas.draw()
        self._cfg_run_log.clear()
        self._cfg_run_log.append(f"<span style='color:#569cd6;'>=== 启动仿真 ===</span>")
        self._cfg_run_log.append(f"Case: {case_dir}")
        self._cfg_run_log.append(f"Solver: {solver}")
        self._cfg_run_log.append("")
        self._residual_data: dict[str, list[float]] = {"Ux": [], "Uy": [], "p": [], "iter": []}

        self._foam_process = QProcess(self)
        self._foam_process.setProgram("bash")
        self._foam_process.setArguments(["-lc", command])
        self._foam_process.readyReadStandardOutput.connect(self._on_sim_stdout)
        self._foam_process.readyReadStandardError.connect(self._on_sim_stderr)
        self._foam_process.finished.connect(self._on_sim_finished)
        self._foam_process.start()
        self._set_status("仿真已启动")

    def _on_sim_stdout(self) -> None:
        if self._foam_process:
            text = bytes(self._foam_process.readAllStandardOutput()).decode(errors="replace")
            self._cfg_run_log.append(text.rstrip())
            self._cfg_run_log.verticalScrollBar().setValue(
                self._cfg_run_log.verticalScrollBar().maximum()
            )
            self._parse_residual_lines(text)

    def _on_sim_stderr(self) -> None:
        if self._foam_process:
            text = bytes(self._foam_process.readAllStandardError()).decode(errors="replace")
            self._cfg_run_log.append(
                f"<span style='color:#f48771;'>{text.rstrip()}</span>"
            )

    def _parse_residual_lines(self, text: str) -> None:
        import re
        for line in text.splitlines():
            m = re.search(r"Solving for Ux.*Initial residual = ([0-9.e+\-]+).*Final residual = ([0-9.e+\-]+)", line)
            if m:
                self._residual_data["Ux"].append(float(m.group(1)))
                self._residual_data["Uy"].append(float(m.group(2)))
                continue
            m = re.search(r"Solving for Uy.*Initial residual = ([0-9.e+\-]+).*Final residual = ([0-9.e+\-]+)", line)
            if m:
                self._residual_data["Ux"].append(self._residual_data["Ux"][-1] if self._residual_data["Ux"] else 0)
                self._residual_data["Uy"].append(float(m.group(1)))
                continue
            m = re.search(r"Solving for p.*Initial residual = ([0-9.e+\-]+).*Final residual = ([0-9.e+\-]+)", line)
            if m:
                self._residual_data["p"].append(float(m.group(1)))
                self._residual_data["iter"].append(len(self._residual_data["p"]) + 1)
        self._redraw_residuals()

    def _redraw_residuals(self) -> None:
        self._cfg_residual_axes.clear()
        self._cfg_residual_axes.set_facecolor("#1e1e1e")
        self._cfg_residual_axes.tick_params(colors="#cccccc", labelsize=9)
        for spine in self._cfg_residual_axes.spines.values():
            spine.set_color("#2d2d30")
        colors = {"Ux": "#569cd6", "Uy": "#6a9955", "p": "#ce9178"}
        for key, color in colors.items():
            data = self._residual_data.get(key, [])
            if data:
                iters = self._residual_data["iter"][:len(data)]
                if iters:
                    self._cfg_residual_axes.plot(iters, data, color=color, linewidth=1.5, label=key)
        self._cfg_residual_axes.set_yscale("log")
        self._cfg_residual_axes.grid(True, alpha=0.2, color="#2d2d30")
        self._cfg_residual_axes.legend(loc="upper right", fontsize=8,
            facecolor="#1e1e1e", edgecolor="#2d2d30", labelcolor="#cccccc")
        self._cfg_residual_canvas.draw()

    def _on_sim_finished(self, exit_code: int, _exit_status) -> None:
        self._cfg_start_btn.setEnabled(True)
        self._cfg_stop_btn.setEnabled(False)
        if exit_code == 0:
            self._cfg_status_label.setText("状态：完成")
            self._cfg_status_label.setStyleSheet("font-weight: 600; color: #89d185;")
            self._cfg_progress.setRange(0, 100)
            self._cfg_progress.setValue(100)
            self._cfg_progress.setVisible(True)
            self._cfg_run_log.append("<span style='color:#89d185;'>=== 仿真完成 ===</span>")
        else:
            self._cfg_status_label.setText(f"状态：异常 (退出码 {exit_code})")
            self._cfg_status_label.setStyleSheet("font-weight: 600; color: #f48771;")
            self._cfg_progress.setVisible(False)
            self._cfg_run_log.append(f"<span style='color:#f48771;'>=== 仿真异常 (退出码 {exit_code}) ===</span>")
        self._set_status(f"仿真结束，退出码 {exit_code}")

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

