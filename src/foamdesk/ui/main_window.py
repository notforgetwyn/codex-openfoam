from __future__ import annotations

import shlex

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtCore import QProcess
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
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
from foamdesk.domain.models import SimulationProject
from foamdesk.ui.theme import THEMES
from foamdesk.ui.visualization_widgets import NativeVtkPreviewWidget, NativeVtkViewerDialog
from foamdesk.ui.main_window_geometry_logic import GeometryLogicMixin
from foamdesk.ui.main_window_results_logic import ResultsLogicMixin
from foamdesk.ui.main_window_project_logic import ProjectProcessLogicMixin
from foamdesk.ui.main_window_settings_physics_logic import SettingsPhysicsLogicMixin
from foamdesk.ui.draw_geometry_tab import DrawGeometryLogicMixin
from foamdesk.ui.sketch_logic import SketchLogicMixin
from foamdesk.ui import domain_templates




class MainWindow(GeometryLogicMixin, ResultsLogicMixin, SettingsPhysicsLogicMixin, ProjectProcessLogicMixin, DrawGeometryLogicMixin, SketchLogicMixin, QMainWindow):
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
        "p",
        "T",
    ]
    RESULT_FIELD_UNITS = {
        "U": "m/s",
        "p": "m2/s2 或 Pa",
        "T": "K",
    }
    RESULT_DISPLAY_MODES = [
        "速度云图",
        "速度切片",
        "流线 streamlines",
        "压力云图",
        "压力等值面",
        "压力切片",
        "壁面压力分布",
        "温度云图",
        "温度切面",
        "壁面温度",
    ]
    RESULT_FIELD_DISPLAY_MODES = {
        "U": [
            "速度云图",
            "速度切片",
            "流线 streamlines",
        ],
        "p": [
            "压力云图",
            "压力等值面",
            "压力切片",
            "壁面压力分布",
        ],
        "T": [
            "温度云图",
            "温度切面",
            "壁面温度",
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
        self._native_vtk_viewer: NativeVtkViewerDialog | None = None
        self.setWindowTitle("FoamDesk")
        self.resize(1400, 900)
        self._init_modeling_state()
        self._init_sketch_state()
        self._build_ui()
        for obj in self._modeling_objects:
            self._add_modeling_object_actors(obj)
            self._apply_transform(obj)
        self._rebuild_tree()
        self._apply_settings_theme()
        self._refresh_status_bar()
        if initial_project is not None:
            self._activate_project(initial_project, "已恢复上次项目。")

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_modeling_state()
        self._save_sketches()
        self._clear_draw_geometry_cache()
        self._save_sim_config_state()
        self._save_solver_run_state()
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
        self._workspace_tabs.currentChanged.connect(self._on_workspace_tab_changed)
        layout.addWidget(self._workspace_tabs)
        return container

    def _on_workspace_tab_changed(self, index: int) -> None:
        if index == self.TAB_DRAW_GEOMETRY:
            self._init_sketch_state()
            self._init_modeling_state()
        elif index == self.TAB_MESH_GENERATION:
            self._init_mesh_import_state()
            self._load_mesh_workflow_state()
        elif index == self.TAB_SIMULATION_CONFIG:
            self._load_sim_config_state()
        elif index == self.TAB_SOLVER_RUN:
            self._load_solver_run_state()
        elif index == self.TAB_RESULTS:
            self._load_results_residual()
            self._refresh_result_field_panel(show_errors=False)

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
        self._mesh_grid_vtk.enable_orientation_axes()
        vtk_layout.addWidget(self._mesh_grid_vtk)

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

        # Group 3 — domain face boundary configuration
        g2 = QGroupBox("计算域面 / 边界配置")
        g2_layout = QVBoxLayout(g2)
        self._domain_face_table = QTableWidget(0, 7)
        self._domain_face_table.setHorizontalHeaderLabels(["显示", "面/区域", "边界类型", "patch名称", "速度 U", "压力 p", "加密"])
        self._domain_face_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._domain_face_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._domain_face_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._domain_face_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._domain_face_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self._domain_face_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self._domain_face_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        self._domain_face_table.verticalHeader().setVisible(False)
        self._domain_face_table.verticalHeader().setDefaultSectionSize(36)
        self._domain_face_table.setMinimumHeight(260)
        g2_layout.addWidget(self._domain_face_table)
        face_btn_row = QHBoxLayout()
        infer_btn = QPushButton("从文件名识别")
        infer_btn.clicked.connect(self._infer_domain_faces_from_filenames)
        highlight_btn = QPushButton("高亮选中面")
        highlight_btn.clicked.connect(self._redraw_mesh_grid_vtk)
        reset_faces_btn = QPushButton("恢复默认边界")
        reset_faces_btn.clicked.connect(self._init_domain_face_table)
        face_btn_row.addWidget(infer_btn)
        face_btn_row.addWidget(highlight_btn)
        face_btn_row.addWidget(reset_faces_btn)
        face_btn_row.addStretch(1)
        g2_layout.addLayout(face_btn_row)
        scroll_layout.addWidget(g2)
        self._init_domain_face_table()

        # Group 2 — domain definition (type + parameters, drives blockMeshDict)
        g3 = QGroupBox("计算域定义")
        g3_layout = QVBoxLayout(g3)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("计算域类型"))
        self._mesh_domain_type_combo = QComboBox()
        for key, spec in domain_templates.DOMAIN_TEMPLATES.items():
            self._mesh_domain_type_combo.addItem(spec["label"], key)
        self._mesh_domain_type_combo.currentIndexChanged.connect(self._on_domain_template_changed)
        type_row.addWidget(self._mesh_domain_type_combo)
        import_domain_btn = QPushButton("导入计算域几何体")
        import_domain_btn.clicked.connect(self._import_cad_domain_files)
        type_row.addWidget(import_domain_btn)
        cad_reimport_btn = QPushButton("重新导入")
        cad_reimport_btn.clicked.connect(self._import_cad_domain_files)
        type_row.addWidget(cad_reimport_btn)
        cad_check_btn = QPushButton("检查封闭性")
        cad_check_btn.clicked.connect(self._check_cad_domain_closure)
        type_row.addWidget(cad_check_btn)
        cad_wrap_btn = QPushButton("自动包围障碍物")
        cad_wrap_btn.clicked.connect(self._auto_wrap_cad_domain_around_obstacles)
        type_row.addWidget(cad_wrap_btn)
        cad_remove_btn = QPushButton("删除选中")
        cad_remove_btn.clicked.connect(self._remove_selected_cad_domain_file)
        type_row.addWidget(cad_remove_btn)
        cad_clear_btn = QPushButton("清空计算域")
        cad_clear_btn.clicked.connect(self._clear_cad_domain_files)
        type_row.addWidget(cad_clear_btn)
        type_row.addStretch(1)
        g3_layout.addLayout(type_row)

        self._domain_param_stack = QStackedWidget()

        # page 0 — bounds (长方体 / 风洞)
        bounds_page = QWidget()
        bp_layout = QVBoxLayout(bounds_page)
        bp_layout.setContentsMargins(0, 0, 0, 0)
        size_label = QLabel("计算域尺寸（坐标范围）")
        size_label.setStyleSheet("font-weight: 600; color: #cccccc;")
        bp_layout.addWidget(size_label)
        bounds_grid = QGridLayout()
        bounds_grid.setSpacing(6)
        axes = [("X 最小", "X 最大"), ("Y 最小", "Y 最大"), ("Z 最小", "Z 最大")]
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
        bp_layout.addLayout(bounds_grid)
        res_label = QLabel("网格分辨率")
        res_label.setStyleSheet("font-weight: 600; color: #cccccc; margin-top: 8px;")
        bp_layout.addWidget(res_label)
        res_row = QHBoxLayout()
        self._domain_cells_inputs: dict[str, QSpinBox] = {}
        for axis in ("X", "Y", "Z"):
            res_row.addWidget(QLabel(f"{axis}方向"))
            spin = QSpinBox()
            spin.setRange(1, 9999)
            spin.setValue(20)
            res_row.addWidget(spin)
            self._domain_cells_inputs[axis] = spin
        res_row.addStretch(1)
        bp_layout.addLayout(res_row)
        self._domain_param_stack.addWidget(bounds_page)

        # page 1 — cylinder (O-grid)
        cyl_page = QWidget()
        cyl_grid = QGridLayout(cyl_page)
        cyl_grid.setContentsMargins(0, 0, 0, 0)
        cyl_grid.setSpacing(6)
        self._cyl_radius = QDoubleSpinBox()
        self._cyl_radius.setRange(0.0001, 1e6)
        self._cyl_radius.setDecimals(4)
        self._cyl_radius.setValue(1.0)
        self._cyl_length = QDoubleSpinBox()
        self._cyl_length.setRange(0.0001, 1e6)
        self._cyl_length.setDecimals(4)
        self._cyl_length.setValue(3.0)
        self._cyl_axis_combo = QComboBox()
        for ax in ("X", "Y", "Z"):
            self._cyl_axis_combo.addItem(f"{ax} 轴", ax)
        self._cyl_axis_combo.setCurrentIndex(2)
        self._cyl_center_x = QDoubleSpinBox(); self._cyl_center_x.setRange(-1e6, 1e6); self._cyl_center_x.setDecimals(4); self._cyl_center_x.setValue(0.0)
        self._cyl_center_y = QDoubleSpinBox(); self._cyl_center_y.setRange(-1e6, 1e6); self._cyl_center_y.setDecimals(4); self._cyl_center_y.setValue(0.0)
        self._cyl_center_z = QDoubleSpinBox(); self._cyl_center_z.setRange(-1e6, 1e6); self._cyl_center_z.setDecimals(4); self._cyl_center_z.setValue(1.5)
        self._cyl_ncirc = QSpinBox(); self._cyl_ncirc.setRange(1, 999); self._cyl_ncirc.setValue(8)
        self._cyl_nradial = QSpinBox(); self._cyl_nradial.setRange(1, 999); self._cyl_nradial.setValue(5)
        self._cyl_naxial = QSpinBox(); self._cyl_naxial.setRange(1, 9999); self._cyl_naxial.setValue(20)
        cyl_grid.addWidget(QLabel("半径 R"), 0, 0); cyl_grid.addWidget(self._cyl_radius, 0, 1)
        cyl_grid.addWidget(QLabel("长度 L"), 0, 2); cyl_grid.addWidget(self._cyl_length, 0, 3)
        cyl_grid.addWidget(QLabel("轴向"), 0, 4); cyl_grid.addWidget(self._cyl_axis_combo, 0, 5)
        cyl_grid.addWidget(QLabel("中心 X"), 1, 0); cyl_grid.addWidget(self._cyl_center_x, 1, 1)
        cyl_grid.addWidget(QLabel("中心 Y"), 1, 2); cyl_grid.addWidget(self._cyl_center_y, 1, 3)
        cyl_grid.addWidget(QLabel("中心 Z"), 1, 4); cyl_grid.addWidget(self._cyl_center_z, 1, 5)
        cyl_grid.addWidget(QLabel("周向网格"), 2, 0); cyl_grid.addWidget(self._cyl_ncirc, 2, 1)
        cyl_grid.addWidget(QLabel("径向网格"), 2, 2); cyl_grid.addWidget(self._cyl_nradial, 2, 3)
        cyl_grid.addWidget(QLabel("轴向网格"), 2, 4); cyl_grid.addWidget(self._cyl_naxial, 2, 5)
        for w in (self._cyl_radius, self._cyl_length, self._cyl_center_x, self._cyl_center_y, self._cyl_center_z,
                  self._cyl_ncirc, self._cyl_nradial, self._cyl_naxial):
            w.valueChanged.connect(self._on_domain_param_changed)
        self._cyl_axis_combo.currentIndexChanged.connect(self._on_domain_param_changed)
        self._domain_param_stack.addWidget(cyl_page)

        # page 2 — nozzle / diffuser (渐变通道)
        nozzle_page = QWidget()
        nz_grid = QGridLayout(nozzle_page)
        nz_grid.setContentsMargins(0, 0, 0, 0)
        nz_grid.setSpacing(6)
        self._nozzle_hin = QDoubleSpinBox(); self._nozzle_hin.setRange(0.0001, 1e6); self._nozzle_hin.setDecimals(4); self._nozzle_hin.setValue(1.0)
        self._nozzle_hout = QDoubleSpinBox(); self._nozzle_hout.setRange(0.0001, 1e6); self._nozzle_hout.setDecimals(4); self._nozzle_hout.setValue(0.4)
        self._nozzle_length = QDoubleSpinBox(); self._nozzle_length.setRange(0.0001, 1e6); self._nozzle_length.setDecimals(4); self._nozzle_length.setValue(3.0)
        self._nozzle_width = QDoubleSpinBox(); self._nozzle_width.setRange(0.0001, 1e6); self._nozzle_width.setDecimals(4); self._nozzle_width.setValue(1.0)
        self._nozzle_center_x = QDoubleSpinBox(); self._nozzle_center_x.setRange(-1e6, 1e6); self._nozzle_center_x.setDecimals(4); self._nozzle_center_x.setValue(1.5)
        self._nozzle_center_y = QDoubleSpinBox(); self._nozzle_center_y.setRange(-1e6, 1e6); self._nozzle_center_y.setDecimals(4); self._nozzle_center_y.setValue(0.0)
        self._nozzle_center_z = QDoubleSpinBox(); self._nozzle_center_z.setRange(-1e6, 1e6); self._nozzle_center_z.setDecimals(4); self._nozzle_center_z.setValue(0.5)
        self._nozzle_cx = QSpinBox(); self._nozzle_cx.setRange(1, 9999); self._nozzle_cx.setValue(30)
        self._nozzle_cy = QSpinBox(); self._nozzle_cy.setRange(1, 9999); self._nozzle_cy.setValue(12)
        self._nozzle_cz = QSpinBox(); self._nozzle_cz.setRange(1, 9999); self._nozzle_cz.setValue(8)
        nz_grid.addWidget(QLabel("进口高度 H_in"), 0, 0); nz_grid.addWidget(self._nozzle_hin, 0, 1)
        nz_grid.addWidget(QLabel("出口高度 H_out"), 0, 2); nz_grid.addWidget(self._nozzle_hout, 0, 3)
        nz_grid.addWidget(QLabel("长度 L (流向X)"), 1, 0); nz_grid.addWidget(self._nozzle_length, 1, 1)
        nz_grid.addWidget(QLabel("宽度 W (Z)"), 1, 2); nz_grid.addWidget(self._nozzle_width, 1, 3)
        nz_grid.addWidget(QLabel("中心 X"), 2, 0); nz_grid.addWidget(self._nozzle_center_x, 2, 1)
        nz_grid.addWidget(QLabel("中心 Y"), 2, 2); nz_grid.addWidget(self._nozzle_center_y, 2, 3)
        nz_grid.addWidget(QLabel("中心 Z"), 2, 4); nz_grid.addWidget(self._nozzle_center_z, 2, 5)
        nz_grid.addWidget(QLabel("X网格"), 3, 0); nz_grid.addWidget(self._nozzle_cx, 3, 1)
        nz_grid.addWidget(QLabel("Y网格"), 3, 2); nz_grid.addWidget(self._nozzle_cy, 3, 3)
        nz_grid.addWidget(QLabel("Z网格"), 3, 4); nz_grid.addWidget(self._nozzle_cz, 3, 5)
        for w in (self._nozzle_hin, self._nozzle_hout, self._nozzle_length, self._nozzle_width,
                  self._nozzle_center_x, self._nozzle_center_y, self._nozzle_center_z,
                  self._nozzle_cx, self._nozzle_cy, self._nozzle_cz):
            w.valueChanged.connect(self._on_domain_param_changed)
        self._domain_param_stack.addWidget(nozzle_page)

        # page 3 — imported CAD domain STL patches
        cad_page = QWidget()
        cad_layout = QVBoxLayout(cad_page)
        cad_layout.setContentsMargins(0, 0, 0, 0)
        cad_layout.setSpacing(8)
        self._cad_domain_file_table = QTableWidget(0, 4)
        self._cad_domain_file_table.setHorizontalHeaderLabels(["序号", "patch名称", "文件名", "路径"])
        self._cad_domain_file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._cad_domain_file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._cad_domain_file_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._cad_domain_file_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._cad_domain_file_table.verticalHeader().setVisible(False)
        self._cad_domain_file_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._cad_domain_file_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._cad_domain_file_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._cad_domain_file_table.setMinimumHeight(110)
        self._cad_domain_file_table.setMaximumHeight(150)
        self._cad_domain_file_table.itemSelectionChanged.connect(self._on_cad_domain_file_selection_changed)
        cad_layout.addWidget(self._cad_domain_file_table)
        transform_grid = QGridLayout()
        transform_grid.setSpacing(6)
        self._cad_domain_transform_inputs = {}
        transform_rows = [
            ("位置", "translate", -1e6, 1e6, 0.0, ""),
            ("旋转", "rotate", -360.0, 360.0, 0.0, "°"),
            ("缩放", "scale", 0.001, 1e6, 1.0, ""),
        ]
        for row, (label, prefix, min_v, max_v, default_v, suffix) in enumerate(transform_rows):
            transform_grid.addWidget(QLabel(label), row, 0)
            for col, axis in enumerate(("X", "Y", "Z")):
                spin = QDoubleSpinBox()
                spin.setRange(min_v, max_v)
                spin.setDecimals(4 if prefix != "rotate" else 2)
                spin.setValue(default_v)
                if suffix:
                    spin.setSuffix(suffix)
                spin.valueChanged.connect(self._on_cad_domain_transform_changed)
                transform_grid.addWidget(QLabel(axis), row, col * 2 + 1)
                transform_grid.addWidget(spin, row, col * 2 + 2)
                self._cad_domain_transform_inputs[f"{prefix}_{axis.lower()}"] = spin
        cad_layout.addLayout(transform_grid)
        cad_cells = QHBoxLayout()
        self._cad_domain_cells_inputs = {}
        for axis, value in (("X", 40), ("Y", 24), ("Z", 24)):
            cad_cells.addWidget(QLabel(f"背景{axis}网格"))
            spin = QSpinBox()
            spin.setRange(1, 9999)
            spin.setValue(value)
            spin.valueChanged.connect(self._on_domain_param_changed)
            cad_cells.addWidget(spin)
            self._cad_domain_cells_inputs[axis] = spin
        cad_cells.addSpacing(16)
        cad_cells.addWidget(QLabel("locationInMesh"))
        self._cad_location_inputs = {}
        for axis in ("X", "Y", "Z"):
            cad_cells.addWidget(QLabel(axis))
            spin = QDoubleSpinBox()
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(5)
            spin.setValue(0.0)
            spin.valueChanged.connect(self._on_cad_location_changed)
            cad_cells.addWidget(spin)
            self._cad_location_inputs[axis.lower()] = spin
        auto_loc_btn = QPushButton("自动推荐位置")
        auto_loc_btn.clicked.connect(self._auto_recommend_location_in_mesh)
        cad_cells.addWidget(auto_loc_btn)
        self._cad_domain_status_label = QLabel("包围状态：未导入计算域")
        self._cad_domain_status_label.setStyleSheet("color: #d7ba7d;")
        cad_cells.addWidget(self._cad_domain_status_label)
        cad_cells.addStretch(1)
        cad_layout.addLayout(cad_cells)
        self._domain_param_stack.addWidget(cad_page)

        g3_layout.addWidget(self._domain_param_stack)

        # Kept as hidden/default state for existing save/load and unit-conversion code.
        self._domain_orthogonal_check = QCheckBox("正交网格")
        self._domain_unit_combo = QComboBox()
        self._domain_unit_combo.addItem("米 (m)", "m")
        self._domain_unit_combo.addItem("毫米 (mm)", "mm")
        # 计算域定义放在边界面表之前：先选区域，再设边界面
        scroll_layout.insertWidget(1, g3)

        # Group 4 — snappyHexMesh refinement + quality constraints
        g4 = QGroupBox("模型贴体网格加密和网格质量约束")
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

        quality_label = QLabel("网格质量约束")
        quality_label.setStyleSheet("font-weight: 600; color: #cccccc; margin-top: 4px;")
        g4_layout.addWidget(quality_label)
        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("最大网格畸变率:"))
        self._quality_max_skew = QDoubleSpinBox()
        self._quality_max_skew.setRange(0.1, 1.0)
        self._quality_max_skew.setDecimals(2)
        self._quality_max_skew.setSingleStep(0.05)
        self._quality_max_skew.setValue(0.8)
        self._quality_max_skew.setToolTip("超过此畸变率的网格将在导出时剔除")
        quality_row.addWidget(self._quality_max_skew)
        quality_row.addWidget(QLabel("最小网格体积:"))
        self._quality_min_volume = QDoubleSpinBox()
        self._quality_min_volume.setRange(0.0, 1.0)
        self._quality_min_volume.setDecimals(8)
        self._quality_min_volume.setValue(1e-12)
        self._quality_min_volume.setToolTip("体积小于此值的网格视为无效")
        quality_row.addWidget(self._quality_min_volume)
        self._quality_del_negative = QCheckBox("自动删除负体积网格")
        self._quality_del_negative.setChecked(True)
        quality_row.addWidget(self._quality_del_negative)
        self._quality_smooth = QCheckBox("网格平滑处理")
        self._quality_smooth.setChecked(True)
        quality_row.addWidget(self._quality_smooth)
        quality_row.addStretch(1)
        g4_layout.addLayout(quality_row)
        scroll_layout.addWidget(g4)

        # Group 6 — action button bar
        btn_row = QHBoxLayout()
        btn_gen = QPushButton("生成网格")
        btn_gen.clicked.connect(self._on_generate_and_execute)
        btn_gen.setStyleSheet("font-weight: bold;")
        btn_preview = QPushButton("预览网格")
        btn_preview.clicked.connect(self._on_preview_mesh)
        btn_check = QPushButton("检查网格质量")
        btn_check.clicked.connect(self._on_check_quality)
        btn_reset = QPushButton("重置所有参数")
        btn_reset.clicked.connect(self._on_reset_all_params)
        for btn in (btn_gen, btn_preview, btn_check, btn_reset):
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

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_content = QWidget()
        scroll.setWidget(scroll_content)
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.addWidget(title)

        # top area: parameter groups in a 2x2 grid
        top = QHBoxLayout()
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        left_col.setSpacing(8)
        right_col.setSpacing(8)

        # === solver basics group ===
        solver_grp = QGroupBox("求解器基础")
        solver_form = QFormLayout(solver_grp)
        self._cfg_flow_type_combo = QComboBox()
        self._cfg_flow_type_combo.addItem("不可压缩（低速/常密度）", "incompressible")
        self._cfg_flow_type_combo.addItem("可压缩（高速/密度变化）", "compressible")
        self._cfg_flow_type_combo.currentIndexChanged.connect(self._on_physics_model_changed)
        self._cfg_time_type_combo = QComboBox()
        self._cfg_time_type_combo.addItem("稳态", "steady")
        self._cfg_time_type_combo.addItem("瞬态", "transient")
        self._cfg_time_type_combo.currentIndexChanged.connect(self._on_physics_model_changed)
        self._cfg_recommended_solver = QLabel("simpleFoam")
        self._cfg_recommended_solver.setStyleSheet("color: #9da5b4;")
        self._cfg_solver_combo = QComboBox()
        self._cfg_solver_combo.currentIndexChanged.connect(self._refresh_dict_preview)
        self._cfg_end_time = QLineEdit("1.0")
        self._cfg_delta_t = QLineEdit("0.001")
        self._cfg_write_interval = QSpinBox()
        self._cfg_write_interval.setRange(1, 1000000)
        self._cfg_write_interval.setValue(100)
        self._cfg_init_velocity = QLineEdit("(0 0 0)")
        self._cfg_init_pressure = QLineEdit("0")
        solver_form.addRow("流体类型", self._cfg_flow_type_combo)
        solver_form.addRow("时间类型", self._cfg_time_type_combo)
        solver_form.addRow("推荐求解器", self._cfg_recommended_solver)
        solver_form.addRow("求解类型", self._cfg_solver_combo)
        solver_form.addRow("总计算时长 (s)", self._cfg_end_time)
        solver_form.addRow("时间步长 (s)", self._cfg_delta_t)
        solver_form.addRow("输出间隔 (步)", self._cfg_write_interval)
        solver_form.addRow("初始速度 (m/s)", self._cfg_init_velocity)
        self._cfg_pressure_label = QLabel("初始压力 p (m2/s2)")
        solver_form.addRow(self._cfg_pressure_label, self._cfg_init_pressure)
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
        self._cfg_temperature = QLineEdit("300")
        self._cfg_cp = QLineEdit("1005")
        self._cfg_gas_r = QLineEdit("287")
        self._cfg_prandtl = QLineEdit("0.7")
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

        self._cfg_compressible_group = QGroupBox("可压缩参数")
        compressible_form = QFormLayout(self._cfg_compressible_group)
        self._cfg_temperature_label = QLabel("初始温度 T (K)")
        self._cfg_cp_label = QLabel("定压比热 Cp")
        self._cfg_gas_r_label = QLabel("气体常数 R")
        self._cfg_prandtl_label = QLabel("Prandtl 数")
        compressible_form.addRow(self._cfg_temperature_label, self._cfg_temperature)
        compressible_form.addRow(self._cfg_cp_label, self._cfg_cp)
        compressible_form.addRow(self._cfg_gas_r_label, self._cfg_gas_r)
        compressible_form.addRow(self._cfg_prandtl_label, self._cfg_prandtl)
        self._cfg_compressible_rows = [
            (self._cfg_temperature_label, self._cfg_temperature),
            (self._cfg_cp_label, self._cfg_cp),
            (self._cfg_gas_r_label, self._cfg_gas_r),
            (self._cfg_prandtl_label, self._cfg_prandtl),
        ]
        left_col.addWidget(self._cfg_compressible_group)

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
        self._rebuild_solver_options()
        self._update_compressible_controls()

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
        self._cfg_fv_schemes.currentIndexChanged.connect(self._refresh_dict_preview)
        self._cfg_fv_solution = QComboBox()
        self._cfg_fv_solution.addItem("默认收敛", "default")
        self._cfg_fv_solution.addItem("严格收敛", "strict")
        self._cfg_fv_solution.addItem("快速粗糙", "fast")
        self._cfg_fv_solution.currentIndexChanged.connect(self._refresh_dict_preview)
        ctrl_form.addRow("残差收敛阈值", self._cfg_residual)
        ctrl_form.addRow("最大迭代步数", self._cfg_max_iters)
        ctrl_form.addRow("松弛因子", self._cfg_relaxation)
        ctrl_form.addRow("数值格式 fvSchemes", self._cfg_fv_schemes)
        ctrl_form.addRow("求解设置 fvSolution", self._cfg_fv_solution)
        right_col.addWidget(ctrl_grp)

        top.addLayout(left_col)
        top.addLayout(right_col)
        scroll_layout.addLayout(top)

        # === bottom: dictionary preview + export ===
        preview_label = QLabel("字典预览")
        preview_label.setStyleSheet("font-weight: 600; margin-top: 8px;")
        scroll_layout.addWidget(preview_label)
        self._cfg_dict_preview = QTextEdit()
        self._cfg_dict_preview.setReadOnly(True)
        self._cfg_dict_preview.setMinimumHeight(150)
        self._cfg_dict_preview.setPlaceholderText("配置参数后点击 [刷新预览] 查看生成的 OpenFOAM 字典文件内容...")
        scroll_layout.addWidget(self._cfg_dict_preview, 1)

        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("刷新预览")
        refresh_btn.clicked.connect(self._refresh_dict_preview)
        export_btn = QPushButton("导出全部仿真字典")
        export_btn.clicked.connect(self._export_sim_dicts)
        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(export_btn)
        btn_row.addStretch(1)
        scroll_layout.addLayout(btn_row)

        root.addWidget(scroll)
        return wrapper

    def _read_boundary_rows(self) -> list[dict]:
        rows: list[dict] = []
        if hasattr(self, "_domain_face_table"):
            merged: dict[str, dict] = {}
            for face in self._get_domain_face_definitions():
                name = face.get("name", "").strip()
                if not name or name in merged:
                    continue
                merged[name] = {
                    "name": name,
                    "role": face.get("type", ""),
                    "u_value": face.get("u_value", "zeroGradient"),
                    "p_value": face.get("p_value", "zeroGradient"),
                }
            for asset in getattr(self, "_mesh_imports", []):
                name = getattr(asset, "name", "").strip()
                if not name or name in merged:
                    continue
                merged[name] = {
                    "name": name,
                    "role": "wall",
                    "u_value": "noSlip",
                    "p_value": "zeroGradient",
                }
            if getattr(self, "_current_domain_template_key", lambda: "")() == "cad" and "background" not in merged:
                merged["background"] = {
                    "name": "background",
                    "role": "patch",
                    "u_value": "zeroGradient",
                    "p_value": "zeroGradient",
                }
            return list(merged.values())
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
                elif val == "empty":
                    blocks.append(f"    {name}\n    {{\n        type            empty;\n    }}")
                else:
                    blocks.append(f"    {name}\n    {{\n        type            fixedValue;\n        value           uniform {val};\n    }}")
            else:  # p
                val = bc["p_value"]
                if val == "zeroGradient":
                    blocks.append(f"    {name}\n    {{\n        type            zeroGradient;\n    }}")
                elif val == "symmetry":
                    blocks.append(f"    {name}\n    {{\n        type            symmetry;\n    }}")
                elif val == "empty":
                    blocks.append(f"    {name}\n    {{\n        type            empty;\n    }}")
                else:
                    blocks.append(f"    {name}\n    {{\n        type            fixedValue;\n        value           uniform {val};\n    }}")
        return "\n".join(blocks)

    def _scalar_boundary_block(self, field: str, inlet_value: str, wall_value: str = "zeroGradient") -> str:
        rows = self._read_boundary_rows()
        if not rows:
            return "    // 无边界"
        blocks = []
        for bc in rows:
            name = bc["name"]
            lowered = name.lower()
            role = str(bc.get("role", "")).lower()
            if bc.get("u_value") in {"symmetry", "empty"}:
                blocks.append(f"    {name}\n    {{\n        type            {bc['u_value']};\n    }}")
            elif "inlet" in lowered or "inlet" in role:
                blocks.append(f"    {name}\n    {{\n        type            fixedValue;\n        value           uniform {inlet_value};\n    }}")
            elif "wall" in lowered or role == "wall":
                if wall_value == "zeroGradient":
                    blocks.append(f"    {name}\n    {{\n        type            zeroGradient;\n    }}")
                else:
                    blocks.append(f"    {name}\n    {{\n        type            fixedValue;\n        value           uniform {wall_value};\n    }}")
            else:
                blocks.append(f"    {name}\n    {{\n        type            zeroGradient;\n    }}")
        return "\n".join(blocks)

    def _vol_scalar_field_text(self, name: str, dimensions: str, internal: str, boundary: str) -> str:
        return (
            self._foam_header("volScalarField", name) +
            f"dimensions      {dimensions};\n"
            f"internalField   uniform {internal};\n"
            f"boundaryField\n{{\n{boundary}\n}}\n"
            f"// ************************************************************************* //\n"
        )

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
        is_compressible = self._cfg_flow_type_combo.currentData() == "compressible"

        u_fmt = init_u.strip()
        if not u_fmt.startswith("(") and " " in u_fmt:
            u_fmt = f"({u_fmt})"
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
            f"internalField   uniform {u_fmt};\n"
            f"boundaryField\n{{\n"
            f"{self._build_boundary_block('U')}\n"
            f"}}\n\n"
            f"// 0/p\n"
            f"dimensions      {'[1 -1 -2 0 0 0 0]' if is_compressible else '[0 2 -2 0 0 0 0]'};\n"
            f"internalField   uniform {init_p};\n"
            f"boundaryField\n{{\n"
            f"{self._build_boundary_block('p')}\n"
            f"}}\n\n"
            + (
                f"// 0/T\n"
                f"dimensions      [0 0 0 1 0 0 0];\n"
                f"internalField   uniform {self._cfg_temperature.text().strip() or '300'};\n"
                f"boundaryField\n{{\n"
                f"{self._scalar_boundary_block('T', self._cfg_temperature.text().strip() or '300')}\n"
                f"}}\n\n"
                if is_compressible else ""
            ) +
            f"// constant/{'thermophysicalProperties' if is_compressible else 'physicalProperties'}\n"
            f"{self._physical_properties_text(rho, nu, mu)}\n\n"
            f"// constant/momentumTransport\n"
            f"{self._momentum_transport_text()}\n\n"
            f"// system/fvSchemes\n{fv_schemes_text}\n\n"
            f"// system/fvSolution\n{fv_solution_text}\n"
        )
        self._cfg_dict_preview.setPlainText(preview)
        self._set_status("字典预览已刷新")

    def _build_fv_schemes_text(self, key: str) -> str:
        solver = self._cfg_solver_combo.currentData() if hasattr(self, "_cfg_solver_combo") else "simpleFoam"
        is_compressible = solver in {"rhoSimpleFoam", "rhoPimpleFoam"}
        if key == "stable":
            div_u = "Gauss upwind"
            ddt = "Euler"
        elif key == "accurate":
            div_u = "Gauss linear"
            ddt = "backward"
        else:
            div_u = "Gauss linearUpwind grad(U)"
            ddt = "Euler"
        if is_compressible:
            div_h = "Gauss upwind" if key == "stable" else "Gauss linearUpwind grad(T)" if key == "balanced" else "Gauss linear"
            return (
                f"ddtSchemes\n{{\n    default         {ddt};\n}}\n\n"
                "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
                "divSchemes\n{\n"
                "    default         none;\n"
                f"    div(phi,U)      {div_u};\n"
                f"    div(phi,K)      {div_u};\n"
                f"    div(phi,h)      {div_h};\n"
                f"    div(phi,e)      {div_h};\n"
                "    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;\n"
                "}\n\n"
                "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n\n"
                "interpolationSchemes\n{\n    default         linear;\n}\n\n"
                "snGradSchemes\n{\n    default         corrected;\n}"
            )
        return (
            f"ddtSchemes\n{{\n    default         {ddt};\n}}\n\n"
            "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
            "divSchemes\n{\n"
            "    default         none;\n"
            f"    div(phi,U)      {div_u};\n"
            "    div((nuEff*dev2(T(grad(U))))) Gauss linear;\n"
            "}\n\n"
            "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n\n"
            "interpolationSchemes\n{\n    default         linear;\n}\n\n"
            "snGradSchemes\n{\n    default         corrected;\n}"
        )

    def _build_fv_solution_text(
        self,
        key: str,
        residual: str,
        relaxation: str,
        solver_override: str | None = None,
    ) -> str:
        ncorrectors = "1" if key == "fast" else "2"
        reltol = "0.1" if key == "fast" else "0.01" if key == "default" else "0.001"
        smoother = "GaussSeidel" if key == "fast" else "symGaussSeidel"
        solver = solver_override or self._cfg_solver_combo.currentData()
        is_simple = solver in {"simpleFoam", "rhoSimpleFoam"}
        is_piso = solver in {"pisoFoam", "icoFoam"}
        is_pimple = solver in {"pimpleFoam", "rhoPimpleFoam"}
        is_compressible = solver in {"rhoSimpleFoam", "rhoPimpleFoam"}
        outer_correctors = "1" if is_simple else ncorrectors
        extra_solvers = ""
        if is_compressible:
            extra_solvers = (
                f"    T\n    {{\n        solver          smoothSolver;\n        smoother        {smoother};\n"
                f"        tolerance       {residual};\n        relTol          {reltol};\n    }}\n"
                f"    TFinal\n    {{\n        $T;\n        relTol          0;\n    }}\n"
                f"    rho\n    {{\n        solver          diagonal;\n    }}\n"
            )

        algorithm_block = ""
        if is_simple:
            temp_control = f"        T               {residual};\n" if is_compressible else ""
            algorithm_block = (
                "SIMPLE\n"
                "{\n"
                "    nNonOrthogonalCorrectors 0;\n"
                "    residualControl\n"
                "    {\n"
                f"        p               {residual};\n"
                f"        U               {residual};\n"
                f"{temp_control}"
                + "    }\n"
                "}\n"
                "\n"
                "PIMPLE\n"
                "{\n"
                "    nOuterCorrectors 1;\n"
                f"    nCorrectors      {ncorrectors};\n"
                "    nNonOrthogonalCorrectors 0;\n"
                "    momentumPredictor yes;\n"
                "    pRefCell         0;\n"
                "    pRefValue        0;\n"
                "    residualControl\n"
                "    {\n"
                f"        p               {residual};\n"
                f"        U               {residual};\n"
                f"{temp_control}"
                "    }\n"
                "}\n"
            )
        elif is_piso:
            algorithm_block = (
                "PISO\n"
                "{\n"
                f"    nCorrectors     {ncorrectors};\n"
                "    nNonOrthogonalCorrectors 0;\n"
                "    pRefCell        0;\n"
                "    pRefValue       0;\n"
                "}\n"
            )
        elif is_pimple:
            algorithm_block = (
                "PIMPLE\n"
                "{\n"
                f"    nOuterCorrectors {outer_correctors};\n"
                f"    nCorrectors      {ncorrectors};\n"
                "    nNonOrthogonalCorrectors 0;\n"
                "    momentumPredictor yes;\n"
                "    pRefCell         0;\n"
                "    pRefValue        0;\n"
                "}\n"
            )

        return (
            f"solvers\n{{\n"
            f"    p\n    {{\n        solver          PCG;\n        preconditioner  DIC;\n"
            f"        tolerance       {residual};\n        relTol          {reltol};\n    }}\n"
            f"    pFinal\n    {{\n        $p;\n        relTol          0;\n    }}\n"
            f"    U\n    {{\n        solver          smoothSolver;\n        smoother        {smoother};\n"
            f"        tolerance       {residual};\n        relTol          {reltol};\n    }}\n"
            f"    UFinal\n    {{\n        $U;\n        relTol          0;\n    }}\n"
            f"{extra_solvers}"
            f"}}\n"
            f"{algorithm_block}"
            f"relaxationFactors\n{{\n"
            f"    fields\n    {{\n        p               0.3;\n    }}\n"
            f"    equations\n    {{\n        U               {relaxation};\n    }}\n"
            f"}}"
        )

    def _foam_header(self, class_name: str, object_name: str) -> str:
        return (
            "/*--------------------------------*- C++ -*----------------------------------*\\\n"
            "| =========                 |                                                 |\n"
            "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"
            "|  \\\\    /   O peration     | Website:  https://openfoam.org                |\n"
            "|   \\\\  /    A nd           | Version:  dev                                   |\n"
            "|    \\\\/     M anipulation  |                                                 |\n"
            "\\*---------------------------------------------------------------------------*/\n"
            f"FoamFile\n{{\n    version     2.0;\n    format      ascii;\n"
            f"    class       {class_name};\n    object      {object_name};\n}}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        )

    def _write_zero_field_files(self, case_dir) -> None:
        (case_dir / "0").mkdir(parents=True, exist_ok=True)
        init_u = self._cfg_init_velocity.text().strip()
        init_p = self._cfg_init_pressure.text().strip()
        is_compressible = self._cfg_flow_type_combo.currentData() == "compressible"
        u_fmt = init_u.strip()
        if not u_fmt.startswith("(") and " " in u_fmt:
            u_fmt = f"({u_fmt})"
        u_field = (
            self._foam_header("volVectorField", "U") +
            f"dimensions      [0 1 -1 0 0 0 0];\n"
            f"internalField   uniform {u_fmt};\n"
            f"boundaryField\n{{\n"
            f"{self._build_boundary_block('U')}\n"
            f"}}\n"
            f"// ************************************************************************* //\n"
        )
        (case_dir / "0" / "U").write_text(u_field, encoding="utf-8")

        p_field = (
            self._foam_header("volScalarField", "p") +
            f"dimensions      {'[1 -1 -2 0 0 0 0]' if is_compressible else '[0 2 -2 0 0 0 0]'};\n"
            f"internalField   uniform {init_p};\n"
            f"boundaryField\n{{\n"
            f"{self._build_boundary_block('p')}\n"
            f"}}\n"
            f"// ************************************************************************* //\n"
        )
        (case_dir / "0" / "p").write_text(p_field, encoding="utf-8")
        if is_compressible:
            temperature = self._cfg_temperature.text().strip() or "300"
            (case_dir / "0" / "T").write_text(
                self._vol_scalar_field_text(
                    "T", "[0 0 0 1 0 0 0]", temperature,
                    self._scalar_boundary_block("T", temperature),
                ),
                encoding="utf-8",
            )
        self._write_turbulence_field_files(case_dir)

    def _write_turbulence_field_files(self, case_dir) -> None:
        zero_dir = case_dir / "0"
        model = self._cfg_turb_model.currentData()
        if model == "laminar":
            return
        intensity = self._cfg_turb_intensity.text().strip() or "0.05"
        length = self._cfg_turb_length.text().strip() or "0.1"
        k_value = "0.01"
        epsilon_value = "0.01"
        omega_value = "1"
        try:
            u_text = self._cfg_init_velocity.text().replace("(", " ").replace(")", " ")
            values = [float(part) for part in u_text.split()[:3]]
            speed = max(float(sum(v * v for v in values) ** 0.5), 1e-9)
            intensity_float = max(float(intensity), 1e-9)
            length_float = max(float(length), 1e-9)
            k_float = 1.5 * (speed * intensity_float) ** 2
            k_value = f"{k_float:.6g}"
            epsilon_value = f"{0.09 ** 0.75 * k_float ** 1.5 / length_float:.6g}"
            omega_value = f"{(k_float ** 0.5) / (0.09 ** 0.25 * length_float):.6g}"
        except (ValueError, ZeroDivisionError):
            pass
        if model in {"kEpsilon", "kOmegaSST"}:
            (zero_dir / "k").write_text(
                self._vol_scalar_field_text("k", "[0 2 -2 0 0 0 0]", k_value, self._scalar_boundary_block("k", k_value, "0")),
                encoding="utf-8",
            )
            (zero_dir / "nut").write_text(
                self._vol_scalar_field_text("nut", "[0 2 -1 0 0 0 0]", "0", self._scalar_boundary_block("nut", "0", "0")),
                encoding="utf-8",
            )
        if model == "kEpsilon":
            (zero_dir / "epsilon").write_text(
                self._vol_scalar_field_text("epsilon", "[0 2 -3 0 0 0 0]", epsilon_value, self._scalar_boundary_block("epsilon", epsilon_value, "0")),
                encoding="utf-8",
            )
        elif model == "kOmegaSST":
            (zero_dir / "omega").write_text(
                self._vol_scalar_field_text("omega", "[0 0 -1 0 0 0 0]", omega_value, self._scalar_boundary_block("omega", omega_value, omega_value)),
                encoding="utf-8",
            )
        elif model == "SpalartAllmaras":
            (zero_dir / "nuTilda").write_text(
                self._vol_scalar_field_text("nuTilda", "[0 2 -1 0 0 0 0]", self._cfg_nu.text().strip() or "1e-5", self._scalar_boundary_block("nuTilda", self._cfg_nu.text().strip() or "1e-5", "0")),
                encoding="utf-8",
            )

    def _physical_properties_text(self, rho: str, nu: str, mu: str) -> str:
        if self._cfg_flow_type_combo.currentData() == "compressible":
            cp = self._cfg_cp.text().strip() or "1005"
            pr = self._cfg_prandtl.text().strip() or "0.7"
            mol_weight = "28.9"
            try:
                gas_r = max(float(self._cfg_gas_r.text().strip() or "287"), 1e-9)
                mol_weight = f"{8314.462618 / gas_r:.6g}"
            except ValueError:
                pass
            return (
                self._foam_header("dictionary", "thermophysicalProperties") +
                "thermoType\n{\n"
                "    type            hePsiThermo;\n"
                "    mixture         pureMixture;\n"
                "    transport       const;\n"
                "    thermo          hConst;\n"
                "    equationOfState perfectGas;\n"
                "    specie          specie;\n"
                "    energy          sensibleEnthalpy;\n"
                "}\n\n"
                "mixture\n{\n"
                "    specie\n    {\n"
                "        nMoles      1;\n"
                f"        molWeight   {mol_weight};\n"
                "    }\n"
                "    thermodynamics\n    {\n"
                f"        Cp          {cp};\n"
                "        Hf          0;\n"
                "    }\n"
                "    transport\n    {\n"
                f"        mu          {mu};\n"
                f"        Pr          {pr};\n"
                "    }\n"
                "}\n"
                "// ************************************************************************* //\n"
            )
        return (
            self._foam_header("dictionary", "physicalProperties") +
            f"viscosityModel  constant;\n"
            f"rho             rho [1 -3 0 0 0 0 0] {rho};\n"
            f"nu              nu [0 2 -1 0 0 0 0] {nu};\n"
            f"// ************************************************************************* //\n"
        )

    def _momentum_transport_text(self) -> str:
        turb_model = self._cfg_turb_model.currentData()
        if turb_model == "laminar":
            turb = "simulationType  laminar;\n"
        else:
            ras_model = "kEpsilon" if turb_model == "kEpsilon" else "kOmegaSST" if turb_model == "kOmegaSST" else turb_model
            turb = (
                f"simulationType  RAS;\n"
                f"RAS\n{{\n"
                f"    model           {ras_model};\n"
                f"    turbulence      on;\n"
                f"    printCoeffs     on;\n"
                f"}}\n"
            )
        return self._foam_header("dictionary", "momentumTransport") + turb + "// ************************************************************************* //\n"

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
        residual = self._cfg_residual.text().strip()
        relaxation = self._cfg_relaxation.text().strip()
        fv_schemes_key = self._cfg_fv_schemes.currentData()
        fv_solution_key = self._cfg_fv_solution.currentData()
        rho = self._cfg_density.text().strip()
        nu = self._cfg_nu.text().strip()
        mu = self._cfg_mu.text().strip()

        control_dict = (
            self._foam_header("dictionary", "controlDict") +
            f"application     {solver};\n"
            f"startFrom       startTime;\n"
            f"startTime       0;\n"
            f"stopAt          endTime;\n"
            f"endTime         {end_time};\n"
            f"deltaT          {delta_t};\n"
            f"writeControl    timeStep;\n"
            f"writeInterval   {write_interval};\n"
            f"// ************************************************************************* //\n"
        )
        (case_dir / "system" / "controlDict").write_text(control_dict, encoding="utf-8")

        self._write_zero_field_files(case_dir)

        (case_dir / "constant").mkdir(parents=True, exist_ok=True)
        if self._cfg_flow_type_combo.currentData() == "compressible":
            (case_dir / "constant" / "thermophysicalProperties").write_text(
                self._physical_properties_text(rho, nu, mu), encoding="utf-8"
            )
            for stale in (case_dir / "constant" / "physicalProperties",):
                if stale.exists():
                    stale.unlink()
        else:
            (case_dir / "constant" / "physicalProperties").write_text(
                self._physical_properties_text(rho, nu, mu), encoding="utf-8"
            )
            for stale in (
                case_dir / "constant" / "thermophysicalProperties",
                case_dir / "0" / "T",
            ):
                if stale.exists():
                    stale.unlink()

        (case_dir / "constant" / "momentumTransport").write_text(
            self._momentum_transport_text(), encoding="utf-8"
        )

        (case_dir / "system" / "fvSchemes").write_text(
            self._foam_header("dictionary", "fvSchemes") +
            self._build_fv_schemes_text(fv_schemes_key) + "\n"
            "// ************************************************************************* //\n",
            encoding="utf-8"
        )
        (case_dir / "system" / "fvSolution").write_text(
            self._foam_header("dictionary", "fvSolution") +
            self._build_fv_solution_text(fv_solution_key, residual, relaxation) + "\n"
            "// ************************************************************************* //\n",
            encoding="utf-8"
        )

        self._set_status(f"字典已导出到 {case_dir}")
        self._workspace_tabs.setCurrentIndex(self.TAB_SOLVER_RUN)





    def _build_sketch_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet("background: #2d2d30;")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        layout.addWidget(QLabel("草图平面"))
        plane_combo = QComboBox()
        for plane in ("XY", "YZ", "XZ"):
            plane_combo.addItem(plane, plane)
        plane_combo.currentTextChanged.connect(self._set_sketch_plane)
        self._sketch_plane_combo = plane_combo
        layout.addWidget(plane_combo)
        layout.addSpacing(10)

        tools = [
            ("select", "选择"), ("point", "点"), ("line", "直线"),
            ("circle", "圆"), ("arc", "圆弧"), ("rect", "矩形"),
            ("polyline", "多段线"),
        ]
        for tool, label in tools:
            btn = QPushButton(label)
            btn.setFixedHeight(28)
            btn.clicked.connect(lambda _checked=False, t=tool: self._set_sketch_tool(t))
            layout.addWidget(btn)
        layout.addSpacing(10)

        self._sketch_closed_label = QLabel("闭合轮廓：❌ 未闭合")
        self._sketch_closed_label.setStyleSheet("color: #d7ba7d;")
        layout.addWidget(self._sketch_closed_label)
        layout.addSpacing(10)

        extrude_btn = QPushButton("拉伸")
        extrude_btn.setFixedHeight(28)
        extrude_btn.clicked.connect(self._extrude_sketch)
        layout.addWidget(extrude_btn)
        revolve_btn = QPushButton("旋转成型")
        revolve_btn.setFixedHeight(28)
        revolve_btn.clicked.connect(self._revolve_sketch)
        layout.addWidget(revolve_btn)
        layout.addSpacing(10)

        layout.addWidget(QLabel("图元"))
        self._sketch_entity_combo = QComboBox()
        self._sketch_entity_combo.setMinimumWidth(110)
        layout.addWidget(self._sketch_entity_combo)
        edit_dim_btn = QPushButton("编辑尺寸")
        edit_dim_btn.setFixedHeight(28)
        edit_dim_btn.clicked.connect(self._edit_selected_sketch_dimension)
        layout.addWidget(edit_dim_btn)
        del_entity_btn = QPushButton("删除图元")
        del_entity_btn.setFixedHeight(28)
        del_entity_btn.clicked.connect(self._delete_selected_sketch_entity)
        layout.addWidget(del_entity_btn)
        clear_btn = QPushButton("清空草图")
        clear_btn.setFixedHeight(28)
        clear_btn.clicked.connect(self._clear_sketch_entities)
        layout.addWidget(clear_btn)

        layout.addStretch(1)
        exit_btn = QPushButton("退出草图")
        exit_btn.setFixedHeight(28)
        exit_btn.setStyleSheet("color: #f48771;")
        exit_btn.clicked.connect(self._exit_sketch_mode)
        layout.addWidget(exit_btn)

        self._sketch_toolbar = bar
        bar.setVisible(False)
        return bar

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
            ("airfoil", "机翼"), ("bend_pipe", "弯管"),
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

        edit_menu = QMenu(self)
        edit_menu.addAction("按点移动", lambda: self._edit_selected_vertices("point"))
        edit_menu.addAction("按线移动", lambda: self._edit_selected_vertices("edge"))
        edit_menu.addAction("按面拉伸", lambda: self._edit_selected_vertices("face"))
        edit_btn = QPushButton("点/线/面编辑")
        edit_btn.setFixedHeight(30)
        edit_btn.setMenu(edit_menu)
        toolbar.addWidget(edit_btn)

        boolean_menu = QMenu(self)
        boolean_menu.addAction("并集", lambda: self._boolean_selected("union"))
        boolean_menu.addAction("差集", lambda: self._boolean_selected("difference"))
        boolean_menu.addAction("交集", lambda: self._boolean_selected("intersection"))
        boolean_btn = QPushButton("布尔运算")
        boolean_btn.setFixedHeight(30)
        boolean_btn.setMenu(boolean_menu)
        toolbar.addWidget(boolean_btn)

        sketch_btn = QPushButton("新建草图")
        sketch_btn.setFixedHeight(30)
        sketch_btn.setStyleSheet("color: #4fc3f7; font-weight: 600;")
        sketch_btn.clicked.connect(self._new_sketch)
        toolbar.addWidget(sketch_btn)

        toolbar.addSpacing(12)

        finish_draw_btn = QPushButton("完成绘制")
        finish_draw_btn.setFixedHeight(30)
        finish_draw_btn.clicked.connect(self._finish_draw_geometry)
        toolbar.addWidget(finish_draw_btn)

        toolbar.addSpacing(12)

        reset_btn = QPushButton("重置视角")
        reset_btn.setFixedHeight(30)
        reset_btn.clicked.connect(self._reset_camera)
        toolbar.addWidget(reset_btn)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        # --- sketch sub-toolbar (hidden until 草图 mode) ---
        root.addWidget(self._build_sketch_toolbar())

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
        self._modeling_viewport.enable_orientation_axes()
        self._install_interactive_edit_handlers()
        vp_layout.addWidget(self._modeling_viewport, 1)
        body.addWidget(viewport_wrapper)

        # property panel
        prop_wrapper = QWidget()
        prop_wrapper.setStyleSheet("background: #252526;")
        prop_wrapper.setMinimumWidth(340)
        prop_outer_layout = QVBoxLayout(prop_wrapper)
        prop_outer_layout.setContentsMargins(12, 12, 12, 12)
        prop_outer_layout.setSpacing(10)
        prop_title = QLabel("属性")
        prop_title.setStyleSheet("font-size: 15px; font-weight: 600;")
        prop_outer_layout.addWidget(prop_title)

        prop_scroll = QScrollArea()
        prop_scroll.setWidgetResizable(True)
        prop_scroll.setFrameShape(QFrame.Shape.NoFrame)
        prop_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        prop_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        prop_content = QWidget()
        prop_content.setStyleSheet("background: #252526;")
        prop_layout = QVBoxLayout(prop_content)
        prop_layout.setContentsMargins(0, 0, 0, 0)
        prop_layout.setSpacing(10)
        prop_scroll.setWidget(prop_content)
        prop_outer_layout.addWidget(prop_scroll, 1)

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
        edit_dx = self._make_modeling_spinbox(-100, 100, 0.01, None)
        edit_dy = self._make_modeling_spinbox(-100, 100, 0.01, None)
        edit_dz = self._make_modeling_spinbox(-100, 100, 0.01, None)
        self._modeling_edit_dx, self._modeling_edit_dy, self._modeling_edit_dz = edit_dx, edit_dy, edit_dz
        self._modeling_apply_edit_delta_btn = QPushButton("应用位移")
        self._modeling_apply_edit_delta_btn.clicked.connect(self._apply_interactive_edit_delta_from_fields)

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
        prop_layout.addWidget(self._modeling_group("三轴编辑位移", edit_dx, edit_dy, edit_dz))
        prop_layout.addWidget(self._modeling_apply_edit_delta_btn)
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

        body.setSizes([200, 720, 380])
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
        sb.setMinimumWidth(80)
        sb.setMinimumHeight(30)
        sb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if callback is not None:
            sb.valueChanged.connect(callback)
        return sb

    def _modeling_group(self, label_text, x, y, z) -> QGroupBox:
        gb = QGroupBox(label_text)
        gb.setMinimumHeight(108)
        form = QFormLayout(gb)
        form.setContentsMargins(10, 14, 10, 10)
        form.setVerticalSpacing(8)
        form.setHorizontalSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignVCenter)
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
        self._cfg_continue_btn = QPushButton("继续计算")
        self._cfg_continue_btn.setFixedHeight(34)
        self._cfg_continue_btn.setEnabled(False)
        self._cfg_continue_btn.clicked.connect(self._continue_simulation)
        self._cfg_stop_btn = QPushButton("暂停计算")
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
        control_row.addWidget(self._cfg_continue_btn)
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
        self._cfg_run_log.setPlaceholderText("点击 [启动计算] 从 0 开始，点击 [继续计算] 从暂停时间步继续。")
        root.addWidget(self._cfg_run_log, 1)

        return wrapper

    def _sim_config_path(self) -> str:
        if hasattr(self, "_current_project") and self._current_project is not None:
            return str(self._current_project.case_dir / "sim_config.json")
        from pathlib import Path
        return str(Path(__file__).parent.parent.parent.parent / "config" / "sim_config.json")

    def _save_sim_config_state(self) -> None:
        import json
        data = {
            "flow_type": self._cfg_flow_type_combo.currentData(),
            "time_type": self._cfg_time_type_combo.currentData(),
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
            "temperature": self._cfg_temperature.text(),
            "cp": self._cfg_cp.text(),
            "gas_r": self._cfg_gas_r.text(),
            "prandtl": self._cfg_prandtl.text(),
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
        import re
        if self._current_project is None:
            return
        case_dir = self._current_project.case_dir
        cd = case_dir / "system" / "controlDict"
        if cd.exists():
            try:
                content = cd.read_text(encoding="utf-8")
                solver_match = re.search(r"application\s+(\S+);", content)
                if solver_match:
                    solver_name = solver_match.group(1)
                    self._cfg_flow_type_combo.setCurrentIndex(1 if solver_name.startswith("rho") else 0)
                    transient = solver_name in {"pimpleFoam", "pisoFoam", "icoFoam", "rhoPimpleFoam"}
                    self._cfg_time_type_combo.setCurrentIndex(1 if transient else 0)
                    self._rebuild_solver_options(solver_name)
                for key, widget, pattern in [
                    ("solver", self._cfg_solver_combo, r"application\s+(\S+);"),
                    ("end_time", self._cfg_end_time, r"endTime\s+([0-9.e+\-]+);"),
                    ("delta_t", self._cfg_delta_t, r"deltaT\s+([0-9.e+\-]+);"),
                    ("write_interval", None, r"writeInterval\s+(\d+);"),
                ]:
                    m = re.search(pattern, content)
                    if m:
                        if key == "solver":
                            idx = self._cfg_solver_combo.findData(m.group(1))
                            if idx >= 0: self._cfg_solver_combo.setCurrentIndex(idx)
                        elif key == "write_interval":
                            self._cfg_write_interval.setValue(int(m.group(1)))
                        elif widget:
                            widget.setText(m.group(1))
            except Exception: pass
        self._update_compressible_controls()
        fs = case_dir / "system" / "fvSchemes"
        if fs.exists():
            try:
                content = fs.read_text(encoding="utf-8")
                key = "balanced" if "linearUpwind" in content else ("accurate" if "backward" in content else "stable")
                idx = self._cfg_fv_schemes.findData(key)
                if idx >= 0: self._cfg_fv_schemes.setCurrentIndex(idx)
            except Exception: pass
        fv = case_dir / "system" / "fvSolution"
        if fv.exists():
            try:
                content = fv.read_text(encoding="utf-8")
                m = re.search(r"tolerance\s+([0-9.e+\-]+);", content)
                if m: self._cfg_residual.setText(m.group(1))
                m = re.search(r"relaxationFactors\s*\{\s*U\s+([0-9.e+\-]+);", content)
                if m: self._cfg_relaxation.setText(m.group(1))
                if "relTol          0" in content:
                    idx = self._cfg_fv_solution.findData("strict")
                    if idx >= 0: self._cfg_fv_solution.setCurrentIndex(idx)
            except Exception: pass
        pp = case_dir / "constant" / "physicalProperties"
        if pp.exists():
            try:
                content = pp.read_text(encoding="utf-8")
                m = re.search(r"nu\s+.*?([0-9.e+\-]+);", content)
                if m: self._cfg_nu.setText(m.group(1))
                m = re.search(r"rho\s+.*?([0-9.e+\-]+);", content)
                if m:
                    self._cfg_density.setText(m.group(1))
                    try:
                        self._cfg_mu.setText(f"{float(m.group(1))*float(self._cfg_nu.text()):.6g}")
                    except ValueError: pass
            except Exception: pass
        thermo = case_dir / "constant" / "thermophysicalProperties"
        if thermo.exists():
            try:
                content = thermo.read_text(encoding="utf-8")
                self._cfg_flow_type_combo.setCurrentIndex(1)
                self._rebuild_solver_options()
                m = re.search(r"Cp\s+([0-9.e+\-]+);", content)
                if m: self._cfg_cp.setText(m.group(1))
                m = re.search(r"mu\s+([0-9.e+\-]+);", content)
                if m: self._cfg_mu.setText(m.group(1))
                m = re.search(r"Pr\s+([0-9.e+\-]+);", content)
                if m: self._cfg_prandtl.setText(m.group(1))
            except Exception: pass
        mt = case_dir / "constant" / "momentumTransport"
        if mt.exists():
            try:
                content = mt.read_text(encoding="utf-8")
                if "simulationType  laminar" in content: self._cfg_turb_model.setCurrentIndex(0)
                elif "kEpsilon" in content:
                    idx = self._cfg_turb_model.findData("kEpsilon")
                    if idx >= 0: self._cfg_turb_model.setCurrentIndex(idx)
                elif "kOmegaSST" in content:
                    idx = self._cfg_turb_model.findData("kOmegaSST")
                    if idx >= 0: self._cfg_turb_model.setCurrentIndex(idx)
            except Exception: pass
        for fn, widget, pattern in [
            ("0/U", self._cfg_init_velocity, r"internalField\s+uniform\s+\(([^)]+)\)"),
            ("0/p", self._cfg_init_pressure, r"internalField\s+uniform\s+([0-9.e+\-]+)"),
            ("0/T", self._cfg_temperature, r"internalField\s+uniform\s+([0-9.e+\-]+)"),
        ]:
            fp = case_dir / fn
            if fp.exists():
                try:
                    m = re.search(pattern, fp.read_text(encoding="utf-8"))
                    if m: widget.setText(m.group(1).strip())
                except Exception: pass
        self._sync_material_preset_to_values()
        self._update_compressible_controls()

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
            self._sync_material_preset_to_values()
        except ValueError:
            pass

    def _sync_material_preset_to_values(self) -> None:
        if not hasattr(self, "_cfg_material_combo"):
            return
        presets = {
            "air": (1.225, 1.48e-5, 1.812e-5),
            "water": (1000.0, 1e-6, 0.001),
            "oil": (880.0, 5e-5, 0.044),
        }
        key = self._cfg_material_combo.currentData()
        if key not in presets:
            return
        try:
            values = (
                float(self._cfg_density.text().strip()),
                float(self._cfg_nu.text().strip()),
                float(self._cfg_mu.text().strip()),
            )
        except ValueError:
            return
        expected = presets[key]
        matched = all(abs(a - b) <= max(abs(b), 1.0) * 1e-6 for a, b in zip(values, expected))
        if not matched:
            index = self._cfg_material_combo.findData("custom")
            if index >= 0:
                self._cfg_material_combo.blockSignals(True)
                self._cfg_material_combo.setCurrentIndex(index)
                self._cfg_material_combo.blockSignals(False)

    def _solver_options_for_model(self) -> list[tuple[str, str]]:
        flow = self._cfg_flow_type_combo.currentData() if hasattr(self, "_cfg_flow_type_combo") else "incompressible"
        time_type = self._cfg_time_type_combo.currentData() if hasattr(self, "_cfg_time_type_combo") else "steady"
        if flow == "compressible":
            return [("rhoSimpleFoam - 稳态可压缩", "rhoSimpleFoam")] if time_type == "steady" else [
                ("rhoPimpleFoam - 瞬态可压缩", "rhoPimpleFoam")
            ]
        if time_type == "steady":
            return [("simpleFoam - 稳态不可压", "simpleFoam")]
        return [
            ("pimpleFoam - 瞬态不可压", "pimpleFoam"),
            ("pisoFoam - 瞬态不可压", "pisoFoam"),
            ("icoFoam - 入门瞬态", "icoFoam"),
        ]

    def _rebuild_solver_options(self, preferred: str | None = None) -> None:
        if not hasattr(self, "_cfg_solver_combo"):
            return
        current = preferred or self._cfg_solver_combo.currentData()
        options = self._solver_options_for_model()
        self._cfg_solver_combo.blockSignals(True)
        self._cfg_solver_combo.clear()
        for label, value in options:
            self._cfg_solver_combo.addItem(label, value)
        index = self._cfg_solver_combo.findData(current)
        if index < 0:
            index = 0
        self._cfg_solver_combo.setCurrentIndex(index)
        self._cfg_solver_combo.blockSignals(False)
        if hasattr(self, "_cfg_recommended_solver"):
            self._cfg_recommended_solver.setText(str(self._cfg_solver_combo.currentData()))

    def _update_compressible_controls(self) -> None:
        if not hasattr(self, "_cfg_temperature"):
            return
        is_compressible = self._cfg_flow_type_combo.currentData() == "compressible"
        if hasattr(self, "_cfg_pressure_label"):
            unit = "Pa" if is_compressible else "m2/s2"
            self._cfg_pressure_label.setText(f"初始压力 p ({unit})")
        if hasattr(self, "_cfg_compressible_group"):
            self._cfg_compressible_group.setVisible(is_compressible)
        for label, widget in getattr(self, "_cfg_compressible_rows", []):
            label.setVisible(is_compressible)
            widget.setVisible(is_compressible)
            widget.setEnabled(is_compressible)

    def _on_physics_model_changed(self) -> None:
        self._rebuild_solver_options()
        self._update_compressible_controls()
        self._refresh_dict_preview()

    def _on_turb_model_changed(self) -> None:
        key = self._cfg_turb_model.currentData()
        if key == "laminar":
            self._cfg_turb_intensity.setEnabled(False)
            self._cfg_turb_length.setEnabled(False)
        else:
            self._cfg_turb_intensity.setEnabled(True)
            self._cfg_turb_length.setEnabled(True)

    def _solver_run_state_path(self):
        from pathlib import Path
        return (self._current_project.case_dir / "solver_run_state.json"
                if self._current_project is not None
                else Path(__file__).parent.parent.parent.parent / "config" / "solver_run_state.json")

    def _latest_case_time_value(self) -> float | None:
        if self._current_project is None:
            return None
        case_dir = self._current_project.case_dir
        if not case_dir.exists():
            return None
        values: list[float] = []
        for path in case_dir.iterdir():
            if path.is_dir() and self._is_openfoam_time_dir(path.name):
                try:
                    values.append(float(path.name))
                except ValueError:
                    pass
        return max(values) if values else None

    def _restore_solver_progress(self, saved: dict) -> None:
        total = int(saved.get("total_steps") or 0)
        current = int(saved.get("current_step") or 0)
        if total > 0:
            self._cfg_progress.setRange(0, total)
            self._cfg_progress.setValue(max(0, min(current, total)))
            self._cfg_progress.setFormat("%v / %m")
            self._cfg_progress.setTextVisible(True)
            self._cfg_progress.setVisible(True)
        else:
            self._cfg_progress.setVisible(False)

    def _load_solver_run_state(self) -> None:
        import json
        path = self._solver_run_state_path()
        saved = {}
        if path.exists():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        if self._current_project is None:
            if saved.get("log"):
                self._cfg_run_log.setHtml(saved["log"])
            return
        cd = self._current_project.case_dir / "system" / "controlDict"
        has_mesh = (self._current_project.case_dir / "constant" / "polyMesh").exists()
        has_dicts = cd.exists()
        if has_mesh and has_dicts:
            self._cfg_status_label.setText(saved.get("status_text", "状态：就绪"))
            self._cfg_status_label.setStyleSheet("font-weight: 600; color: #89d185;")
            self._cfg_run_log.setPlaceholderText("网格和字典已就绪，点击 [启动计算] 从 0 开始仿真。")
            self._cfg_start_btn.setEnabled(True)
            self._cfg_continue_btn.setEnabled(bool(saved.get("can_continue")))
        elif has_dicts:
            self._cfg_status_label.setText("状态：缺少网格")
            self._cfg_status_label.setStyleSheet("font-weight: 600; color: #d7ba7d;")
            self._cfg_run_log.setPlaceholderText("字典已就绪，请先在网格生成页运行 blockMesh。")
            self._cfg_continue_btn.setEnabled(False)
        else:
            self._cfg_status_label.setText("状态：空闲")
            self._cfg_status_label.setStyleSheet("font-weight: 600;")
            self._cfg_run_log.setPlaceholderText("请先在仿真参数配置页导出字典。")
            self._cfg_continue_btn.setEnabled(False)
        if saved.get("log"):
            self._cfg_run_log.setHtml(saved["log"])
        self._sim_latest_time = saved.get("latest_time")
        self._sim_current_step = int(saved.get("current_step") or 0)
        self._sim_total_steps = int(saved.get("total_steps") or 0)
        self._sim_delta_t = float(saved.get("delta_t") or 0.0)
        self._sim_end_time = float(saved.get("end_time") or 0.0)
        self._restore_solver_progress(saved)

    def _save_solver_run_state(self) -> None:
        import json
        path = self._solver_run_state_path()
        data = {
            "status_text": self._cfg_status_label.text(),
            "log": self._cfg_run_log.toHtml() if hasattr(self, "_cfg_run_log") else "",
            "latest_time": getattr(self, "_sim_latest_time", None),
            "current_step": int(getattr(self, "_sim_current_step", 0) or 0),
            "total_steps": int(getattr(self, "_sim_total_steps", 0) or 0),
            "delta_t": float(getattr(self, "_sim_delta_t", 0.0) or 0.0),
            "end_time": float(getattr(self, "_sim_end_time", 0.0) or 0.0),
            "can_continue": bool(getattr(self, "_sim_can_continue", False)),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _set_control_dict_start_from(self, ctrl_dict, start_from: str) -> None:
        import re
        content = ctrl_dict.read_text(encoding="utf-8")
        if re.search(r"\bstartFrom\s+\S+;", content):
            content = re.sub(r"\bstartFrom\s+\S+;", f"startFrom       {start_from};", content, count=1)
        else:
            content = content.replace("application", f"startFrom       {start_from};\napplication", 1)
        ctrl_dict.write_text(content, encoding="utf-8")

    def _clear_solver_time_dirs_for_fresh_start(self, case_dir) -> None:
        root = case_dir.resolve()
        for child in case_dir.iterdir():
            if not child.is_dir() or not self._is_openfoam_time_dir(child.name) or child.name == "0":
                continue
            target = child.resolve()
            if target == root or root not in target.parents:
                raise OSError(f"拒绝删除 Case 外路径：{target}")
            import shutil
            shutil.rmtree(child)
        for generated in (case_dir / "postProcessing", case_dir / "VTK"):
            if generated.exists():
                target = generated.resolve()
                if target == root or root not in target.parents:
                    raise OSError(f"拒绝删除 Case 外路径：{target}")
                import shutil
                shutil.rmtree(generated)

    def _continue_simulation(self) -> None:
        self._start_simulation(resume=True)

    def _solver_runtime_command(self, solver: str) -> str:
        if solver == "simpleFoam":
            return "foamRun -solver incompressibleFluid"
        if solver == "rhoSimpleFoam":
            return "foamRun -solver compressibleFluid"
        return shlex.quote(solver)

    def _ensure_fv_solution_matches_solver(self, case_dir, solver: str) -> None:
        if solver not in {"simpleFoam", "rhoSimpleFoam", "pimpleFoam", "rhoPimpleFoam"}:
            return
        fv_solution = case_dir / "system" / "fvSolution"
        if fv_solution.exists() and "PIMPLE" in fv_solution.read_text(encoding="utf-8", errors="replace"):
            return
        key = self._cfg_fv_solution.currentData() if hasattr(self, "_cfg_fv_solution") else "default"
        residual = self._cfg_residual.text().strip() if hasattr(self, "_cfg_residual") else "1e-6"
        relaxation = self._cfg_relaxation.text().strip() if hasattr(self, "_cfg_relaxation") else "0.7"
        fv_solution.write_text(
            self._foam_header("dictionary", "fvSolution")
            + self._build_fv_solution_text(key, residual, relaxation, solver_override=solver)
            + "\n// ************************************************************************* //\n",
            encoding="utf-8",
        )

    def _start_simulation(self, resume: bool = False) -> None:
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
        import re
        case_dir = self._current_project.case_dir
        ctrl_dict = case_dir / "system" / "controlDict"
        if not ctrl_dict.exists():
            self._set_status("请先在仿真参数配置页导出字典")
            return
        if not (case_dir / "constant" / "polyMesh").exists():
            self._set_status("未生成网格，请先在网格生成页运行 blockMesh")
            return
        if resume:
            latest_time = self._latest_case_time_value()
            if latest_time is None or latest_time <= 0:
                self._set_status("没有可继续的时间步，请先启动一次仿真并写出结果。")
                return
            self._set_control_dict_start_from(ctrl_dict, "latestTime")
        else:
            try:
                self._clear_solver_time_dirs_for_fresh_start(case_dir)
            except OSError as error:
                self._set_status(f"清理旧仿真时间步失败：{error}")
                return
            self._set_control_dict_start_from(ctrl_dict, "startTime")
        content = ctrl_dict.read_text(encoding="utf-8")
        m = re.search(r"application\s+(\S+);", content)
        solver = m.group(1) if m else "simpleFoam"
        self._ensure_fv_solution_matches_solver(case_dir, solver)
        end_m = re.search(r"endTime\s+([0-9.e+\-]+);", content)
        dt_m = re.search(r"deltaT\s+([0-9.e+\-]+);", content)
        try:
            end_time_val = float(end_m.group(1)) if end_m else 1.0
            dt_val = float(dt_m.group(1)) if dt_m else 0.001
            self._sim_total_steps = int(end_time_val / dt_val)
            self._sim_delta_t = dt_val
            self._sim_end_time = end_time_val
        except (ValueError, ZeroDivisionError):
            self._sim_total_steps = 0
            self._sim_delta_t = 0.0
            self._sim_end_time = 0.0
        resume_time = self._latest_case_time_value() if resume else 0.0
        if resume and self._sim_delta_t:
            self._sim_current_step = int(round((resume_time or 0.0) / self._sim_delta_t))
        else:
            self._sim_current_step = 0
        self._sim_latest_time = resume_time
        self._sim_can_continue = False
        self._sim_pause_requested = False
        self._sim_finish_handled = False
        command = (
            f"source {shlex.quote(status.env_script_path)} >/dev/null 2>&1 && "
            f"cd {shlex.quote(str(case_dir))} && "
            f"{self._solver_runtime_command(solver)}"
        )
        self._cfg_start_btn.setEnabled(False)
        self._cfg_continue_btn.setEnabled(False)
        self._cfg_stop_btn.setEnabled(True)
        self._cfg_status_label.setText("状态：计算中")
        if self._sim_total_steps > 0:
            self._cfg_progress.setRange(0, self._sim_total_steps)
            self._cfg_progress.setValue(max(0, min(self._sim_current_step, self._sim_total_steps)))
            self._cfg_progress.setFormat("%v / %m")
            self._cfg_progress.setTextVisible(True)
        else:
            self._cfg_progress.setRange(0, 0)
        self._cfg_progress.setVisible(True)
        self._cfg_residual_axes.clear()
        self._cfg_residual_axes.set_facecolor("#1e1e1e")
        self._cfg_residual_axes.set_yscale("log")
        self._cfg_residual_axes.grid(True, alpha=0.2, color="#2d2d30")
        self._cfg_residual_canvas.draw()
        if not resume:
            self._cfg_run_log.clear()
        self._cfg_run_log.append(f"<span style='color:#569cd6;'>=== {'继续' if resume else '启动'}仿真 ===</span>")
        self._cfg_run_log.append(f"Case: {case_dir}")
        self._cfg_run_log.append(f"Solver: {solver}")
        self._cfg_run_log.append("")
        self._residual_data: dict[str, list[float]] = {"iter": []}

        self._foam_process = QProcess(self)
        self._active_process_kind = "simulation"
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
        changed = False
        for line in text.splitlines():
            tm = re.search(r"^Time\s*=\s*([0-9.e+\-]+)", line)
            if tm:
                try:
                    self._sim_latest_time = float(tm.group(1))
                except ValueError:
                    self._sim_latest_time = None
                if getattr(self, "_sim_delta_t", 0.0):
                    self._sim_current_step = int(round((self._sim_latest_time or 0.0) / self._sim_delta_t))
                else:
                    self._sim_current_step += 1
                if self._sim_total_steps > 0:
                    self._cfg_progress.setValue(min(self._sim_current_step, self._sim_total_steps))
                    self._cfg_progress.setFormat("%v / %m")
                    self._cfg_progress.setTextVisible(True)
            m = re.search(
                r"Solving for\s+([^,\s]+).*?Initial residual\s*=\s*([0-9.e+\-]+)"
                r"(?:,\s*Final residual\s*=\s*([0-9.e+\-]+))?",
                line,
            )
            if m:
                field = m.group(1).rstrip(",")
                try:
                    value = float(m.group(2))
                except ValueError:
                    continue
                self._residual_data.setdefault(field, []).append(value)
                max_len = max((len(v) for k, v in self._residual_data.items() if k != "iter"), default=0)
                self._residual_data["iter"] = list(range(1, max_len + 1))
                changed = True
        if changed:
            self._redraw_residuals()
        self._save_solver_run_state()

    def _redraw_residuals(self) -> None:
        self._cfg_residual_axes.clear()
        self._cfg_residual_axes.set_facecolor("#1e1e1e")
        self._cfg_residual_axes.tick_params(colors="#cccccc", labelsize=9)
        for spine in self._cfg_residual_axes.spines.values():
            spine.set_color("#2d2d30")
        palette = ["#569cd6", "#6a9955", "#dcdcaa", "#ce9178", "#c586c0", "#4ec9b0", "#d7ba7d", "#9cdcfe"]
        fields = [k for k in self._residual_data.keys() if k != "iter" and self._residual_data.get(k)]
        order = ["Ux", "Uy", "Uz", "U", "p", "p_rgh", "T", "h", "e", "rho"]
        fields = [k for k in order if k in fields] + [k for k in fields if k not in order]
        for index, key in enumerate(fields):
            data = self._residual_data.get(key, [])
            if data:
                iters = list(range(1, len(data) + 1))
                self._cfg_residual_axes.plot(iters, data, color=palette[index % len(palette)], linewidth=1.5, label=key)
        self._cfg_residual_axes.set_yscale("log")
        self._cfg_residual_axes.grid(True, alpha=0.2, color="#2d2d30")
        if any(data for data in self._residual_data.values() if isinstance(data, list) and data):
            self._cfg_residual_axes.legend(loc="upper right", fontsize=8,
                facecolor="#1e1e1e", edgecolor="#2d2d30", labelcolor="#cccccc")
        self._cfg_residual_canvas.draw()
        self._save_residual_data()

    def _save_residual_data(self) -> None:
        if self._current_project is None:
            return
        import json
        save = {k: v for k, v in self._residual_data.items() if v}
        if not save:
            return
        (self._current_project.case_dir / "residual_data.json").write_text(
            json.dumps(save), encoding="utf-8")

    def _on_sim_finished(self, exit_code: int, _exit_status) -> None:
        if getattr(self, "_sim_finish_handled", False):
            return
        if getattr(self, "_sim_pause_requested", False):
            self._mark_simulation_paused()
            return
        self._active_process_kind = "idle"
        self._cfg_start_btn.setEnabled(True)
        self._cfg_stop_btn.setEnabled(False)
        if exit_code == 0:
            self._cfg_status_label.setText("状态：完成")
            self._cfg_status_label.setStyleSheet("font-weight: 600; color: #89d185;")
            if self._sim_total_steps > 0:
                self._cfg_progress.setValue(min(self._sim_current_step, self._sim_total_steps))
                self._cfg_progress.setVisible(True)
            self._cfg_run_log.append("<span style='color:#89d185;'>=== 仿真完成 ===</span>")
            self._sim_can_continue = False
            self._cfg_continue_btn.setEnabled(False)
            self._run_foam_to_vtk()
        else:
            self._cfg_status_label.setText(f"状态：异常 (退出码 {exit_code})")
            self._cfg_status_label.setStyleSheet("font-weight: 600; color: #f48771;")
            self._cfg_run_log.append(f"<span style='color:#f48771;'>=== 仿真异常 (退出码 {exit_code}) ===</span>")
        self._set_status(f"仿真结束，退出码 {exit_code}")
        self._save_residual_data()
        self._save_solver_run_state()

    def _mark_simulation_paused(self) -> None:
        self._sim_finish_handled = True
        self._active_process_kind = "idle"
        latest_time = self._latest_case_time_value()
        if latest_time is not None:
            self._sim_latest_time = latest_time
            if getattr(self, "_sim_delta_t", 0.0):
                self._sim_current_step = int(round(latest_time / self._sim_delta_t))
                if self._sim_total_steps > 0:
                    self._cfg_progress.setValue(min(self._sim_current_step, self._sim_total_steps))
        self._sim_can_continue = bool((self._sim_latest_time or 0) > 0)
        self._sim_pause_requested = False
        self._cfg_start_btn.setEnabled(True)
        self._cfg_continue_btn.setEnabled(self._sim_can_continue)
        self._cfg_stop_btn.setEnabled(False)
        self._cfg_status_label.setText("状态：已暂停")
        self._cfg_status_label.setStyleSheet("font-weight: 600; color: #d7ba7d;")
        self._cfg_run_log.append("<span style='color:#d7ba7d;'>=== 仿真已暂停，可从最新写出的时间步继续 ===</span>")
        self._set_status("仿真已暂停。")
        self._save_solver_run_state()

    def _run_foam_to_vtk(self) -> None:
        """Run foamToVTK -latestTime to export VTK files into case/VTK/."""
        if self._current_project is None:
            return
        status = self._context.environment_detector.detect()
        env_script = status.env_script_path or ""
        case_dir = self._current_project.case_dir
        if env_script:
            cmd = (
                f'source {shlex.quote(env_script)} >/dev/null 2>&1 && '
                f'cd {shlex.quote(str(case_dir))} && '
                f'foamToVTK -latestTime'
            )
        else:
            cmd = (
                f'cd {shlex.quote(str(case_dir))} && '
                f'foamToVTK -latestTime'
            )
        self._vtk_export_process = QProcess(self)
        self._vtk_export_process.setProgram("bash")
        self._vtk_export_process.setArguments(["-lc", cmd])
        self._vtk_export_process.readyReadStandardOutput.connect(self._read_vtk_export_stdout)
        self._vtk_export_process.finished.connect(self._on_vtk_export_finished)
        self._vtk_export_process.start()
        self._cfg_run_log.append("<span style='color:#569cd6;'>正在生成 VTK 文件...</span>")
        self._append_log("正在执行 foamToVTK -latestTime ...")
        self._set_status("正在生成 VTK 文件...")

    def _read_vtk_export_stdout(self) -> None:
        if self._vtk_export_process is None:
            return
        data = self._vtk_export_process.readAllStandardOutput()
        text = bytes(data).decode("utf-8", errors="replace")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                self._cfg_run_log.append(stripped)
                self._append_log(stripped)

    def _on_vtk_export_finished(self, exit_code: int) -> None:
        if self._current_project is None:
            return
        vtk_dir = self._current_project.case_dir / "VTK"
        if exit_code == 0 and vtk_dir.exists():
            msg = f"VTK 文件已生成：{vtk_dir}"
            self._cfg_run_log.append(f"<span style='color:#89d185;'>{msg}</span>")
            self._append_log(msg)
            vol_files = list((vtk_dir / 'volumes').glob('*.vtu')) if (vtk_dir / 'volumes').exists() else []
            surf_files = list((vtk_dir / 'surfaces').glob('*.vtp')) if (vtk_dir / 'surfaces').exists() else []
            self._append_log(f"  volumes: {[f.name for f in vol_files]}")
            self._append_log(f"  surfaces: {[f.name for f in surf_files]}")
            self._set_status("VTK 文件生成完成。")
        else:
            self._cfg_run_log.append(
                f"<span style='color:#f48771;'>VTK 导出失败 (exit={exit_code})</span>"
            )
            self._append_log(f"foamToVTK 失败，退出码 {exit_code}")
            self._set_status(f"VTK 导出失败 (exit={exit_code})")

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
        description = QLabel("选择结果场和显示方式，查看仿真后处理结果。")
        description.setWordWrap(True)

        self._results_residual_figure = Figure(figsize=(8, 3.0), facecolor="#1e1e1e")
        self._results_residual_axes = self._results_residual_figure.add_subplot(111)
        self._results_residual_axes.set_facecolor("#1e1e1e")
        self._results_residual_axes.tick_params(colors="#cccccc", labelsize=9)
        for spine in self._results_residual_axes.spines.values():
            spine.set_color("#2d2d30")
        self._results_residual_canvas = FigureCanvas(self._results_residual_figure)
        self._results_residual_canvas.setMaximumHeight(180)
        self._load_results_residual()

        residual_label = QLabel("残差曲线")
        residual_label.setStyleSheet("font-weight: 600; margin-top: 4px;")

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
        self._result_unit_label = QLabel("单位：-")
        self._result_minmax_label = QLabel("最大/最小值：未刷新")
        refresh_fields_button = QPushButton("刷新字段/时间步")
        refresh_fields_button.clicked.connect(lambda _checked=False: self._refresh_result_field_panel())
        self._result_field_combo.currentTextChanged.connect(lambda _text: self._on_result_field_changed())
        field_row.addWidget(QLabel("变量"))
        field_row.addWidget(self._result_field_combo, 2)
        field_row.addWidget(QLabel("时间步"))
        field_row.addWidget(self._result_time_combo, 2)
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
        load_display_button = QPushButton("加载显示")
        load_display_button.clicked.connect(lambda _checked=False: self._load_selected_result_display())
        display_row.addWidget(QLabel("显示方式"))
        display_row.addWidget(self._result_display_combo, 2)
        display_row.addWidget(load_display_button)
        display_hint = QLabel(
            "只展示速度、压强、温度三类结果：云图、切片/切面、壁面分布、压力等值面和流线。"
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

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(residual_label)
        layout.addWidget(self._results_residual_canvas)
        layout.addWidget(field_group)
        layout.addWidget(display_group)
        layout.addWidget(self._results_text, 1)
        return wrapper

    def _load_results_residual(self) -> None:
        import json
        self._results_residual_axes.clear()
        self._results_residual_axes.set_facecolor("#1e1e1e")
        self._results_residual_axes.tick_params(colors="#cccccc", labelsize=9)
        for spine in self._results_residual_axes.spines.values():
            spine.set_color("#2d2d30")
        if self._current_project is None:
            self._results_residual_canvas.draw(); return
        rp = self._current_project.case_dir / "residual_data.json"
        if not rp.exists():
            self._results_residual_canvas.draw(); return
        try:
            data = json.loads(rp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        palette = ["#569cd6", "#6a9955", "#dcdcaa", "#ce9178", "#c586c0", "#4ec9b0", "#d7ba7d", "#9cdcfe"]
        fields = [k for k, v in data.items() if k != "iter" and v]
        order = ["Ux", "Uy", "Uz", "U", "p", "p_rgh", "T", "h", "e", "rho"]
        fields = [k for k in order if k in fields] + [k for k in fields if k not in order]
        for index, key in enumerate(fields):
            vals = data.get(key, [])
            if vals:
                self._results_residual_axes.plot(
                    list(range(1, len(vals) + 1)),
                    vals,
                    color=palette[index % len(palette)],
                    linewidth=1.5,
                    label=key,
                )
        self._results_residual_axes.set_yscale("log")
        self._results_residual_axes.grid(True, alpha=0.2, color="#2d2d30")
        if fields:
            self._results_residual_axes.legend(loc="upper right", fontsize=8,
                facecolor="#1e1e1e", edgecolor="#2d2d30", labelcolor="#cccccc")
        self._results_residual_canvas.draw()

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
