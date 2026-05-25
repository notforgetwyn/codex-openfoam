from __future__ import annotations

import json
import shlex
import re
from pathlib import Path
from typing import Callable

import vtk
import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QPoint, Qt
from PySide6.QtCore import QProcess, QTimer
from PySide6.QtGui import QFont
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkCommonMath import vtkRungeKutta4
from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PySide6.QtWidgets import (
    QComboBox,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFontComboBox,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QApplication,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
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
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from foamdesk.app.bootstrap import ApplicationContext
from foamdesk.domain.models import SimulationParameters, SimulationProject
from foamdesk.services.geometry_import_service import GeometryAsset, SnappyHexMeshSettings, StlTransform
from foamdesk.services.project_service import BoundaryConditionSettings, ComputationDomainTemplate
from foamdesk.ui.startup_window import StartupWindow
from foamdesk.ui.theme import THEMES, build_stylesheet
from foamdesk.ui.visualization_widgets import NativeVtkPreviewWidget, NativeVtkViewerDialog, VtkViewerDialog
from foamdesk.ui.main_window_geometry_logic import GeometryLogicMixin
from foamdesk.ui.main_window_results_logic import ResultsLogicMixin
from foamdesk.ui.main_window_parameters_logic import ParametersLogicMixin
from foamdesk.ui.main_window_project_logic import ProjectProcessLogicMixin
from foamdesk.ui.main_window_settings_physics_logic import SettingsPhysicsLogicMixin


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


class MainWindow(GeometryLogicMixin, ResultsLogicMixin, ParametersLogicMixin, SettingsPhysicsLogicMixin, ProjectProcessLogicMixin, QMainWindow):
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
        "epsilon",
        "omega",
        "nut",
        "yPlus",
        "wallShearStress",
        "vorticity",
        "Q",
        "alpha.water",
    ]
    RESULT_FIELD_UNITS = {
        "U": "m/s",
        "mag(U)": "m/s",
        "p": "m2/s2 或 Pa",
        "T": "K",
        "k": "m2/s2",
        "epsilon": "m2/s3",
        "omega": "1/s",
        "nut": "m2/s",
        "yPlus": "无量纲",
        "wallShearStress": "Pa 或 m2/s2",
        "vorticity": "1/s",
        "Q": "1/s2",
        "alpha.water": "0-1",
    }
    RESULT_DISPLAY_MODES = [
        "Surface 表面云图",
        "Slice 切片",
        "Contour 等值线",
        "Iso-surface 等值面",
        "Vector 矢量箭头",
        "Streamline 流线",
        "Glyph 箭头",
        "Volume rendering 体渲染",
    ]

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
        self._build_ui()
        self._apply_settings_theme()
        self._refresh_status_bar()
        if initial_project is not None:
            self._activate_project(initial_project, "已恢复上次项目。")

    def closeEvent(self, event) -> None:  # noqa: N802
        self._persist_draw_geometry_state()
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
        root_layout = QVBoxLayout(wrapper)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("绘制几何")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        desc = QLabel("可视化编辑 blockMesh。计算域 + 几何体共用同一套顶点/边编辑。")
        desc.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(desc)

        self._geo_objects = self._default_draw_geometry_objects()
        self._active_obj_idx = 0

        obj_row = QHBoxLayout()
        obj_row.setSpacing(8)
        self._obj_combo = QComboBox()
        self._obj_combo.currentIndexChanged.connect(self._switch_active_object)
        new_btn = QPushButton("+ 新建几何体")
        new_btn.clicked.connect(self._new_geo_object)
        del_btn = QPushButton("- 删除当前")
        del_btn.clicked.connect(self._delete_geo_object)
        obj_row.addWidget(QLabel("编辑对象"))
        obj_row.addWidget(self._obj_combo, 1)
        obj_row.addWidget(new_btn)
        obj_row.addWidget(del_btn)
        layout.addLayout(obj_row)
        self._rebuild_obj_combo()

        vh = QHBoxLayout()
        vh.addWidget(QLabel("顶点列表"))
        add_v = QPushButton("+ 添加")
        csv_v = QPushButton("导入CSV")
        csv_v.clicked.connect(self._import_vertices_csv)
        del_v = QPushButton("- 删除")
        vh.addStretch(1); vh.addWidget(add_v); clear_v = QPushButton("清空"); clear_v.clicked.connect(self._clear_all_vertices); vh.addWidget(del_v); vh.addWidget(clear_v); vh.addWidget(csv_v)
        layout.addLayout(vh)

        self._vertex_table = QTableWidget(0, 3)
        self._vertex_table.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self._vertex_table.horizontalHeader().setStretchLastSection(True)
        self._vertex_table.setMinimumHeight(180)
        self._vertex_table.cellChanged.connect(self._on_vertex_table_changed)
        add_v.clicked.connect(lambda: (self._add_vertex_row(0,0,0), self._save_vertex_table(), self._refresh_draw_geo_preview()))
        del_v.clicked.connect(self._delete_selected_vertex)
        layout.addWidget(self._vertex_table)

        elab = QLabel("边 (Edges)")
        elab.setStyleSheet("font-weight: 600; margin-top: 8px;")
        layout.addWidget(elab)
        ef = QHBoxLayout(); ef.setSpacing(6)
        self._edge_type_combo = QComboBox()
        self._edge_type_combo.addItems(["arc","spline","polyLine","BSpline"])
        self._edge_start = QSpinBox(); self._edge_start.setRange(0,99); self._edge_start.setValue(0)
        self._edge_end = QSpinBox(); self._edge_end.setRange(0,99); self._edge_end.setValue(1)
        self._edge_interp = QLineEdit("0.5 0.5 0.5")
        self._edge_interp.setPlaceholderText("x y z")
        self._edge_interp.setMinimumWidth(160)
        ae = QPushButton("添加边")
        ae.clicked.connect(self._add_edge_to_current)
        ef.addWidget(QLabel("类型")); ef.addWidget(self._edge_type_combo)
        ef.addWidget(QLabel("起点")); ef.addWidget(self._edge_start)
        ef.addWidget(QLabel("终点")); ef.addWidget(self._edge_end)
        ef.addWidget(QLabel("插值点")); ef.addWidget(self._edge_interp)
        csv_e = QPushButton("导入CSV")
        csv_e.clicked.connect(self._import_edges_csv)
        ef.addWidget(ae); ef.addWidget(csv_e); ef.addStretch(1)
        layout.addLayout(ef)

        self._edge_list = QTextEdit()
        self._edge_list.setReadOnly(True); self._edge_list.setMaximumHeight(80)
        self._edge_list.setPlaceholderText("当前对象的边显示在这里。")
        layout.addWidget(self._edge_list)
        de_btn = QPushButton("删除选中边")
        de_btn.clicked.connect(self._delete_selected_edge)
        clr_e = QPushButton("清空边")
        clr_e.clicked.connect(self._clear_all_edges)
        er = QHBoxLayout(); er.setSpacing(6)
        er.addWidget(de_btn); er.addWidget(clr_e); er.addStretch(1)
        layout.addLayout(er)
        self._edge_defs = []

        self._block_widget = QWidget()
        blk = QVBoxLayout(self._block_widget)
        blk.setContentsMargins(0,0,0,0); blk.setSpacing(6)
        bl = QLabel("块 (Blocks)")
        bl.setStyleSheet("font-weight: 600;")
        blk.addWidget(bl)
        br = QHBoxLayout(); br.setSpacing(6)
        self._block_vert_inputs = []
        for i in range(8):
            sb = QSpinBox(); sb.setRange(0,99); sb.setValue(i)
            sb.valueChanged.connect(self._on_block_vert_changed)
            self._block_vert_inputs.append(sb)
            br.addWidget(QLabel(str(i))); br.addWidget(sb)
        br.addStretch(1)
        blk.addLayout(br)

        cr = QHBoxLayout(); cr.setSpacing(8)
        self._geo_nx = QSpinBox(); self._geo_nx.setRange(1,500); self._geo_nx.setValue(10)
        self._geo_ny = QSpinBox(); self._geo_ny.setRange(1,500); self._geo_ny.setValue(10)
        self._geo_nz = QSpinBox(); self._geo_nz.setRange(1,500); self._geo_nz.setValue(10)
        for w in (self._geo_nx,self._geo_ny,self._geo_nz):
            w.valueChanged.connect(self._on_cell_changed); w.setMinimumWidth(80)
        cr.addWidget(QLabel("Nx")); cr.addWidget(self._geo_nx)
        cr.addWidget(QLabel("Ny")); cr.addWidget(self._geo_ny)
        cr.addWidget(QLabel("Nz")); cr.addWidget(self._geo_nz)
        self._geo_grading = QLineEdit("1 1 1")
        self._geo_grading.setMaximumWidth(120)
        self._geo_grading.textChanged.connect(self._on_grading_changed)
        cr.addWidget(QLabel("Grading")); cr.addWidget(self._geo_grading)
        cr.addStretch(1)
        blk.addLayout(cr)
        layout.addWidget(self._block_widget)

        self._boundary_widget = QWidget()
        bnr = QHBoxLayout(self._boundary_widget)
        bnr.setContentsMargins(0,0,0,0); bnr.setSpacing(8)
        bnr.addWidget(QLabel("入口"))
        self._geo_inlet = QLineEdit("inlet"); bnr.addWidget(self._geo_inlet)
        bnr.addWidget(QLabel("出口"))
        self._geo_outlet = QLineEdit("outlet"); bnr.addWidget(self._geo_outlet)
        bnr.addWidget(QLabel("壁面"))
        self._geo_walls = QLineEdit("fixedWalls"); bnr.addWidget(self._geo_walls)
        bnr.addStretch(1)
        layout.addWidget(self._boundary_widget)

        self._geo_preview_canvas = NativeVtkPreviewWidget(wrapper, background=(0.12, 0.12, 0.12))
        self._geo_preview_canvas.setMinimumHeight(380)
        layout.addWidget(self._geo_preview_canvas)

        br2 = QHBoxLayout(); br2.setSpacing(8)
        gb = QPushButton("生成 blockMeshDict + STL")
        gb.clicked.connect(lambda _checked=False: self._apply_draw_geometry())
        zoom_btn = QPushButton("放大预览")
        zoom_btn.clicked.connect(lambda _checked=False: self._open_draw_geometry_preview_dialog())
        rb = QPushButton("重置全部")
        rb.clicked.connect(lambda _checked=False: self._reset_draw_geometry())
        br2.addWidget(gb); br2.addWidget(zoom_btn); br2.addWidget(rb); br2.addStretch(1)
        layout.addLayout(br2)

        layout.addStretch(1)
        scroll.setWidget(content)
        root_layout.addWidget(scroll)
        self._load_draw_geometry_state()
        self._load_active_object_to_ui()
        self._refresh_draw_geo_preview()
        return wrapper

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
        description = QLabel("当前页面按“结果场选择 + 显示方式选择”组织后处理入口。先接入表面云图、切片、矢量箭头和流线。")
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
        self._result_field_combo.currentTextChanged.connect(lambda _text: self._update_result_field_metadata())
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
            "已接入：Surface、Slice、Vector、Streamline、Glyph。Contour、Iso-surface、Volume rendering 后续接入 VTK 专项算法。"
        )
        display_hint.setWordWrap(True)
        display_hint.setObjectName("sectionHint")
        display_layout.addWidget(display_title)
        display_layout.addLayout(display_row)
        display_layout.addWidget(display_hint)

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
