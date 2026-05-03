from __future__ import annotations

import json
import shlex
import re
from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata
from PySide6.QtCore import QPoint, Qt
from PySide6.QtCore import QProcess, QTimer
from PySide6.QtGui import QFont
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkCommonMath import vtkRungeKutta4
from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer
from vtkmodules.vtkIOGeometry import vtkSTLReader
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
    QListWidget,
    QListWidgetItem,
    QApplication,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QScrollArea,
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
    QWidgetAction,
)
from vtkmodules.util.numpy_support import vtk_to_numpy

from foamdesk.app.bootstrap import ApplicationContext
from foamdesk.domain.models import SimulationParameters, SimulationProject
from foamdesk.services.geometry_import_service import GeometryAsset, SnappyHexMeshSettings, StlTransform
from foamdesk.services.project_service import BoundaryConditionSettings, ComputationDomainTemplate
from foamdesk.ui.startup_window import StartupWindow
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


class TutorialOverlay(QFrame):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("tutorialPanel")
        self.hide()
        self.setFixedWidth(560)
        self._drag_start: QPoint | None = None

        panel_layout = QVBoxLayout(self)
        panel_layout.setContentsMargins(24, 20, 24, 20)
        panel_layout.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("FoamDesk 新手教程")
        title.setObjectName("tutorialTitle")
        close_button = QPushButton("×")
        close_button.setObjectName("tutorialIconButton")
        close_button.setFixedSize(32, 30)
        close_button.clicked.connect(self.hide)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(close_button)

        body = QLabel(
            "1. 点击“新建项目”，输入项目名称。\n"
            "2. 左侧 Case 树会显示真实项目，点击项目设为当前 Case。\n"
            "3. 打开“设置”，确认工作区、字体、字号和 OpenFOAM 环境脚本。\n"
            "4. 打开“环境检查”，确认 OpenFOAM 环境是否可用。\n"
            "5. 点击“运行”，当前阶段会执行 blockMesh + icoFoam 最小仿真。\n"
            "6. 底部“日志 / 任务 / 问题”面板会显示运行信息。\n\n"
            "当前 Sprint：Sprint 2 UI 可用性与设置系统。\n"
            "下一阶段：Sprint 3 项目管理与 OpenFOAM 最小执行闭环。"
        )
        body.setObjectName("tutorialBody")
        body.setWordWrap(True)

        panel_layout.addLayout(header)
        panel_layout.addWidget(body)

    def show_overlay(self) -> None:
        if self.parentWidget():
            parent_rect = self.parentWidget().rect()
            self.adjustSize()
            margin = 18
            self.move(parent_rect.right() - self.width() - margin, 118)
        self.raise_()
        self.show()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_start is not None and event.buttons() & Qt.MouseButton.LeftButton:
            target = event.globalPosition().toPoint() - self._drag_start
            if self.parentWidget():
                parent_rect = self.parentWidget().rect()
                target.setX(max(8, min(target.x(), parent_rect.width() - self.width() - 8)))
                target.setY(max(8, min(target.y(), parent_rect.height() - self.height() - 8)))
            self.move(target)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_start = None
        super().mouseReleaseEvent(event)


class VtkViewerDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("FoamDesk 3D 视图")
        self.resize(920, 680)
        self._active_plot_callback: Callable[[], None] | None = None
        self._tab_refresh_callbacks: dict[QWidget, Callable[[], None]] = {}
        self._is_refreshing_tab = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        action_row = QHBoxLayout()
        self._geometry_assets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._filter_mode_combo = QComboBox()
        self._filter_mode_combo.addItems(["全部显示", "所有 STL 附近", "单个 STL 附近"])
        self._filter_asset_combo = QComboBox()
        self._filter_radius_input = QDoubleSpinBox()
        self._filter_radius_input.setRange(0.0, 100000.0)
        self._filter_radius_input.setDecimals(4)
        self._filter_radius_input.setSingleStep(0.05)
        self._filter_radius_input.setValue(0.1)
        self._filter_mode_combo.currentIndexChanged.connect(self._refresh_current_tab)
        self._filter_asset_combo.currentIndexChanged.connect(self._refresh_current_tab)
        self._filter_radius_input.valueChanged.connect(self._refresh_current_tab)
        action_row.addWidget(QLabel("显示范围"))
        action_row.addWidget(self._filter_mode_combo)
        action_row.addWidget(QLabel("STL"))
        action_row.addWidget(self._filter_asset_combo)
        action_row.addWidget(QLabel("附近半径"))
        action_row.addWidget(self._filter_radius_input)
        action_row.addStretch(1)
        clear_button = QPushButton("清空视图")
        clear_button.clicked.connect(self._clear_tabs)
        export_button = QPushButton("导出 PNG")
        export_button.clicked.connect(self._export_png)
        action_row.addWidget(clear_button)
        action_row.addWidget(export_button)
        layout.addLayout(action_row)

        self._last_plot_title = "foamdesk_visualization"
        self.figure: Figure | None = None
        self.canvas: FigureCanvas | None = None
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setTabsClosable(True)
        self._tabs.tabCloseRequested.connect(self._close_tab)
        layout.addWidget(self._tabs)

    def set_geometry_assets(self, assets: list[GeometryAsset]) -> None:
        self._geometry_assets = {}
        current_name = self._filter_asset_combo.currentText()
        self._filter_asset_combo.blockSignals(True)
        self._filter_asset_combo.clear()
        for asset in assets:
            if asset.format.upper() != "STL" or not asset.stored_path.exists():
                continue
            bounds = self._stl_bounds(asset.stored_path)
            if bounds is None:
                continue
            self._geometry_assets[asset.name] = bounds
            self._filter_asset_combo.addItem(asset.name)
        if current_name:
            index = self._filter_asset_combo.findText(current_name)
            if index >= 0:
                self._filter_asset_combo.setCurrentIndex(index)
        self._filter_asset_combo.blockSignals(False)

    def _stl_bounds(self, path: Path) -> tuple[np.ndarray, np.ndarray] | None:
        reader = vtkSTLReader()
        reader.SetFileName(str(path))
        reader.Update()
        points = self._points(reader.GetOutput())
        if points.size == 0:
            return None
        return points.min(axis=0), points.max(axis=0)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._clear_tabs()
        super().closeEvent(event)

    def plot_cube(self) -> None:
        self._begin_plot(lambda: self.plot_cube())
        axes = self._reset_axes("3D Preview")
        corners = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=float,
        )
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        for start, end in edges:
            axes.plot(
                corners[[start, end], 0],
                corners[[start, end], 1],
                corners[[start, end], 2],
                color="#4fc3ff",
                linewidth=2.0,
            )
        self._finish_axes(axes, corners)

    def plot_polydata_points(self, poly_data, title: str) -> None:
        self._begin_plot(lambda: self.plot_polydata_points(poly_data, title))
        points = self._points(poly_data)
        axes = self._reset_axes(title)
        if points.size == 0:
            axes.text2D(0.08, 0.5, "No point data in current case.", color="#d4d4d4")
            self._show()
            return
        points = self._filter_points(points)
        if points.size == 0:
            axes.text2D(0.08, 0.5, "No point data in current display range.", color="#d4d4d4")
            self._show()
            return
        points = self._sample_points(points)
        axes.scatter(points[:, 0], points[:, 1], points[:, 2], s=6, c="#4fc3ff", alpha=0.85)
        self._finish_axes(axes, points)

    def plot_stl_file(self, path: Path) -> tuple[int, int]:
        self._begin_plot(lambda: self.plot_stl_file(path))
        reader = vtkSTLReader()
        reader.SetFileName(str(path))
        reader.Update()
        poly_data = reader.GetOutput()
        points = self._points(poly_data)
        faces = self._faces(poly_data)
        axes = self._reset_axes(f"STL {path.name}")
        if points.size == 0 or faces.size == 0:
            axes.text2D(0.08, 0.5, "No STL surface data to display.", color="#d4d4d4")
            self._show()
            return 0, 0
        faces = self._sample_faces(faces)
        polygons = [points[np.asarray(face, dtype=int)] for face in faces]
        collection = Poly3DCollection(
            polygons,
            facecolors=(0.58, 0.62, 0.66, 0.92),
            edgecolors=(0.12, 0.12, 0.12, 0.35),
            linewidths=0.25,
        )
        axes.add_collection3d(collection)
        self._finish_axes(axes, points)
        return len(points), len(faces)

    def plot_field_surface(
        self,
        poly_data,
        field_array,
        scalar_range: tuple[float, float],
        field_name: str,
    ) -> int:
        self._begin_plot(lambda: self.plot_field_surface(poly_data, field_array, scalar_range, field_name))
        points = self._points(poly_data)
        faces = self._faces(poly_data)
        values = self._scalar_values(field_array)
        axes = self._reset_axes(f"{field_name} Surface")
        if points.size == 0 or faces.size == 0 or values.size == 0:
            axes.text2D(0.08, 0.5, "No surface data to display.", color="#d4d4d4")
            self._show()
            return 0

        faces = self._sample_faces(faces)
        faces = self._filter_faces_by_center(points, faces)
        if faces.size == 0:
            axes.text2D(0.08, 0.5, "No surface faces in current display range.", color="#d4d4d4")
            self._show()
            return 0
        face_values = np.array(
            [values[np.asarray(face, dtype=int)].mean() for face in faces],
            dtype=float,
        )
        polygons = [points[np.asarray(face, dtype=int)] for face in faces]
        normalizer = Normalize(vmin=scalar_range[0], vmax=scalar_range[1])
        cmap = colormaps["turbo"]
        collection = Poly3DCollection(
            polygons,
            facecolors=cmap(normalizer(face_values)),
            edgecolors=(0.18, 0.18, 0.18, 0.35),
            linewidths=0.35,
            alpha=0.96,
        )
        axes.add_collection3d(collection)
        scalar_mappable = colormaps["turbo"]
        colorbar = self.figure.colorbar(
            self._scalar_mappable(scalar_range, scalar_mappable),
            ax=axes,
            shrink=0.72,
            pad=0.08,
        )
        colorbar.set_label(field_name, color="#d4d4d4")
        colorbar.ax.yaxis.set_tick_params(color="#d4d4d4")
        for label in colorbar.ax.get_yticklabels():
            label.set_color("#d4d4d4")
        self._finish_axes(axes, points)
        return len(faces)

    def plot_pressure_surface(self, poly_data, pressure_array, scalar_range: tuple[float, float]) -> int:
        return self.plot_field_surface(poly_data, pressure_array, scalar_range, "p")

    def plot_velocity_vectors(self, poly_data, vector_array, limit: int = 700) -> tuple[int, tuple[float, float]]:
        self._begin_plot(lambda: self.plot_velocity_vectors(poly_data, vector_array, limit))
        points = self._points(poly_data)
        vectors = vtk_to_numpy(vector_array)
        axes = self._reset_axes("Velocity Vectors |U|")
        if points.size == 0 or vectors.size == 0 or vectors.ndim != 2 or vectors.shape[1] < 3:
            axes.text2D(0.08, 0.5, "No velocity vector data to display.", color="#d4d4d4")
            self._show()
            return 0, (0.0, 0.0)

        count = min(len(points), len(vectors))
        points = points[:count]
        vectors = vectors[:count, :3]
        mask = self._filter_mask(points)
        points = points[mask]
        vectors = vectors[mask]
        speeds = np.linalg.norm(vectors, axis=1)
        non_zero = speeds > 1e-12
        if np.any(non_zero):
            points = points[non_zero]
            vectors = vectors[non_zero]
            speeds = speeds[non_zero]
        if len(points) == 0:
            axes.text2D(0.08, 0.5, "Velocity field is zero everywhere.", color="#d4d4d4")
            self._show()
            return 0, (0.0, 0.0)

        if len(points) > limit:
            indices = np.linspace(0, len(points) - 1, limit, dtype=int)
            points = points[indices]
            vectors = vectors[indices]
            speeds = speeds[indices]

        max_speed = float(speeds.max()) if speeds.size else 1.0
        domain_size = float(np.ptp(points, axis=0).max()) if len(points) else 1.0
        arrow_length = max(domain_size * 0.075 / max(max_speed, 1e-12), 1e-6)
        normalizer = Normalize(vmin=float(speeds.min()), vmax=max_speed)
        cmap = colormaps["turbo"]
        axes.quiver(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            vectors[:, 0],
            vectors[:, 1],
            vectors[:, 2],
            length=arrow_length,
            normalize=False,
            colors=cmap(normalizer(speeds)),
            linewidths=0.8,
            arrow_length_ratio=0.35,
        )
        colorbar = self.figure.colorbar(
            self._scalar_mappable((float(speeds.min()), max_speed), cmap),
            ax=axes,
            shrink=0.72,
            pad=0.08,
        )
        colorbar.set_label("|U|", color="#d4d4d4")
        colorbar.ax.yaxis.set_tick_params(color="#d4d4d4")
        for label in colorbar.ax.get_yticklabels():
            label.set_color("#d4d4d4")
        self._finish_axes(axes, points)
        return len(points), (float(speeds.min()), max_speed)

    def plot_velocity_slice(
        self,
        poly_data,
        vector_array,
        axis_name: str | None = None,
        normalized_position: float = 0.5,
    ) -> tuple[int, str, float, tuple[float, float]]:
        self._begin_plot(lambda: self.plot_velocity_slice(poly_data, vector_array, axis_name, normalized_position))
        points = self._points(poly_data)
        vectors = vtk_to_numpy(vector_array)
        axes = self._reset_axes("Velocity Slice |U|")
        if points.size == 0 or vectors.size == 0 or vectors.ndim != 2 or vectors.shape[1] < 3:
            axes.text2D(0.08, 0.5, "No velocity slice data to display.", color="#d4d4d4")
            self._show()
            return 0, "X", 0.0, (0.0, 0.0)

        count = min(len(points), len(vectors))
        points = points[:count]
        vectors = vectors[:count, :3]
        mask = self._filter_mask(points)
        points = points[mask]
        vectors = vectors[mask]
        if len(points) == 0:
            axes.text2D(0.08, 0.5, "No velocity slice points in current display range.", color="#d4d4d4")
            self._show()
            return 0, "X", 0.0, (0.0, 0.0)
        speeds = np.linalg.norm(vectors, axis=1)
        normalized_position = min(max(float(normalized_position), 0.0), 1.0)

        def build_mask(axis: int) -> tuple[float, np.ndarray]:
            axis_min = float(points[:, axis].min())
            axis_max = float(points[:, axis].max())
            center = axis_min + (axis_max - axis_min) * normalized_position
            span = float(axis_max - axis_min)
            tolerance = max(span * 0.055, 1e-9)
            mask = np.abs(points[:, axis] - center) <= tolerance
            if mask.sum() < 8:
                nearest = np.argsort(np.abs(points[:, axis] - center))[: min(200, len(points))]
                mask = np.zeros(len(points), dtype=bool)
                mask[nearest] = True
            return center, mask

        requested_axis = None if not axis_name or axis_name == "自动" else "XYZ".find(axis_name)
        if requested_axis is not None and requested_axis >= 0:
            axis = requested_axis
            center, mask = build_mask(axis)
        else:
            candidates: list[tuple[float, int, float, np.ndarray]] = []
            for candidate_axis in range(3):
                center, mask = build_mask(candidate_axis)
                slice_speeds = speeds[mask]
                score = float(slice_speeds.max() - slice_speeds.min()) if slice_speeds.size else -1.0
                candidates.append((score, candidate_axis, center, mask))
            _, axis, center, mask = max(candidates, key=lambda item: item[0])

        resolved_axis_name = "XYZ"[axis]
        slice_points = points[mask]
        slice_speeds = speeds[mask]
        if len(slice_points) == 0:
            axes.text2D(0.08, 0.5, "No points near the selected slice.", color="#d4d4d4")
            self._show()
            return 0, resolved_axis_name, center, (0.0, 0.0)

        speed_range = (float(slice_speeds.min()), float(slice_speeds.max()))
        if speed_range[0] == speed_range[1]:
            speed_range = (speed_range[0] - 1.0, speed_range[1] + 1.0)
        scatter = axes.scatter(
            slice_points[:, 0],
            slice_points[:, 1],
            slice_points[:, 2],
            c=slice_speeds,
            cmap="turbo",
            s=28,
            alpha=0.96,
            vmin=speed_range[0],
            vmax=speed_range[1],
        )
        colorbar = self.figure.colorbar(scatter, ax=axes, shrink=0.72, pad=0.08)
        colorbar.set_label("|U|", color="#d4d4d4")
        colorbar.ax.yaxis.set_tick_params(color="#d4d4d4")
        for label in colorbar.ax.get_yticklabels():
            label.set_color("#d4d4d4")
        self._finish_axes(axes, points)
        return len(slice_points), resolved_axis_name, center, speed_range

    def plot_velocity_slice_contour(
        self,
        poly_data,
        vector_array,
        axis_name: str | None = None,
        normalized_position: float = 0.5,
        resolution: int = 80,
    ) -> tuple[int, str, float, tuple[float, float]]:
        self._begin_plot(
            lambda: self.plot_velocity_slice_contour(poly_data, vector_array, axis_name, normalized_position, resolution)
        )
        points = self._points(poly_data)
        vectors = vtk_to_numpy(vector_array)
        axes = self._reset_axes("Velocity Slice Contour |U|")
        if points.size == 0 or vectors.size == 0 or vectors.ndim != 2 or vectors.shape[1] < 3:
            axes.text2D(0.08, 0.5, "No velocity slice data to display.", color="#d4d4d4")
            self._show()
            return 0, "X", 0.0, (0.0, 0.0)

        count = min(len(points), len(vectors))
        points = points[:count]
        vectors = vectors[:count, :3]
        mask = self._filter_mask(points)
        points = points[mask]
        vectors = vectors[mask]
        if len(points) == 0:
            axes.text2D(0.08, 0.5, "No contour slice points in current display range.", color="#d4d4d4")
            self._show()
            return 0, "X", 0.0, (0.0, 0.0)
        speeds = np.linalg.norm(vectors, axis=1)
        normalized_position = min(max(float(normalized_position), 0.0), 1.0)

        def build_mask(axis: int) -> tuple[float, np.ndarray]:
            axis_min = float(points[:, axis].min())
            axis_max = float(points[:, axis].max())
            center = axis_min + (axis_max - axis_min) * normalized_position
            span = float(axis_max - axis_min)
            tolerance = max(span * 0.055, 1e-9)
            mask = np.abs(points[:, axis] - center) <= tolerance
            if mask.sum() < 8:
                nearest = np.argsort(np.abs(points[:, axis] - center))[: min(240, len(points))]
                mask = np.zeros(len(points), dtype=bool)
                mask[nearest] = True
            return center, mask

        requested_axis = None if not axis_name or axis_name == "自动" else "XYZ".find(axis_name)
        if requested_axis is not None and requested_axis >= 0:
            axis = requested_axis
            center, mask = build_mask(axis)
        else:
            candidates: list[tuple[float, int, float, np.ndarray]] = []
            for candidate_axis in range(3):
                center, mask = build_mask(candidate_axis)
                slice_speeds = speeds[mask]
                score = float(slice_speeds.max() - slice_speeds.min()) if slice_speeds.size else -1.0
                candidates.append((score, candidate_axis, center, mask))
            _, axis, center, mask = max(candidates, key=lambda item: item[0])

        resolved_axis_name = "XYZ"[axis]
        slice_points = points[mask]
        slice_speeds = speeds[mask]
        if len(slice_points) < 4:
            axes.text2D(0.08, 0.5, "Not enough points for contour interpolation.", color="#d4d4d4")
            self._show()
            return len(slice_points), resolved_axis_name, center, (0.0, 0.0)

        plane_axes = [index for index in range(3) if index != axis]
        plane_points = slice_points[:, plane_axes]
        x_values = np.linspace(float(plane_points[:, 0].min()), float(plane_points[:, 0].max()), resolution)
        y_values = np.linspace(float(plane_points[:, 1].min()), float(plane_points[:, 1].max()), resolution)
        grid_x, grid_y = np.meshgrid(x_values, y_values)
        grid_speed = griddata(plane_points, slice_speeds, (grid_x, grid_y), method="linear")
        if np.isnan(grid_speed).all():
            grid_speed = griddata(plane_points, slice_speeds, (grid_x, grid_y), method="nearest")
        else:
            nearest_speed = griddata(plane_points, slice_speeds, (grid_x, grid_y), method="nearest")
            grid_speed = np.where(np.isnan(grid_speed), nearest_speed, grid_speed)

        speed_range = (float(np.nanmin(grid_speed)), float(np.nanmax(grid_speed)))
        if speed_range[0] == speed_range[1]:
            speed_range = (speed_range[0] - 1.0, speed_range[1] + 1.0)
        grid_points = np.zeros((resolution, resolution, 3), dtype=float)
        grid_points[:, :, axis] = center
        grid_points[:, :, plane_axes[0]] = grid_x
        grid_points[:, :, plane_axes[1]] = grid_y
        surface = axes.plot_surface(
            grid_points[:, :, 0],
            grid_points[:, :, 1],
            grid_points[:, :, 2],
            facecolors=colormaps["turbo"](Normalize(vmin=speed_range[0], vmax=speed_range[1])(grid_speed)),
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
            alpha=0.96,
        )
        surface.set_edgecolor("none")
        colorbar = self.figure.colorbar(
            self._scalar_mappable(speed_range, colormaps["turbo"]),
            ax=axes,
            shrink=0.72,
            pad=0.08,
        )
        colorbar.set_label("|U|", color="#d4d4d4")
        colorbar.ax.yaxis.set_tick_params(color="#d4d4d4")
        for label in colorbar.ax.get_yticklabels():
            label.set_color("#d4d4d4")
        self._finish_axes(axes, points)
        return len(slice_points), resolved_axis_name, center, speed_range

    def plot_velocity_streamlines(
        self,
        source_poly_data,
        streamline_poly_data,
        main_axis: str,
        speed_range: tuple[float, float],
    ) -> tuple[int, int, str, tuple[float, float]]:
        self._begin_plot(
            lambda: self.plot_velocity_streamlines(source_poly_data, streamline_poly_data, main_axis, speed_range)
        )
        source_points = self._points(source_poly_data)
        line_points = self._points(streamline_poly_data)
        axes = self._reset_axes("Velocity Streamlines |U|")
        if source_points.size == 0 or line_points.size == 0 or streamline_poly_data.GetNumberOfLines() == 0:
            axes.text2D(0.08, 0.5, "No VTK streamline data to display.", color="#d4d4d4")
            self._show()
            return 0, 0, main_axis, speed_range

        vectors = streamline_poly_data.GetPointData().GetArray("U")
        line_speeds = self._scalar_values(vectors) if vectors is not None else np.zeros(len(line_points))
        lines = vtk_to_numpy(streamline_poly_data.GetLines().GetData())
        cmap = colormaps["turbo"]
        normalizer = Normalize(vmin=speed_range[0], vmax=speed_range[1])
        index = 0
        line_count = 0
        total_line_points = 0
        while index < len(lines):
            count = int(lines[index])
            index += 1
            ids = lines[index : index + count].astype(int)
            index += count
            if count < 2:
                continue
            line = line_points[ids]
            line_mask = self._filter_mask(line)
            line = line[line_mask]
            ids = ids[line_mask]
            if len(line) < 2:
                continue
            speed = float(line_speeds[ids].mean()) if len(line_speeds) else 0.0
            axes.plot(
                line[:, 0],
                line[:, 1],
                line[:, 2],
                color=cmap(normalizer(speed)),
                linewidth=1.8,
                alpha=0.95,
            )
            axes.scatter(line[0, 0], line[0, 1], line[0, 2], s=16, c="#ffffff", alpha=0.85)
            line_count += 1
            total_line_points += len(line)

        if line_count == 0:
            axes.text2D(0.08, 0.5, "No streamlines in current display range.", color="#d4d4d4")
            self._show()
            return 0, 0, main_axis, speed_range

        colorbar = self.figure.colorbar(
            self._scalar_mappable(speed_range, cmap),
            ax=axes,
            shrink=0.72,
            pad=0.08,
        )
        colorbar.set_label("|U|", color="#d4d4d4")
        colorbar.ax.yaxis.set_tick_params(color="#d4d4d4")
        for label in colorbar.ax.get_yticklabels():
            label.set_color("#d4d4d4")
        filtered_source_points = self._filter_points(source_points)
        self._finish_axes(axes, filtered_source_points if filtered_source_points.size else source_points)
        return line_count, total_line_points, main_axis, speed_range

    def plot_pressure_points(self, poly_data, pressure_array, scalar_range: tuple[float, float]) -> None:
        self._begin_plot(lambda: self.plot_pressure_points(poly_data, pressure_array, scalar_range))
        points = self._points(poly_data)
        pressure = vtk_to_numpy(pressure_array)
        axes = self._reset_axes("Pressure Point Cloud")
        if points.size == 0 or pressure.size == 0:
            axes.text2D(0.08, 0.5, "No pressure point data in current case.", color="#d4d4d4")
            self._show()
            return
        count = min(len(points), len(pressure))
        points = points[:count]
        pressure = pressure[:count]
        points, pressure = self._filter_points_values(points, pressure)
        if points.size == 0:
            axes.text2D(0.08, 0.5, "No pressure points in current display range.", color="#d4d4d4")
            self._show()
            return
        points, pressure = self._sample_points(points, pressure)
        scatter = axes.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=pressure,
            cmap="turbo",
            s=8,
            alpha=0.92,
            vmin=scalar_range[0],
            vmax=scalar_range[1],
        )
        colorbar = self.figure.colorbar(scatter, ax=axes, shrink=0.72, pad=0.08)
        colorbar.set_label("p", color="#d4d4d4")
        colorbar.ax.yaxis.set_tick_params(color="#d4d4d4")
        for label in colorbar.ax.get_yticklabels():
            label.set_color("#d4d4d4")
        self._finish_axes(axes, points)

    def _reset_axes(self, title: str):
        self._last_plot_title = title
        self.figure = Figure(figsize=(8, 6), facecolor="#1e1e1e", tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        if self._active_plot_callback is not None:
            self._tab_refresh_callbacks[self.canvas] = self._active_plot_callback
        self._tabs.addTab(self.canvas, self._tab_title(title))
        self._tabs.setCurrentWidget(self.canvas)
        axes = self.figure.add_subplot(111, projection="3d", facecolor="#1e1e1e")
        axes.set_title(title, color="#d4d4d4", pad=14)
        axes.set_xlabel("X", color="#d4d4d4")
        axes.set_ylabel("Y", color="#d4d4d4")
        axes.set_zlabel("Z", color="#d4d4d4")
        axes.tick_params(colors="#d4d4d4")
        axes.grid(True, color="#333333", linestyle="--", linewidth=0.6)
        return axes

    def _finish_axes(self, axes, points) -> None:
        if points.size:
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            center = (mins + maxs) / 2.0
            radius = max((maxs - mins).max() / 2.0, 0.5)
            axes.set_xlim(center[0] - radius, center[0] + radius)
            axes.set_ylim(center[1] - radius, center[1] + radius)
            axes.set_zlim(center[2] - radius, center[2] + radius)
        axes.view_init(elev=24, azim=-55)
        self._show()

    def _show(self) -> None:
        if self.canvas is not None:
            self.canvas.draw()
        self.show()
        self.raise_()
        self.activateWindow()

    def _export_png(self) -> None:
        current_canvas = self._tabs.currentWidget()
        if not isinstance(current_canvas, FigureCanvas):
            QMessageBox.information(self, "暂无视图", "当前没有可导出的 3D 视图。")
            return
        current_title = self._tabs.tabText(self._tabs.currentIndex()) or self._last_plot_title
        safe_title = "".join(
            character if character.isalnum() or character in ("-", "_") else "_"
            for character in current_title.strip().lower()
        ).strip("_") or "foamdesk_visualization"
        default_name = f"{safe_title}.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出当前 3D 图像",
            default_name,
            "PNG 图片 (*.png)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".png"):
            file_path = f"{file_path}.png"
        try:
            current_canvas.figure.savefig(
                file_path,
                dpi=180,
                facecolor=current_canvas.figure.get_facecolor(),
                bbox_inches="tight",
            )
        except OSError as error:
            QMessageBox.warning(self, "导出失败", f"PNG 导出失败：{error}")
            return
        QMessageBox.information(self, "导出完成", f"已导出 PNG：\n{file_path}")

    def export_all_pngs(self, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        used_names: set[str] = set()
        for index in range(self._tabs.count()):
            canvas = self._tabs.widget(index)
            if not isinstance(canvas, FigureCanvas):
                continue
            title = self._tabs.tabText(index) or f"3d_view_{index + 1}"
            base_name = self._safe_file_stem(title)
            file_name = f"{base_name}.png"
            suffix = 2
            while file_name in used_names:
                file_name = f"{base_name}_{suffix}.png"
                suffix += 1
            used_names.add(file_name)
            path = output_dir / file_name
            canvas.figure.savefig(
                path,
                dpi=180,
                facecolor=canvas.figure.get_facecolor(),
                bbox_inches="tight",
            )
            paths.append(path)
        return paths

    def _tab_title(self, title: str) -> str:
        base = title.strip() or "3D View"
        existing_titles = {self._tabs.tabText(index) for index in range(self._tabs.count())}
        if base not in existing_titles:
            return base
        suffix = 2
        while f"{base} {suffix}" in existing_titles:
            suffix += 1
        return f"{base} {suffix}"

    def _safe_file_stem(self, title: str) -> str:
        return "".join(
            character if character.isalnum() or character in ("-", "_") else "_"
            for character in title.strip().lower()
        ).strip("_") or "foamdesk_visualization"

    def _close_tab(self, index: int) -> None:
        widget = self._tabs.widget(index)
        if widget is not None:
            self._tab_refresh_callbacks.pop(widget, None)
        self._tabs.removeTab(index)
        if widget is not None:
            widget.deleteLater()

    def _clear_tabs(self) -> None:
        while self._tabs.count():
            self._close_tab(0)
        self.figure = None
        self.canvas = None

    def _begin_plot(self, callback: Callable[[], None]) -> None:
        self._active_plot_callback = callback

    def _refresh_current_tab(self) -> None:
        if self._is_refreshing_tab:
            return
        current_widget = self._tabs.currentWidget()
        if current_widget is None:
            return
        callback = self._tab_refresh_callbacks.get(current_widget)
        if callback is None:
            return
        current_index = self._tabs.currentIndex()
        self._is_refreshing_tab = True
        try:
            self._close_tab(current_index)
            callback()
        finally:
            self._is_refreshing_tab = False

    def _points(self, poly_data) -> np.ndarray:
        vtk_points = poly_data.GetPoints()
        if vtk_points is None:
            return np.empty((0, 3), dtype=float)
        return vtk_to_numpy(vtk_points.GetData())

    def _scalar_values(self, field_array) -> np.ndarray:
        values = vtk_to_numpy(field_array)
        if values.ndim == 1:
            return values
        return np.linalg.norm(values, axis=1)

    def _faces(self, poly_data) -> np.ndarray:
        polygons = poly_data.GetPolys()
        if polygons is None:
            return np.empty((0,), dtype=object)
        raw = vtk_to_numpy(polygons.GetData())
        faces: list[np.ndarray] = []
        index = 0
        while index < len(raw):
            count = int(raw[index])
            index += 1
            if count >= 3:
                faces.append(raw[index : index + count].astype(int))
            index += count
        return np.array(faces, dtype=object)

    def _filter_points(self, points: np.ndarray) -> np.ndarray:
        return points[self._filter_mask(points)]

    def _filter_points_values(self, points: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = self._filter_mask(points)
        return points[mask], values[mask]

    def _filter_faces_by_center(self, points: np.ndarray, faces: np.ndarray) -> np.ndarray:
        if points.size == 0 or faces.size == 0:
            return faces
        centers = np.array([points[np.asarray(face, dtype=int)].mean(axis=0) for face in faces], dtype=float)
        return faces[self._filter_mask(centers)]

    def _filter_mask(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.zeros((0,), dtype=bool)
        mode = self._filter_mode_combo.currentText() if hasattr(self, "_filter_mode_combo") else "全部显示"
        if mode == "全部显示" or not self._geometry_assets:
            return np.ones((len(points),), dtype=bool)

        radius = self._filter_radius_input.value() if hasattr(self, "_filter_radius_input") else 0.0
        selected_bounds: list[tuple[np.ndarray, np.ndarray]]
        if mode == "单个 STL 附近":
            asset_name = self._filter_asset_combo.currentText() if hasattr(self, "_filter_asset_combo") else ""
            selected_bounds = [self._geometry_assets[asset_name]] if asset_name in self._geometry_assets else []
        else:
            selected_bounds = list(self._geometry_assets.values())
        if not selected_bounds:
            return np.ones((len(points),), dtype=bool)

        mask = np.zeros((len(points),), dtype=bool)
        for mins, maxs in selected_bounds:
            expanded_min = mins - radius
            expanded_max = maxs + radius
            mask |= np.all((points >= expanded_min) & (points <= expanded_max), axis=1)
        return mask

    def _sample_faces(self, faces: np.ndarray, limit: int = 8000) -> np.ndarray:
        if len(faces) <= limit:
            return faces
        indices = np.linspace(0, len(faces) - 1, limit, dtype=int)
        return faces[indices]

    def _scalar_mappable(self, scalar_range: tuple[float, float], cmap):
        from matplotlib.cm import ScalarMappable

        mappable = ScalarMappable(norm=Normalize(vmin=scalar_range[0], vmax=scalar_range[1]), cmap=cmap)
        mappable.set_array([])
        return mappable

    def _sample_points(
        self,
        points: np.ndarray,
        values: np.ndarray | None = None,
        limit: int = 12000,
    ):
        if len(points) <= limit:
            return (points, values) if values is not None else points
        indices = np.linspace(0, len(points) - 1, limit, dtype=int)
        if values is not None:
            return points[indices], values[indices]
        return points[indices]


class MainWindow(QMainWindow):
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
        self._tutorial_overlay: TutorialOverlay | None = None
        self._vtk_viewer: VtkViewerDialog | None = None
        self._visual_animation_timer = QTimer(self)
        self._visual_animation_timer.timeout.connect(self._advance_visualization_frame)
        self.setWindowTitle("FoamDesk")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.resize(1400, 900)
        self._build_ui()
        self._apply_settings_theme()
        self._refresh_status_bar()
        if initial_project is not None:
            self._activate_project(initial_project, "已恢复上次项目。")
        if self._context.settings_service.load().show_tutorial_on_startup:
            QTimer.singleShot(0, self._show_tutorial)

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
        shell_layout.addWidget(WindowTitleBar(self))
        shell_layout.addWidget(self._build_menu_bar())
        shell_layout.addWidget(workbench, 1)
        self._tutorial_overlay = TutorialOverlay(shell)
        self.setCentralWidget(shell)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._tutorial_overlay and self._tutorial_overlay.isVisible():
            self._tutorial_overlay.show_overlay()

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
        geometry_menu.addAction("打开几何/CAD 页", self._open_geometry_tab)
        geometry_menu.addAction("导入 STL 几何", self._import_stl_geometry)
        geometry_menu.addAction("生成 snappyHexMeshDict", self._generate_snappy_hex_mesh_dict)
        geometry_menu.addAction("运行 snappyHexMesh", self._run_snappy_hex_mesh)
        geometry_menu.addAction("运行 checkMesh", self._run_check_mesh)
        geometry_menu.addAction("一键前处理", self._run_preprocess_pipeline)
        geometry_menu.addAction("一键仿真流水线", self._run_simulation_pipeline)
        geometry_menu.addAction("预览已导入 STL", self._preview_imported_stl)

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
        toolbar.addAction("项目选择", self._return_to_project_selection)
        toolbar.addAction("保存", self._save_current_state)
        toolbar.addSeparator()
        toolbar.addAction("运行", self._run_minimal_simulation)
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
        self._workspace_tabs.addTab(self._build_parameter_tab(), "参数配置")
        self._workspace_tabs.addTab(self._build_solver_run_tab(), "求解运行")
        self._workspace_tabs.addTab(self._build_environment_tab(), "环境检查")
        self._workspace_tabs.addTab(self._build_settings_tab(), "设置")
        self._workspace_tabs.addTab(self._build_results_tab(), "结果")
        self._workspace_tabs.addTab(self._build_geometry_tab(), "几何/CAD")
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
            "- Case 树和结果/日志面板占位\n\n"
            "下一步将接入：\n"
            "- 项目新建流程\n"
            "- blockMesh + icoFoam 最小仿真\n"
            "- 实时日志和任务状态"
        )
        layout.addWidget(title)
        layout.addWidget(summary)
        return wrapper

    def _build_parameter_tab(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("参数配置")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        description = QLabel("当前阶段接入 controlDict 与 physicalProperties 的基础参数。")
        description.setWordWrap(True)

        form = QFormLayout()
        self._end_time_input = QLineEdit()
        self._delta_t_input = QLineEdit()
        self._write_interval_input = QSpinBox()
        self._write_interval_input.setRange(1, 1000000)
        self._viscosity_input = QLineEdit()
        form.addRow("结束时间 endTime", self._end_time_input)
        form.addRow("时间步长 deltaT", self._delta_t_input)
        form.addRow("写出间隔 writeInterval", self._write_interval_input)
        form.addRow("运动粘度 nu", self._viscosity_input)

        button_row = QHBoxLayout()
        load_button = QPushButton("加载当前项目参数")
        save_button = QPushButton("保存参数到 Case")
        default_button = QPushButton("恢复默认参数")
        load_button.clicked.connect(lambda _checked=False: self._load_case_parameters())
        save_button.clicked.connect(lambda _checked=False: self._save_case_parameters())
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
            "- endTime：仿真结束时间，越大运行越久。\n"
            "- deltaT：每一步的时间步长，越小越稳定但更慢。\n"
            "- writeInterval：每隔多少步写一次结果。\n"
            "- nu：运动粘度，当前最小算例默认 0.01。"
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

    def _build_geometry_tab(self) -> QWidget:
        wrapper = QWidget()
        root_layout = QVBoxLayout(wrapper)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("几何/CAD")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        description = QLabel(
            "当前 MVP 先支持 STL 导入到 OpenFOAM case 的 constant/triSurface，"
            "STEP/IGES/CATIA/SolidWorks 后续需要接入 CAD 内核。"
        )
        description.setWordWrap(True)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        action_row_2 = QHBoxLayout()
        action_row_2.setSpacing(8)
        import_button = QPushButton("导入 STL")
        refresh_button = QPushButton("刷新几何清单")
        snappy_button = QPushButton("生成 snappyHexMeshDict")
        run_snappy_button = QPushButton("运行 snappyHexMesh")
        check_mesh_button = QPushButton("运行 checkMesh")
        preprocess_button = QPushButton("一键前处理")
        simulation_pipeline_button = QPushButton("一键仿真流水线")
        preview_button = QPushButton("预览已导入 STL")
        edit_stl_button = QPushButton("编辑 STL 位置")
        precheck_button = QPushButton("运行前检查")
        limitation_button = QPushButton("STEP/IGES 支持说明")
        import_button.clicked.connect(lambda _checked=False: self._import_stl_geometry())
        refresh_button.clicked.connect(lambda _checked=False: self._refresh_geometry_panel())
        snappy_button.clicked.connect(lambda _checked=False: self._generate_snappy_hex_mesh_dict())
        run_snappy_button.clicked.connect(lambda _checked=False: self._run_snappy_hex_mesh())
        check_mesh_button.clicked.connect(lambda _checked=False: self._run_check_mesh())
        preprocess_button.clicked.connect(lambda _checked=False: self._run_preprocess_pipeline())
        simulation_pipeline_button.clicked.connect(lambda _checked=False: self._run_simulation_pipeline())
        preview_button.clicked.connect(lambda _checked=False: self._preview_imported_stl())
        edit_stl_button.clicked.connect(lambda _checked=False: self._edit_imported_stl_transform())
        precheck_button.clicked.connect(lambda _checked=False: self._run_preflight_check())
        limitation_button.clicked.connect(lambda _checked=False: self._show_cad_import_limitations())
        action_row.addWidget(import_button)
        action_row.addWidget(refresh_button)
        action_row_2.addWidget(snappy_button)
        action_row_2.addWidget(run_snappy_button)
        action_row_2.addWidget(check_mesh_button)
        action_row_2.addWidget(preprocess_button)
        action_row_2.addWidget(simulation_pipeline_button)
        action_row_2.addWidget(preview_button)
        action_row_2.addWidget(edit_stl_button)
        action_row_2.addWidget(precheck_button)
        action_row_2.addWidget(limitation_button)
        action_row.addStretch(1)
        action_row_2.addStretch(1)

        domain_form = QFormLayout()
        self._domain_template_combo = QComboBox()
        for template in self._context.project_service.domain_templates():
            self._domain_template_combo.addItem(template.name, template.key)
        self._domain_template_hint = QLabel("计算域 = 流体存在的空间盒子；STL 是放在盒子里的固体障碍物。")
        self._domain_template_hint.setWordWrap(True)
        self._domain_apply_state_label = QLabel("当前预览尚未绑定项目。")
        self._domain_apply_state_label.setWordWrap(True)
        self._domain_apply_state_label.setObjectName("DomainApplyStateLabel")
        self._domain_template_combo.currentIndexChanged.connect(self._refresh_domain_template_hint)
        apply_domain_button = QPushButton("应用计算域模板")
        import_template_stl_button = QPushButton("一键导入模板 STL")
        apply_domain_button.clicked.connect(lambda _checked=False: self._apply_domain_template())
        import_template_stl_button.clicked.connect(lambda _checked=False: self._import_template_stl())
        domain_row = QHBoxLayout()
        domain_row.setSpacing(8)
        domain_row.addWidget(self._domain_template_combo)
        domain_row.addWidget(apply_domain_button)
        domain_row.addWidget(import_template_stl_button)
        domain_row.addStretch(1)
        domain_widget = QWidget()
        domain_widget.setLayout(domain_row)
        domain_form.addRow("计算域模板", domain_widget)
        domain_form.addRow("应用状态", self._domain_apply_state_label)
        domain_form.addRow("模板说明", self._domain_template_hint)
        custom_domain_size_row = QHBoxLayout()
        custom_domain_size_row.setSpacing(8)
        custom_domain_cells_row = QHBoxLayout()
        custom_domain_cells_row.setSpacing(8)
        self._domain_length_x_input = QDoubleSpinBox()
        self._domain_length_y_input = QDoubleSpinBox()
        self._domain_length_z_input = QDoubleSpinBox()
        for input_widget in (
            self._domain_length_x_input,
            self._domain_length_y_input,
            self._domain_length_z_input,
        ):
            input_widget.setRange(0.01, 100000.0)
            input_widget.setDecimals(3)
            input_widget.setSingleStep(0.5)
            input_widget.setValue(1.0)
        self._domain_cells_x_input = QSpinBox()
        self._domain_cells_y_input = QSpinBox()
        self._domain_cells_z_input = QSpinBox()
        for input_widget in (
            self._domain_cells_x_input,
            self._domain_cells_y_input,
            self._domain_cells_z_input,
        ):
            input_widget.setRange(1, 100000)
            input_widget.setValue(10)
        apply_custom_domain_button = QPushButton("应用自定义计算域")
        apply_custom_domain_button.clicked.connect(lambda _checked=False: self._apply_custom_domain())
        for label, input_widget in (
            ("Lx", self._domain_length_x_input),
            ("Ly", self._domain_length_y_input),
            ("Lz", self._domain_length_z_input),
        ):
            input_widget.setMinimumWidth(96)
            custom_domain_size_row.addWidget(QLabel(label))
            custom_domain_size_row.addWidget(input_widget)
        custom_domain_size_row.addStretch(1)
        for label, input_widget in (
            ("Nx", self._domain_cells_x_input),
            ("Ny", self._domain_cells_y_input),
            ("Nz", self._domain_cells_z_input),
        ):
            input_widget.setMinimumWidth(96)
            custom_domain_cells_row.addWidget(QLabel(label))
            custom_domain_cells_row.addWidget(input_widget)
        custom_domain_cells_row.addWidget(apply_custom_domain_button)
        custom_domain_cells_row.addStretch(1)
        custom_domain_widget = QWidget()
        custom_domain_layout = QVBoxLayout(custom_domain_widget)
        custom_domain_layout.setContentsMargins(0, 0, 0, 0)
        custom_domain_layout.setSpacing(6)
        custom_domain_layout.addLayout(custom_domain_size_row)
        custom_domain_layout.addLayout(custom_domain_cells_row)
        domain_form.addRow("自定义尺寸/网格", custom_domain_widget)

        boundary_row_1 = QHBoxLayout()
        boundary_row_1.setSpacing(8)
        boundary_row_2 = QHBoxLayout()
        boundary_row_2.setSpacing(8)
        self._inlet_velocity_x_input = QDoubleSpinBox()
        self._inlet_velocity_y_input = QDoubleSpinBox()
        self._inlet_velocity_z_input = QDoubleSpinBox()
        for input_widget in (
            self._inlet_velocity_x_input,
            self._inlet_velocity_y_input,
            self._inlet_velocity_z_input,
        ):
            input_widget.setRange(-100000.0, 100000.0)
            input_widget.setDecimals(4)
            input_widget.setSingleStep(0.1)
            input_widget.setValue(0.0)
            input_widget.setMinimumWidth(96)
        self._inlet_velocity_x_input.setValue(1.0)
        self._outlet_pressure_input = QDoubleSpinBox()
        self._outlet_pressure_input.setRange(-100000.0, 100000.0)
        self._outlet_pressure_input.setDecimals(4)
        self._outlet_pressure_input.setSingleStep(0.1)
        self._outlet_pressure_input.setValue(0.0)
        self._outlet_pressure_input.setMinimumWidth(96)
        self._wall_type_combo = QComboBox()
        self._wall_type_combo.addItem("无滑移壁面 noSlip", "noSlip")
        self._wall_type_combo.addItem("滑移壁面 slip", "slip")
        apply_boundary_button = QPushButton("应用边界条件")
        apply_boundary_button.clicked.connect(lambda _checked=False: self._apply_boundary_conditions())
        for label, input_widget in (
            ("Ux", self._inlet_velocity_x_input),
            ("Uy", self._inlet_velocity_y_input),
            ("Uz", self._inlet_velocity_z_input),
        ):
            boundary_row_1.addWidget(QLabel(label))
            boundary_row_1.addWidget(input_widget)
        boundary_row_1.addStretch(1)
        boundary_row_2.addWidget(QLabel("出口压力 p"))
        boundary_row_2.addWidget(self._outlet_pressure_input)
        boundary_row_2.addWidget(QLabel("壁面类型"))
        boundary_row_2.addWidget(self._wall_type_combo)
        boundary_row_2.addWidget(apply_boundary_button)
        boundary_row_2.addStretch(1)
        boundary_widget = QWidget()
        boundary_layout = QVBoxLayout(boundary_widget)
        boundary_layout.setContentsMargins(0, 0, 0, 0)
        boundary_layout.setSpacing(6)
        boundary_layout.addLayout(boundary_row_1)
        boundary_layout.addLayout(boundary_row_2)
        domain_form.addRow("边界条件", boundary_widget)

        self._domain_preview_figure = Figure(figsize=(6.8, 2.8), facecolor="#1e1e1e", tight_layout=True)
        self._domain_preview_canvas = FigureCanvas(self._domain_preview_figure)
        self._domain_preview_canvas.setMinimumHeight(320)
        self._domain_preview_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        snappy_form = QFormLayout()
        self._snappy_min_refinement_input = QSpinBox()
        self._snappy_min_refinement_input.setRange(0, 6)
        self._snappy_min_refinement_input.setValue(1)
        self._snappy_max_refinement_input = QSpinBox()
        self._snappy_max_refinement_input.setRange(0, 8)
        self._snappy_max_refinement_input.setValue(2)
        self._snappy_location_x_input = QDoubleSpinBox()
        self._snappy_location_y_input = QDoubleSpinBox()
        self._snappy_location_z_input = QDoubleSpinBox()
        for input_widget in (
            self._snappy_location_x_input,
            self._snappy_location_y_input,
            self._snappy_location_z_input,
        ):
            input_widget.setRange(-100000.0, 100000.0)
            input_widget.setDecimals(4)
            input_widget.setSingleStep(0.1)
            input_widget.setValue(0.5)
        self._snappy_add_layers_checkbox = QCheckBox("启用边界层 addLayers")
        self._snappy_layer_thickness_input = QDoubleSpinBox()
        self._snappy_layer_thickness_input.setRange(0.01, 5.0)
        self._snappy_layer_thickness_input.setDecimals(3)
        self._snappy_layer_thickness_input.setSingleStep(0.05)
        self._snappy_layer_thickness_input.setValue(0.3)

        location_row = QHBoxLayout()
        location_row.setSpacing(8)
        location_row.addWidget(QLabel("X"))
        location_row.addWidget(self._snappy_location_x_input)
        location_row.addWidget(QLabel("Y"))
        location_row.addWidget(self._snappy_location_y_input)
        location_row.addWidget(QLabel("Z"))
        location_row.addWidget(self._snappy_location_z_input)
        location_row.addStretch(1)
        location_widget = QWidget()
        location_widget.setLayout(location_row)

        snappy_form.addRow("最小加密等级", self._snappy_min_refinement_input)
        snappy_form.addRow("最大加密等级", self._snappy_max_refinement_input)
        snappy_form.addRow("流体内部点 locationInMesh", location_widget)
        snappy_form.addRow("边界层", self._snappy_add_layers_checkbox)
        snappy_form.addRow("最终边界层厚度", self._snappy_layer_thickness_input)

        self._geometry_text = QTextEdit()
        self._geometry_text.setReadOnly(True)
        self._geometry_text.setPlainText("请先新建或打开项目，然后导入 STL 几何。")
        self._geometry_text.setMinimumHeight(180)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addLayout(action_row)
        layout.addLayout(action_row_2)
        layout.addLayout(domain_form)
        layout.addWidget(self._domain_preview_canvas)
        layout.addLayout(snappy_form)
        layout.addWidget(self._geometry_text, 1)
        scroll_area.setWidget(content)
        root_layout.addWidget(scroll_area)
        return wrapper

    def _build_solver_run_tab(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("求解运行")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        description = QLabel("当前阶段执行最小链路：blockMesh -> icoFoam。")
        description.setWordWrap(True)

        self._solver_status_label = QLabel("状态：空闲")
        self._solver_project_label = QLabel("当前项目：未选择")
        self._solver_case_path_label = QLabel("Case 路径：未选择")
        self._solver_command_label = QLabel("执行命令：blockMesh && icoFoam")

        action_row = QHBoxLayout()
        run_button = QPushButton("运行最小仿真")
        stop_button = QPushButton("停止当前任务")
        refresh_button = QPushButton("刷新摘要")
        run_button.clicked.connect(lambda _checked=False: self._run_minimal_simulation())
        stop_button.clicked.connect(lambda _checked=False: self._stop_current_process())
        refresh_button.clicked.connect(lambda _checked=False: self._refresh_solver_run_panel())
        action_row.addWidget(run_button)
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
        description = QLabel("当前阶段先索引 OpenFOAM 运行产物，后续再接入曲线和云图。")
        description.setWordWrap(True)
        refresh_visual_fields_button = QPushButton("刷新可视化字段")
        load_selected_surface_button = QPushButton("加载所选字段表面图")
        play_visual_animation_button = QPushButton("播放动画")
        stop_visual_animation_button = QPushButton("暂停动画")
        previous_frame_button = QPushButton("上一帧")
        next_frame_button = QPushButton("下一帧")

        def make_menu_button(title: str, actions: list[tuple[str, object]]) -> QPushButton:
            button = QPushButton(title)
            menu = QMenu(button)
            for label, callback in actions:
                action = menu.addAction(label)
                action.triggered.connect(lambda _checked=False, callback=callback: callback())
            button.setMenu(menu)
            return button

        def make_widget_menu_button(title: str, panel: QWidget) -> QPushButton:
            button = QPushButton(title)
            menu = QMenu(button)
            action = QWidgetAction(menu)
            action.setDefaultWidget(panel)
            menu.addAction(action)
            button.setMenu(menu)
            return button

        result_data_button = make_menu_button(
            "结果数据",
            [
                ("刷新结果索引", self._refresh_results_panel),
                ("导出求解指标", self._export_solver_metrics),
                ("绘制残差曲线", self._plot_residual_curve),
                ("导出 Markdown 报告", self._export_markdown_report),
            ],
        )
        three_d_button = make_menu_button(
            "3D 视图",
            [
                ("加载 3D 技术验证场景", self._load_3d_preview_scene),
                ("加载真实 OpenFOAM 3D Case", self._load_openfoam_3d_case),
            ],
        )
        contour_button = make_menu_button(
            "云图",
            [
                ("加载压力云图", self._load_pressure_cloud),
                ("加载压力表面云图", self._load_pressure_surface_cloud),
                ("加载所选字段表面图", self._load_selected_field_surface),
            ],
        )
        velocity_button = make_menu_button(
            "速度场",
            [
                ("加载速度箭头", self._load_velocity_vectors),
                ("加载速度切面", self._load_velocity_slice),
                ("加载连续速度切面", self._load_velocity_slice_contour),
                ("加载速度流线", self._load_velocity_streamlines),
            ],
        )
        refresh_visual_fields_button.clicked.connect(lambda _checked=False: self._refresh_visualization_selectors())
        load_selected_surface_button.clicked.connect(lambda _checked=False: self._load_selected_field_surface())
        play_visual_animation_button.clicked.connect(lambda _checked=False: self._play_visualization_animation())
        stop_visual_animation_button.clicked.connect(lambda _checked=False: self._stop_visualization_animation())
        previous_frame_button.clicked.connect(lambda _checked=False: self._step_visualization_frame(-1))
        next_frame_button.clicked.connect(lambda _checked=False: self._step_visualization_frame(1))
        action_row = QHBoxLayout()
        action_row.addWidget(result_data_button)
        action_row.addWidget(three_d_button)
        action_row.addWidget(contour_button)
        action_row.addWidget(velocity_button)
        self._visual_field_combo = QComboBox()
        self._visual_time_combo = QComboBox()
        self._slice_axis_combo = QComboBox()
        self._slice_axis_combo.addItems(["自动", "X", "Y", "Z"])
        self._slice_position_input = QDoubleSpinBox()
        self._slice_position_input.setRange(0.0, 1.0)
        self._slice_position_input.setSingleStep(0.05)
        self._slice_position_input.setDecimals(2)
        self._slice_position_input.setValue(0.5)
        self._visual_frame_interval_input = QSpinBox()
        self._visual_frame_interval_input.setRange(100, 5000)
        self._visual_frame_interval_input.setValue(800)
        self._visual_frame_interval_input.setSuffix(" ms")
        self._streamline_seed_count_input = QSpinBox()
        self._streamline_seed_count_input.setRange(4, 96)
        self._streamline_seed_count_input.setSingleStep(4)
        self._streamline_seed_count_input.setValue(24)
        self._streamline_length_factor_input = QDoubleSpinBox()
        self._streamline_length_factor_input.setRange(0.5, 10.0)
        self._streamline_length_factor_input.setSingleStep(0.25)
        self._streamline_length_factor_input.setDecimals(2)
        self._streamline_length_factor_input.setValue(2.5)
        self._streamline_length_factor_input.setSuffix(" x")
        self._streamline_step_percent_input = QDoubleSpinBox()
        self._streamline_step_percent_input.setRange(0.2, 10.0)
        self._streamline_step_percent_input.setSingleStep(0.2)
        self._streamline_step_percent_input.setDecimals(1)
        self._streamline_step_percent_input.setValue(2.0)
        self._streamline_step_percent_input.setSuffix(" %")
        self._visual_field_combo.setMinimumWidth(160)
        self._visual_time_combo.setMinimumWidth(160)
        self._slice_axis_combo.setMinimumWidth(86)
        self._slice_position_input.setMaximumWidth(92)
        self._streamline_seed_count_input.setMaximumWidth(92)
        self._streamline_length_factor_input.setMaximumWidth(92)
        self._streamline_step_percent_input.setMaximumWidth(92)
        field_time_panel = QWidget()
        field_time_layout = QVBoxLayout(field_time_panel)
        field_time_layout.setContentsMargins(10, 10, 10, 10)
        field_time_layout.setSpacing(8)
        field_row = QHBoxLayout()
        field_row.addWidget(QLabel("字段"))
        field_row.addWidget(self._visual_field_combo)
        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("时间步"))
        time_row.addWidget(self._visual_time_combo)
        field_action_row = QHBoxLayout()
        field_action_row.addWidget(refresh_visual_fields_button)
        field_action_row.addWidget(load_selected_surface_button)
        field_time_layout.addLayout(field_row)
        field_time_layout.addLayout(time_row)
        field_time_layout.addLayout(field_action_row)

        slice_panel = QWidget()
        slice_layout = QVBoxLayout(slice_panel)
        slice_layout.setContentsMargins(10, 10, 10, 10)
        slice_layout.setSpacing(8)
        slice_axis_row = QHBoxLayout()
        slice_axis_row.addWidget(QLabel("切面"))
        slice_axis_row.addWidget(self._slice_axis_combo)
        slice_position_row = QHBoxLayout()
        slice_position_row.addWidget(QLabel("位置"))
        slice_position_row.addWidget(self._slice_position_input)
        slice_hint = QLabel("0.00 最小侧，0.50 中间，1.00 最大侧")
        slice_hint.setObjectName("sectionHint")
        slice_layout.addLayout(slice_axis_row)
        slice_layout.addLayout(slice_position_row)
        slice_layout.addWidget(slice_hint)

        streamline_panel = QWidget()
        streamline_layout = QVBoxLayout(streamline_panel)
        streamline_layout.setContentsMargins(10, 10, 10, 10)
        streamline_layout.setSpacing(8)
        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("种子点"))
        seed_row.addWidget(self._streamline_seed_count_input)
        length_row = QHBoxLayout()
        length_row.addWidget(QLabel("追踪长度"))
        length_row.addWidget(self._streamline_length_factor_input)
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("积分步长"))
        step_row.addWidget(self._streamline_step_percent_input)
        streamline_hint = QLabel("种子点越多线越密；追踪长度越大线越长；步长越小越细但更慢。")
        streamline_hint.setWordWrap(True)
        streamline_hint.setObjectName("sectionHint")
        streamline_layout.addLayout(seed_row)
        streamline_layout.addLayout(length_row)
        streamline_layout.addLayout(step_row)
        streamline_layout.addWidget(streamline_hint)

        animation_panel = QWidget()
        animation_layout = QVBoxLayout(animation_panel)
        animation_layout.setContentsMargins(10, 10, 10, 10)
        animation_layout.setSpacing(8)
        frame_row = QHBoxLayout()
        frame_row.addWidget(previous_frame_button)
        frame_row.addWidget(next_frame_button)
        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("帧间隔"))
        interval_row.addWidget(self._visual_frame_interval_input)
        play_row = QHBoxLayout()
        play_row.addWidget(play_visual_animation_button)
        play_row.addWidget(stop_visual_animation_button)
        animation_layout.addLayout(frame_row)
        animation_layout.addLayout(interval_row)
        animation_layout.addLayout(play_row)

        field_time_button = make_widget_menu_button("字段时间", field_time_panel)
        slice_settings_button = make_widget_menu_button("切面设置", slice_panel)
        streamline_settings_button = make_widget_menu_button("流线设置", streamline_panel)
        animation_button = make_widget_menu_button("动画控制", animation_panel)
        action_row.addWidget(field_time_button)
        action_row.addWidget(slice_settings_button)
        action_row.addWidget(streamline_settings_button)
        action_row.addWidget(animation_button)
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
        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setPlainText("请先新建或打开项目，然后运行最小仿真。")
        self._results_text.setMaximumHeight(160)
        self._residual_figure = Figure(figsize=(6, 3), tight_layout=True)
        self._residual_canvas = FigureCanvas(self._residual_figure)
        self._residual_canvas.setMaximumHeight(220)
        self._vtk_hint_label = QLabel(
            "三维视图将以独立窗口打开，避免 WSL 下 VTK 原生控件覆盖 Qt 页面。"
        )
        self._vtk_hint_label.setWordWrap(True)
        self._vtk_hint_label.setObjectName("sectionHint")

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(action_scroll)
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

    def _open_geometry_tab(self) -> None:
        self._workspace_tabs.setCurrentIndex(6)
        self._refresh_geometry_panel()
        self._set_status("已打开几何/CAD 页。")

    def _import_stl_geometry(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "导入 STL 几何",
            str(self._current_project.path),
            "STL 几何 (*.stl *.STL)",
        )
        if not file_path:
            return
        transform = self._read_stl_transform_dialog(Path(file_path))
        if transform is None:
            return
        try:
            asset = self._context.geometry_import_service.import_stl(
                self._current_project,
                Path(file_path),
                transform,
            )
        except (OSError, ValueError) as error:
            self._show_error(f"导入 STL 失败：{error}")
            return
        self._append_log(f"STL 几何已导入：{asset.stored_path}")
        self._refresh_geometry_panel()
        self._workspace_tabs.setCurrentIndex(6)
        self._set_status("STL 几何导入完成。")

    def _read_stl_transform_dialog(
        self,
        source_path: Path,
        initial_transform: StlTransform | None = None,
    ) -> StlTransform | None:
        resolved_transform = initial_transform or StlTransform()
        dialog = QDialog(self)
        dialog.setWindowTitle("导入 STL：位置和缩放")
        dialog.resize(760, 620)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        scale_input = QDoubleSpinBox()
        scale_input.setRange(0.0001, 10000.0)
        scale_input.setDecimals(4)
        scale_input.setValue(resolved_transform.scale)
        scale_input.setSingleStep(0.1)
        x_input = QDoubleSpinBox()
        y_input = QDoubleSpinBox()
        z_input = QDoubleSpinBox()
        rx_input = QDoubleSpinBox()
        ry_input = QDoubleSpinBox()
        rz_input = QDoubleSpinBox()
        for input_widget in (x_input, y_input, z_input):
            input_widget.setRange(-100000.0, 100000.0)
            input_widget.setDecimals(4)
            input_widget.setSingleStep(0.1)
        for input_widget in (rx_input, ry_input, rz_input):
            input_widget.setRange(-3600.0, 3600.0)
            input_widget.setDecimals(3)
            input_widget.setSingleStep(5.0)
        x_input.setValue(resolved_transform.translate[0])
        y_input.setValue(resolved_transform.translate[1])
        z_input.setValue(resolved_transform.translate[2])
        rx_input.setValue(resolved_transform.rotate_degrees[0])
        ry_input.setValue(resolved_transform.rotate_degrees[1])
        rz_input.setValue(resolved_transform.rotate_degrees[2])
        hint = QLabel("这些参数会直接修改导入后的 STL 顶点坐标。变换顺序：缩放 -> 旋转 -> 平移。")
        hint.setWordWrap(True)
        preview_hint = QLabel("预览说明：灰色外框是 1x1x1 计算区域参考，蓝色几何是当前缩放和平移后的 STL 位置。")
        preview_hint.setWordWrap(True)
        preview_figure = Figure(figsize=(6.8, 3.8), facecolor="#1e1e1e", tight_layout=True)
        preview_canvas = FigureCanvas(preview_figure)
        preview_canvas.setMinimumHeight(320)

        def current_transform() -> StlTransform:
            return StlTransform(
                translate=(x_input.value(), y_input.value(), z_input.value()),
                scale=scale_input.value(),
                rotate_degrees=(rx_input.value(), ry_input.value(), rz_input.value()),
            )

        def refresh_preview() -> None:
            self._draw_stl_transform_preview(preview_figure, preview_canvas, source_path, current_transform())

        scale_input.valueChanged.connect(refresh_preview)
        x_input.valueChanged.connect(refresh_preview)
        y_input.valueChanged.connect(refresh_preview)
        z_input.valueChanged.connect(refresh_preview)
        rx_input.valueChanged.connect(refresh_preview)
        ry_input.valueChanged.connect(refresh_preview)
        rz_input.valueChanged.connect(refresh_preview)
        form.addRow("缩放 scale", scale_input)
        form.addRow("平移 X", x_input)
        form.addRow("平移 Y", y_input)
        form.addRow("平移 Z", z_input)
        form.addRow("旋转 X°", rx_input)
        form.addRow("旋转 Y°", ry_input)
        form.addRow("旋转 Z°", rz_input)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(hint)
        layout.addLayout(form)
        layout.addWidget(preview_hint)
        layout.addWidget(preview_canvas)
        layout.addWidget(buttons)
        refresh_preview()
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return current_transform()

    def _draw_stl_transform_preview(
        self,
        figure: Figure,
        canvas: FigureCanvas,
        source_path: Path,
        transform: StlTransform,
    ) -> None:
        figure.clear()
        axes = figure.add_subplot(111, projection="3d", facecolor="#1e1e1e")
        axes.set_title("STL Placement Preview", color="#d4d4d4", pad=10)
        axes.set_xlabel("X", color="#d4d4d4")
        axes.set_ylabel("Y", color="#d4d4d4")
        axes.set_zlabel("Z", color="#d4d4d4")
        axes.tick_params(colors="#d4d4d4")
        axes.grid(True, color="#333333", linestyle="--", linewidth=0.5)
        self._draw_unit_domain_wireframe(axes)

        try:
            points, faces = self._read_stl_preview_mesh(source_path)
        except (OSError, ValueError) as error:
            axes.text2D(0.05, 0.5, f"STL preview failed: {error}", color="#d4d4d4")
            canvas.draw()
            return

        if points.size == 0 or faces.size == 0:
            axes.text2D(0.05, 0.5, "No STL triangles to preview.", color="#d4d4d4")
            canvas.draw()
            return

        rotation = self._context.geometry_import_service._rotation_matrix(transform.rotate_degrees)
        transformed_points = np.array(
            [
                self._context.geometry_import_service._transform_vertex(
                    (float(point[0]), float(point[1]), float(point[2])),
                    float(transform.scale),
                    rotation,
                    transform.translate,
                )
                for point in points
            ],
            dtype=float,
        )
        sampled_faces = faces
        if len(sampled_faces) > 5000:
            indices = np.linspace(0, len(sampled_faces) - 1, 5000, dtype=int)
            sampled_faces = sampled_faces[indices]
        polygons = [transformed_points[np.asarray(face, dtype=int)] for face in sampled_faces]
        collection = Poly3DCollection(
            polygons,
            facecolors=(0.25, 0.74, 1.0, 0.72),
            edgecolors=(0.03, 0.18, 0.28, 0.45),
            linewidths=0.25,
        )
        axes.add_collection3d(collection)

        mins = np.minimum(transformed_points.min(axis=0), np.array([0.0, 0.0, 0.0]))
        maxs = np.maximum(transformed_points.max(axis=0), np.array([1.0, 1.0, 1.0]))
        center = (mins + maxs) / 2.0
        radius = max(float((maxs - mins).max()) / 2.0, 0.55)
        axes.set_xlim(center[0] - radius, center[0] + radius)
        axes.set_ylim(center[1] - radius, center[1] + radius)
        axes.set_zlim(center[2] - radius, center[2] + radius)
        axes.view_init(elev=24, azim=-55)
        canvas.draw()

    def _read_stl_preview_mesh(self, source_path: Path) -> tuple[np.ndarray, np.ndarray]:
        reader = vtkSTLReader()
        reader.SetFileName(str(source_path))
        reader.Update()
        poly_data = reader.GetOutput()
        vtk_points = poly_data.GetPoints()
        polygons = poly_data.GetPolys()
        if vtk_points is None or polygons is None:
            return np.empty((0, 3), dtype=float), np.empty((0,), dtype=object)
        points = vtk_to_numpy(vtk_points.GetData())
        raw = vtk_to_numpy(polygons.GetData())
        faces: list[np.ndarray] = []
        index = 0
        while index < len(raw):
            count = int(raw[index])
            index += 1
            if count >= 3:
                faces.append(raw[index : index + count].astype(int))
            index += count
        return points, np.array(faces, dtype=object)

    def _draw_unit_domain_wireframe(self, axes) -> None:
        self._draw_domain_wireframe(axes, (1.0, 1.0, 1.0))

    def _draw_domain_wireframe(self, axes, size: tuple[float, float, float]) -> None:
        length_x, length_y, length_z = size
        corners = np.array(
            [
                [0, 0, 0],
                [length_x, 0, 0],
                [length_x, length_y, 0],
                [0, length_y, 0],
                [0, 0, length_z],
                [length_x, 0, length_z],
                [length_x, length_y, length_z],
                [0, length_y, length_z],
            ],
            dtype=float,
        )
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        for start, end in edges:
            axes.plot(
                corners[[start, end], 0],
                corners[[start, end], 1],
                corners[[start, end], 2],
                color="#8a8a8a",
                linewidth=1.2,
                alpha=0.85,
            )

    def _draw_domain_template_wireframe(self, axes, template: ComputationDomainTemplate) -> np.ndarray:
        if template.shape == "pipe":
            return self._draw_pipe_domain_wireframe(axes, template)
        if template.shape == "bend":
            return self._draw_bend_domain_wireframe(axes, template)
        corners = np.array(self._context.project_service.domain_vertices(template), dtype=float)
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        for start, end in edges:
            axes.plot(
                corners[[start, end], 0],
                corners[[start, end], 1],
                corners[[start, end], 2],
                color="#8a8a8a",
                linewidth=1.2,
                alpha=0.85,
            )
        return corners

    def _draw_pipe_domain_wireframe(self, axes, template: ComputationDomainTemplate) -> np.ndarray:
        length_x, length_y, length_z = template.size
        center_y = length_y / 2.0
        center_z = length_z / 2.0
        radius = min(length_y, length_z) * 0.42
        theta = np.linspace(0.0, 2.0 * np.pi, 80)
        inlet = np.column_stack(
            [
                np.zeros_like(theta),
                center_y + radius * np.cos(theta),
                center_z + radius * np.sin(theta),
            ]
        )
        outlet = np.column_stack(
            [
                np.full_like(theta, length_x),
                center_y + radius * np.cos(theta),
                center_z + radius * np.sin(theta),
            ]
        )
        axes.plot(inlet[:, 0], inlet[:, 1], inlet[:, 2], color="#8a8a8a", linewidth=1.2, alpha=0.85)
        axes.plot(outlet[:, 0], outlet[:, 1], outlet[:, 2], color="#8a8a8a", linewidth=1.2, alpha=0.85)
        for angle in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
            y = center_y + radius * np.cos(angle)
            z = center_z + radius * np.sin(angle)
            axes.plot([0.0, length_x], [y, y], [z, z], color="#8a8a8a", linewidth=1.2, alpha=0.85)
        return np.vstack([inlet, outlet])

    def _draw_bend_domain_wireframe(self, axes, template: ComputationDomainTemplate) -> np.ndarray:
        length_x, length_y, height = template.size
        inner_radius = min(length_x, length_y) * 0.28
        outer_radius = min(length_x, length_y) * 0.62
        theta = np.linspace(0.0, np.pi / 2.0, 80)
        points: list[np.ndarray] = []
        for radius in (inner_radius, outer_radius):
            for z_value in (0.0, height):
                curve = np.column_stack(
                    [
                        radius * np.cos(theta),
                        radius * np.sin(theta),
                        np.full_like(theta, z_value),
                    ]
                )
                axes.plot(curve[:, 0], curve[:, 1], curve[:, 2], color="#8a8a8a", linewidth=1.2, alpha=0.85)
                points.append(curve)
        for angle in (0.0, np.pi / 2.0):
            for z_value in (0.0, height):
                axes.plot(
                    [inner_radius * np.cos(angle), outer_radius * np.cos(angle)],
                    [inner_radius * np.sin(angle), outer_radius * np.sin(angle)],
                    [z_value, z_value],
                    color="#8a8a8a",
                    linewidth=1.2,
                    alpha=0.85,
                )
            for radius in (inner_radius, outer_radius):
                axes.plot(
                    [radius * np.cos(angle), radius * np.cos(angle)],
                    [radius * np.sin(angle), radius * np.sin(angle)],
                    [0.0, height],
                    color="#8a8a8a",
                    linewidth=1.2,
                    alpha=0.85,
                )
        return np.vstack(points)

    def _refresh_geometry_panel(self) -> None:
        if not hasattr(self, "_geometry_text"):
            return
        if self._current_project is None:
            self._geometry_text.setPlainText("请先新建或打开项目，然后导入 STL 几何。")
            self._refresh_domain_preview()
            return
        self._load_domain_template_into_form()
        self._load_boundary_conditions_into_form()
        self._load_snappy_settings_into_form()
        self._geometry_text.setPlainText(self._context.geometry_import_service.format_assets(self._current_project))
        self._refresh_domain_preview()

    def _load_domain_template_into_form(self) -> None:
        if self._current_project is None or not hasattr(self, "_domain_template_combo"):
            return
        key = self._context.project_service.load_domain_template_key(self._current_project)
        index = self._domain_template_combo.findData(key)
        if index >= 0:
            self._domain_template_combo.blockSignals(True)
            self._domain_template_combo.setCurrentIndex(index)
            self._domain_template_combo.blockSignals(False)
        self._load_custom_domain_inputs()
        self._refresh_domain_template_hint()

    def _load_custom_domain_inputs(self) -> None:
        if self._current_project is None or not hasattr(self, "_domain_length_x_input"):
            return
        config_path = self._current_project.case_dir / "system" / "domain_config.json"
        if not config_path.exists():
            return
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            size = payload.get("size", [1.0, 1.0, 1.0])
            cells = payload.get("cells", [10, 10, 10])
            if len(size) == 3 and len(cells) == 3:
                self._domain_length_x_input.setValue(float(size[0]))
                self._domain_length_y_input.setValue(float(size[1]))
                self._domain_length_z_input.setValue(float(size[2]))
                self._domain_cells_x_input.setValue(int(cells[0]))
                self._domain_cells_y_input.setValue(int(cells[1]))
                self._domain_cells_z_input.setValue(int(cells[2]))
        except (OSError, ValueError, TypeError):
            return

    def _load_boundary_conditions_into_form(self) -> None:
        if self._current_project is None or not hasattr(self, "_inlet_velocity_x_input"):
            return
        settings = self._context.project_service.load_boundary_conditions(self._current_project)
        self._inlet_velocity_x_input.setValue(settings.inlet_velocity[0])
        self._inlet_velocity_y_input.setValue(settings.inlet_velocity[1])
        self._inlet_velocity_z_input.setValue(settings.inlet_velocity[2])
        self._outlet_pressure_input.setValue(settings.outlet_pressure)
        index = self._wall_type_combo.findData(settings.wall_type)
        if index >= 0:
            self._wall_type_combo.setCurrentIndex(index)

    def _refresh_domain_template_hint(self) -> None:
        if not hasattr(self, "_domain_template_combo") or not hasattr(self, "_domain_template_hint"):
            return
        key = self._domain_template_combo.currentData()
        template = next(
            (
                item
                for item in self._context.project_service.domain_templates()
                if item.key == key
            ),
            None,
        )
        if template is None:
            self._domain_template_hint.setText("请选择计算域模板。")
            self._update_domain_apply_state_label()
            self._refresh_domain_preview()
            return
        self._domain_template_hint.setText(
            f"{template.description}\n"
            f"尺寸：{template.size[0]:g} x {template.size[1]:g} x {template.size[2]:g}；"
            f"网格数：{template.cells[0]} x {template.cells[1]} x {template.cells[2]}；"
            "边界：左侧 inlet，右侧 outlet，其余 fixedWalls。\n"
            "建议 STL："
            + self._domain_template_stl_hint(template.key)
        )
        self._update_domain_apply_state_label()
        self._refresh_domain_preview()

    def _update_domain_apply_state_label(self) -> None:
        if not hasattr(self, "_domain_apply_state_label"):
            return
        if self._current_project is None:
            self._domain_apply_state_label.setText("预览中：尚未选择项目；选择项目后才能应用计算域。")
            self._domain_apply_state_label.setStyleSheet("color: #d7ba7d;")
            return
        selected = self._selected_domain_template()
        applied = self._current_domain_template()
        if applied.key == "custom_domain":
            self._domain_apply_state_label.setText(
                f"已应用：当前 Case 使用自定义计算域，尺寸={applied.size}，网格={applied.cells}。"
            )
            self._domain_apply_state_label.setStyleSheet("color: #89d185;")
            return
        if selected.key == applied.key:
            self._domain_apply_state_label.setText(
                f"已应用：当前 Case 正在使用 `{applied.name}`。切换下拉框只会先预览，不会自动改仿真文件。"
            )
            self._domain_apply_state_label.setStyleSheet("color: #89d185;")
            return
        self._domain_apply_state_label.setText(
            f"预览中：正在查看 `{selected.name}`；当前 Case 实际仍是 `{applied.name}`。"
            "需要点击 `应用计算域模板` 后才会写入 OpenFOAM case。"
        )
        self._domain_apply_state_label.setStyleSheet("color: #d7ba7d;")

    def _domain_template_stl_hint(self, key: str) -> str:
        if key == "simple_unit_box":
            return "small_obstacle_cube.stl 或 simple_center_cube.stl。"
        if key == "medium_wind_tunnel":
            return "medium_cylinder_obstacle.stl 或 medium_ramp_wedge.stl。"
        if key == "advanced_long_wind_tunnel":
            return "advanced_simplified_vehicle.stl。"
        if key == "medium_tapered_wind_tunnel":
            return "medium_cylinder_obstacle.stl，适合放在渐扩段中部。"
        if key == "advanced_ramp_channel":
            return "medium_ramp_wedge.stl，适合测试斜坡/地形通道。"
        if key == "medium_round_pipe":
            return "不建议放入障碍 STL；适合直接做管道内流。"
        if key == "advanced_90_bend_channel":
            return "不建议放入障碍 STL；适合观察弯管转弯流动。"
        return "请从 assets/test_geometries 选择匹配的 STL。"

    def _template_stl_path(self, key: str) -> Path:
        geometry_dir = self._context.project_root / "assets" / "test_geometries"
        if key == "simple_unit_box":
            return geometry_dir / "simple_center_cube.stl"
        if key == "medium_wind_tunnel":
            return geometry_dir / "medium_cylinder_obstacle.stl"
        if key == "advanced_long_wind_tunnel":
            return geometry_dir / "advanced_simplified_vehicle.stl"
        if key == "medium_tapered_wind_tunnel":
            return geometry_dir / "medium_cylinder_obstacle.stl"
        if key == "advanced_ramp_channel":
            return geometry_dir / "medium_ramp_wedge.stl"
        if key == "medium_round_pipe":
            return geometry_dir / "small_obstacle_cube.stl"
        if key == "advanced_90_bend_channel":
            return geometry_dir / "small_obstacle_cube.stl"
        return geometry_dir / "small_obstacle_cube.stl"

    def _apply_domain_template(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        key = str(self._domain_template_combo.currentData())
        try:
            template = self._context.project_service.apply_domain_template(self._current_project, key)
        except (OSError, ValueError) as error:
            self._show_error(f"应用计算域模板失败：{error}")
            return
        self._append_log(
            "计算域模板已应用："
            f"{template.name}，尺寸={template.size}，网格={template.cells}，"
            f"建议 locationInMesh={template.suggested_location_in_mesh}"
        )
        self._refresh_geometry_panel()
        self._snappy_location_x_input.setValue(template.suggested_location_in_mesh[0])
        self._snappy_location_y_input.setValue(template.suggested_location_in_mesh[1])
        self._snappy_location_z_input.setValue(template.suggested_location_in_mesh[2])
        self._update_domain_apply_state_label()
        self._set_status("计算域模板已应用。")

    def _apply_custom_domain(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        size = (
            self._domain_length_x_input.value(),
            self._domain_length_y_input.value(),
            self._domain_length_z_input.value(),
        )
        cells = (
            self._domain_cells_x_input.value(),
            self._domain_cells_y_input.value(),
            self._domain_cells_z_input.value(),
        )
        try:
            template = self._context.project_service.apply_custom_domain(self._current_project, size, cells)
        except (OSError, ValueError) as error:
            self._show_error(f"应用自定义计算域失败：{error}")
            return
        self._snappy_location_x_input.setValue(template.suggested_location_in_mesh[0])
        self._snappy_location_y_input.setValue(template.suggested_location_in_mesh[1])
        self._snappy_location_z_input.setValue(template.suggested_location_in_mesh[2])
        self._append_log(
            f"自定义计算域已应用：尺寸={template.size}，网格={template.cells}，"
            f"建议 locationInMesh={template.suggested_location_in_mesh}"
        )
        self._refresh_geometry_panel()
        self._update_domain_apply_state_label()
        self._set_status("自定义计算域已应用。")

    def _apply_boundary_conditions(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        settings = BoundaryConditionSettings(
            inlet_velocity=(
                self._inlet_velocity_x_input.value(),
                self._inlet_velocity_y_input.value(),
                self._inlet_velocity_z_input.value(),
            ),
            outlet_pressure=self._outlet_pressure_input.value(),
            wall_type=str(self._wall_type_combo.currentData()),
        )
        try:
            written_files = self._context.project_service.apply_boundary_conditions(self._current_project, settings)
        except (OSError, ValueError) as error:
            self._show_error(f"应用边界条件失败：{error}")
            return
        self._append_log(
            "边界条件已应用："
            f"inlet U={settings.inlet_velocity}，outlet p={settings.outlet_pressure:g}，"
            f"wall={settings.wall_type}"
        )
        for path in written_files:
            self._append_log(f"- 已写入：{path}")
        self._set_status("边界条件已应用。")

    def _run_preflight_check(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        template = self._current_domain_template()
        domain_points = np.array(self._context.project_service.domain_vertices(template), dtype=float)
        domain_min = domain_points.min(axis=0)
        domain_max = domain_points.max(axis=0)
        messages: list[str] = [
            "运行前检查",
            f"- 当前计算域：{template.name}，shape={template.shape}",
            f"- 计算域包围盒：min={tuple(domain_min.round(4))}, max={tuple(domain_max.round(4))}",
        ]
        has_error = False

        required_files = [
            self._current_project.case_dir / "system" / "blockMeshDict",
            self._current_project.case_dir / "system" / "controlDict",
            self._current_project.case_dir / "system" / "fvSchemes",
            self._current_project.case_dir / "system" / "fvSolution",
            self._current_project.case_dir / "0" / "U",
            self._current_project.case_dir / "0" / "p",
        ]
        for path in required_files:
            if path.exists():
                messages.append(f"- 必需文件存在：{path.relative_to(self._current_project.case_dir)}")
            else:
                has_error = True
                messages.append(f"- 缺少必需文件：{path.relative_to(self._current_project.case_dir)}")

        env_script = self._context.settings_service.load().openfoam_env_script
        if env_script and Path(env_script).exists():
            messages.append(f"- OpenFOAM 环境脚本存在：{env_script}")
        else:
            has_error = True
            messages.append(f"- OpenFOAM 环境脚本不存在：{env_script or '未配置'}")

        boundary_names = self._context.project_service._extract_boundary_names(
            self._current_project.case_dir / "system" / "blockMeshDict"
        )
        if boundary_names:
            messages.append(f"- blockMesh 边界：{', '.join(boundary_names)}")
            for field_name in ("U", "p"):
                field_path = self._current_project.case_dir / "0" / field_name
                missing = [
                    name
                    for name in boundary_names
                    if not re.search(rf"\b{re.escape(name)}\s*\{{", field_path.read_text(encoding="utf-8"))
                ] if field_path.exists() else list(boundary_names)
                if missing:
                    has_error = True
                    messages.append(f"- {field_name} 缺少边界条件：{', '.join(missing)}")
                else:
                    messages.append(f"- {field_name} 边界条件完整。")
        else:
            has_error = True
            messages.append("- blockMeshDict 没有读取到边界定义。")

        location = np.array(
            [
                self._snappy_location_x_input.value(),
                self._snappy_location_y_input.value(),
                self._snappy_location_z_input.value(),
            ],
            dtype=float,
        )
        if np.all((location >= domain_min) & (location <= domain_max)):
            messages.append(f"- locationInMesh 合理：{tuple(location.round(4))} 在计算域包围盒内。")
        else:
            has_error = True
            messages.append(f"- locationInMesh 不合理：{tuple(location.round(4))} 不在计算域包围盒内。")

        assets = [
            asset
            for asset in self._context.geometry_import_service.list_assets(self._current_project)
            if asset.format.upper() == "STL" and asset.stored_path.exists()
        ]
        if not assets:
            messages.append("- 当前 Case 没有导入 STL。管道内流可以不放障碍物；外流/绕流场景建议导入 STL。")
        elif template.shape in {"pipe", "bend"}:
            messages.append("- 当前是管道/弯管内流模板：通常不需要导入障碍 STL，除非你明确要模拟管内障碍物。")
        asset_bounds: list[tuple[str, np.ndarray, np.ndarray]] = []
        for asset in assets:
            try:
                points, _faces = self._read_stl_preview_mesh(asset.stored_path)
            except (OSError, ValueError) as error:
                has_error = True
                messages.append(f"- STL 读取失败：{asset.name}，{error}")
                continue
            if points.size == 0:
                has_error = True
                messages.append(f"- STL 无顶点：{asset.name}")
                continue
            stl_min = points.min(axis=0)
            stl_max = points.max(axis=0)
            inside = np.all((stl_min >= domain_min) & (stl_max <= domain_max))
            if inside:
                messages.append(f"- STL 在计算域内：{asset.name}，bbox={tuple(stl_min.round(4))} -> {tuple(stl_max.round(4))}")
            else:
                has_error = True
                messages.append(
                    f"- STL 可能超出计算域：{asset.name}，bbox={tuple(stl_min.round(4))} -> {tuple(stl_max.round(4))}"
                )
            asset_bounds.append((asset.name, stl_min, stl_max))
            clearance = np.minimum(stl_min - domain_min, domain_max - stl_max)
            nearest_clearance = float(clearance.min())
            if nearest_clearance >= 0.0:
                messages.append(f"- STL 到最近计算域边界距离：{asset.name}，约 {nearest_clearance:.4g}")
                domain_span = float((domain_max - domain_min).max())
                if domain_span > 0.0 and nearest_clearance < domain_span * 0.01:
                    messages.append(
                        f"- 提醒：{asset.name} 离计算域边界很近，snappyHexMesh 或求解时可能受边界影响。"
                    )
            if np.all((location >= stl_min) & (location <= stl_max)):
                has_error = True
                messages.append(
                    f"- locationInMesh 可能落在 STL 固体内部：{asset.name}。"
                    "该点必须放在流体区域里，不能放在物体内部。"
                )

        for index, (name_a, min_a, max_a) in enumerate(asset_bounds):
            for name_b, min_b, max_b in asset_bounds[index + 1:]:
                overlap = np.all((min_a <= max_b) & (max_a >= min_b))
                if overlap:
                    messages.append(
                        f"- 提醒：STL 包围盒可能重叠：{name_a} 与 {name_b}。"
                        "如果不是故意组合几何，建议调整位置避免贴体网格失败。"
                    )
                else:
                    gap_vector = np.maximum(np.maximum(min_a - max_b, min_b - max_a), 0.0)
                    distance = float(np.linalg.norm(gap_vector))
                    messages.append(f"- STL 间最近包围盒距离：{name_a} <-> {name_b}，约 {distance:.4g}")

        if self._snappy_min_refinement_input.value() > self._snappy_max_refinement_input.value():
            has_error = True
            messages.append("- snappy 加密等级不合理：最小加密等级不能大于最大加密等级。")
        else:
            messages.append(
                f"- snappy 加密等级：{self._snappy_min_refinement_input.value()} -> {self._snappy_max_refinement_input.value()}"
            )

        summary = "\n".join(messages)
        self._problem_text.setPlainText(summary)
        self._append_log(summary)
        if has_error:
            self._show_error("运行前检查发现问题，请查看“问题”面板。")
            self._set_status("运行前检查发现问题。")
        else:
            QMessageBox.information(self, "运行前检查通过", "未发现明显问题，可以继续生成网格或运行仿真。")
            self._set_status("运行前检查通过。")

    def _import_template_stl(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        key = str(self._domain_template_combo.currentData())
        stl_path = self._template_stl_path(key)
        if not stl_path.exists():
            self._show_error(f"模板 STL 不存在：{stl_path}")
            return
        try:
            template = self._context.project_service.apply_domain_template(self._current_project, key)
            asset = self._context.geometry_import_service.import_stl(self._current_project, stl_path)
        except (OSError, ValueError) as error:
            self._show_error(f"一键导入模板 STL 失败：{error}")
            return
        self._snappy_location_x_input.setValue(template.suggested_location_in_mesh[0])
        self._snappy_location_y_input.setValue(template.suggested_location_in_mesh[1])
        self._snappy_location_z_input.setValue(template.suggested_location_in_mesh[2])
        self._append_log(f"模板 STL 已导入：{asset.name}")
        self._append_log(f"已同步计算域模板：{template.name}")
        self._refresh_geometry_panel()
        self._set_status("模板 STL 已导入。")

    def _edit_imported_stl_transform(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        assets = [
            asset
            for asset in self._context.geometry_import_service.list_assets(self._current_project)
            if asset.format.upper() == "STL" and asset.stored_path.exists()
        ]
        if not assets:
            self._show_error("当前 Case 没有可编辑的 STL，请先导入 STL。")
            return
        selected_name, ok = QInputDialog.getItem(
            self,
            "编辑 STL 位置",
            "选择要编辑的 STL",
            [asset.name for asset in assets],
            0,
            False,
        )
        if not ok or not selected_name:
            return
        selected_asset = next(asset for asset in assets if asset.name == selected_name)
        base_path = Path(selected_asset.source_path)
        if not base_path.exists():
            base_path = selected_asset.stored_path
        transform = self._read_stl_transform_dialog(base_path, selected_asset.transform or StlTransform())
        if transform is None:
            return
        try:
            updated_asset = self._context.geometry_import_service.update_stl_transform(
                self._current_project,
                selected_asset.name,
                transform,
            )
        except (OSError, ValueError) as error:
            self._show_error(f"编辑 STL 位置失败：{error}")
            return
        self._append_log(
            f"STL 位置已更新：{updated_asset.name}，"
            f"scale={transform.scale:g}，translate={transform.translate}"
        )
        self._refresh_geometry_panel()
        self._set_status("STL 位置已更新。")

    def _refresh_domain_preview(self) -> None:
        if not hasattr(self, "_domain_preview_figure") or not hasattr(self, "_domain_preview_canvas"):
            return
        self._domain_preview_figure.clear()
        axes = self._domain_preview_figure.add_subplot(111, projection="3d", facecolor="#1e1e1e")
        axes.set_title("Domain and STL Position Preview", color="#d4d4d4", pad=10)
        axes.set_xlabel("X", color="#d4d4d4")
        axes.set_ylabel("Y", color="#d4d4d4")
        axes.set_zlabel("Z", color="#d4d4d4")
        axes.tick_params(colors="#d4d4d4")
        axes.grid(True, color="#333333", linestyle="--", linewidth=0.5)

        template = self._selected_domain_template()
        domain_points = self._draw_domain_template_wireframe(axes, template)
        points_for_limits = [domain_points]
        if self._current_project is not None:
            for asset in self._context.geometry_import_service.list_assets(self._current_project):
                if asset.format.upper() != "STL" or not asset.stored_path.exists():
                    continue
                try:
                    points, faces = self._read_stl_preview_mesh(asset.stored_path)
                except (OSError, ValueError):
                    continue
                if points.size == 0 or faces.size == 0:
                    continue
                sampled_faces = faces
                if len(sampled_faces) > 3000:
                    indices = np.linspace(0, len(sampled_faces) - 1, 3000, dtype=int)
                    sampled_faces = sampled_faces[indices]
                polygons = [points[np.asarray(face, dtype=int)] for face in sampled_faces]
                collection = Poly3DCollection(
                    polygons,
                    facecolors=(0.25, 0.74, 1.0, 0.42),
                    edgecolors=(0.03, 0.18, 0.28, 0.35),
                    linewidths=0.2,
                )
                axes.add_collection3d(collection)
                points_for_limits.append(points)
        else:
            axes.text2D(0.05, 0.9, "Select a project first.", color="#d4d4d4")

        all_points = np.vstack(points_for_limits)
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = max(float((maxs - mins).max()) / 2.0, 0.55)
        axes.set_xlim(center[0] - radius, center[0] + radius)
        axes.set_ylim(center[1] - radius, center[1] + radius)
        axes.set_zlim(center[2] - radius, center[2] + radius)
        axes.view_init(elev=24, azim=-55)
        self._domain_preview_canvas.draw()

    def _current_domain_template(self) -> ComputationDomainTemplate:
        if self._current_project is not None:
            config_path = self._current_project.case_dir / "system" / "domain_config.json"
            if config_path.exists():
                try:
                    payload = json.loads(config_path.read_text(encoding="utf-8"))
                    if payload.get("key") == "custom_domain":
                        size = payload.get("size", [1.0, 1.0, 1.0])
                        cells = payload.get("cells", [10, 10, 10])
                        return ComputationDomainTemplate(
                            key="custom_domain",
                            name="自定义计算域",
                            level="自定义",
                            size=(float(size[0]), float(size[1]), float(size[2])),
                            cells=(int(cells[0]), int(cells[1]), int(cells[2])),
                            suggested_location_in_mesh=(
                                float(size[0]) * 0.1,
                                float(size[1]) * 0.5,
                                float(size[2]) * 0.5,
                            ),
                            description="用户自定义计算域。",
                        )
                    key = str(payload.get("key", "simple_unit_box"))
                    for template in self._context.project_service.domain_templates():
                        if template.key == key:
                            return template
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    pass
        key = None
        if hasattr(self, "_domain_template_combo"):
            key = self._domain_template_combo.currentData()
        for template in self._context.project_service.domain_templates():
            if template.key == key:
                return template
        return self._context.project_service.domain_templates()[0]

    def _selected_domain_template(self) -> ComputationDomainTemplate:
        if hasattr(self, "_domain_template_combo"):
            key = self._domain_template_combo.currentData()
            for template in self._context.project_service.domain_templates():
                if template.key == key:
                    return template
        return self._current_domain_template()

    def _load_snappy_settings_into_form(self) -> None:
        if self._current_project is None or not hasattr(self, "_snappy_min_refinement_input"):
            return
        settings = self._context.geometry_import_service.load_snappy_settings(self._current_project)
        if settings is None:
            return
        self._snappy_min_refinement_input.setValue(settings.min_refinement_level)
        self._snappy_max_refinement_input.setValue(settings.max_refinement_level)
        self._snappy_location_x_input.setValue(settings.location_in_mesh[0])
        self._snappy_location_y_input.setValue(settings.location_in_mesh[1])
        self._snappy_location_z_input.setValue(settings.location_in_mesh[2])
        self._snappy_add_layers_checkbox.setChecked(settings.add_layers)
        self._snappy_layer_thickness_input.setValue(settings.final_layer_thickness)

    def _read_snappy_settings(self) -> SnappyHexMeshSettings:
        min_level = self._snappy_min_refinement_input.value()
        max_level = self._snappy_max_refinement_input.value()
        if max_level < min_level:
            max_level = min_level
            self._snappy_max_refinement_input.setValue(max_level)
        return SnappyHexMeshSettings(
            min_refinement_level=min_level,
            max_refinement_level=max_level,
            location_in_mesh=(
                self._snappy_location_x_input.value(),
                self._snappy_location_y_input.value(),
                self._snappy_location_z_input.value(),
            ),
            add_layers=self._snappy_add_layers_checkbox.isChecked(),
            final_layer_thickness=self._snappy_layer_thickness_input.value(),
        )

    def _generate_snappy_hex_mesh_dict(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        assets = self._context.geometry_import_service.list_assets(self._current_project)
        stl_assets = [asset for asset in assets if asset.format.upper() == "STL" and asset.stored_path.exists()]
        if not stl_assets:
            self._show_error("当前 Case 没有可生成网格配置的 STL，请先导入 STL。")
            return

        selected_name = stl_assets[0].name
        if len(stl_assets) > 1:
            names = [asset.name for asset in stl_assets]
            selected_name, ok = QInputDialog.getItem(
                self,
                "生成 snappyHexMeshDict",
                "选择用于 snappyHexMesh 的 STL",
                names,
                0,
                False,
            )
            if not ok or not selected_name:
                return

        try:
            dict_path = self._context.geometry_import_service.generate_snappy_hex_mesh_dict(
                self._current_project,
                selected_name,
                self._read_snappy_settings(),
            )
        except (OSError, ValueError) as error:
            self._show_error(f"生成 snappyHexMeshDict 失败：{error}")
            return

        self._append_log(f"snappyHexMeshDict 已生成：{dict_path}")
        self._append_log(
            "snappy 参数："
            f"level=({self._snappy_min_refinement_input.value()} {self._snappy_max_refinement_input.value()}), "
            f"locationInMesh=({self._snappy_location_x_input.value():.4g} "
            f"{self._snappy_location_y_input.value():.4g} "
            f"{self._snappy_location_z_input.value():.4g}), "
            f"addLayers={self._snappy_add_layers_checkbox.isChecked()}"
        )
        self._refresh_geometry_panel()
        self._workspace_tabs.setCurrentIndex(6)
        self._set_status("snappyHexMeshDict 生成完成。")

    def _select_stl_asset_for_snappy(self) -> str | None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return None
        assets = self._context.geometry_import_service.list_assets(self._current_project)
        stl_assets = [asset for asset in assets if asset.format.upper() == "STL" and asset.stored_path.exists()]
        if not stl_assets:
            self._show_error("当前 Case 没有可用于 snappyHexMesh 的 STL，请先导入 STL。")
            return None
        if len(stl_assets) == 1:
            return stl_assets[0].name

        names = [asset.name for asset in stl_assets]
        selected_name, ok = QInputDialog.getItem(
            self,
            "选择 STL",
            "选择用于 snappyHexMesh 的 STL",
            names,
            0,
            False,
        )
        if not ok or not selected_name:
            return None
        return selected_name

    def _run_snappy_hex_mesh(self) -> None:
        if self._foam_process and self._foam_process.state() != QProcess.ProcessState.NotRunning:
            self._show_error("已有任务正在运行，请先停止当前任务。")
            return
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return

        status = self._context.environment_detector.detect()
        if not status.is_available or not status.env_script_path:
            self._show_error(f"OpenFOAM 环境不可用：{status.detail}")
            return

        snappy_dict = self._current_project.case_dir / "system" / "snappyHexMeshDict"
        if not snappy_dict.exists():
            try:
                snappy_dict = self._context.geometry_import_service.generate_snappy_hex_mesh_dict(self._current_project)
            except (OSError, ValueError) as error:
                self._show_error(f"当前 Case 不能运行 snappyHexMesh：{error}")
                return
            self._append_log(f"缺少 snappyHexMeshDict，已自动生成：{snappy_dict}")

        block_mesh_dict = self._current_project.case_dir / "system" / "blockMeshDict"
        if not block_mesh_dict.exists():
            self._show_error("当前 Case 缺少 system/blockMeshDict，snappyHexMesh 需要先有背景网格。")
            return

        self._workspace_tabs.setCurrentIndex(6)
        self._bottom_tabs.setCurrentIndex(0)
        self._task_text.setPlainText("任务状态：snappyHexMesh 运行中")
        self._current_process_output = ""
        self._last_diagnostic_summary = "snappyHexMesh 正在运行，暂无失败诊断。"
        self._active_process_kind = "snappyHexMesh"
        self._refresh_solver_run_panel("snappyHexMesh 运行中")
        self._set_status("snappyHexMesh 运行中。")

        command = (
            f"source {shlex.quote(status.env_script_path)} >/dev/null 2>&1 && "
            f"cd {shlex.quote(str(self._current_project.case_dir))} && "
            "blockMesh && snappyHexMesh -overwrite"
        )
        self._foam_process = QProcess(self)
        self._foam_process.setProgram("bash")
        self._foam_process.setArguments(["-lc", command])
        self._foam_process.readyReadStandardOutput.connect(self._read_process_stdout)
        self._foam_process.readyReadStandardError.connect(self._read_process_stderr)
        self._foam_process.finished.connect(self._on_process_finished)
        self._foam_process.start()
        self._append_log(f"启动 snappyHexMesh：{self._current_project.case_dir}")
        self._append_log("执行流程：blockMesh -> snappyHexMesh -overwrite")

    def _run_check_mesh(self) -> None:
        if self._foam_process and self._foam_process.state() != QProcess.ProcessState.NotRunning:
            self._show_error("已有任务正在运行，请先停止当前任务。")
            return
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return

        status = self._context.environment_detector.detect()
        if not status.is_available or not status.env_script_path:
            self._show_error(f"OpenFOAM 环境不可用：{status.detail}")
            return

        mesh_dir = self._current_project.case_dir / "constant" / "polyMesh"
        if not mesh_dir.exists():
            self._show_error("当前 Case 还没有网格，请先运行 blockMesh 或 snappyHexMesh。")
            return

        self._workspace_tabs.setCurrentIndex(6)
        self._bottom_tabs.setCurrentIndex(0)
        self._task_text.setPlainText("任务状态：checkMesh 运行中")
        self._current_process_output = ""
        self._last_diagnostic_summary = "checkMesh 正在运行，暂无失败诊断。"
        self._active_process_kind = "checkMesh"
        self._refresh_solver_run_panel("checkMesh 运行中")
        self._set_status("checkMesh 运行中。")

        command = (
            f"source {shlex.quote(status.env_script_path)} >/dev/null 2>&1 && "
            f"cd {shlex.quote(str(self._current_project.case_dir))} && "
            "checkMesh"
        )
        self._foam_process = QProcess(self)
        self._foam_process.setProgram("bash")
        self._foam_process.setArguments(["-lc", command])
        self._foam_process.readyReadStandardOutput.connect(self._read_process_stdout)
        self._foam_process.readyReadStandardError.connect(self._read_process_stderr)
        self._foam_process.finished.connect(self._on_process_finished)
        self._foam_process.start()
        self._append_log(f"启动 checkMesh：{self._current_project.case_dir}")
        self._append_log("执行流程：checkMesh")

    def _run_preprocess_pipeline(self) -> None:
        if self._foam_process and self._foam_process.state() != QProcess.ProcessState.NotRunning:
            self._show_error("已有任务正在运行，请先停止当前任务。")
            return
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return

        status = self._context.environment_detector.detect()
        if not status.is_available or not status.env_script_path:
            self._show_error(f"OpenFOAM 环境不可用：{status.detail}")
            return

        selected_name = self._select_stl_asset_for_snappy()
        if not selected_name:
            return

        block_mesh_dict = self._current_project.case_dir / "system" / "blockMeshDict"
        if not block_mesh_dict.exists():
            self._show_error("当前 Case 缺少 system/blockMeshDict，一键前处理需要先有背景网格配置。")
            return

        try:
            dict_path = self._context.geometry_import_service.generate_snappy_hex_mesh_dict(
                self._current_project,
                selected_name,
                self._read_snappy_settings(),
            )
        except (OSError, ValueError) as error:
            self._show_error(f"一键前处理无法生成 snappyHexMeshDict：{error}")
            return

        self._workspace_tabs.setCurrentIndex(6)
        self._bottom_tabs.setCurrentIndex(0)
        self._task_text.setPlainText("任务状态：一键前处理运行中")
        self._current_process_output = ""
        self._last_diagnostic_summary = "一键前处理正在运行，暂无失败诊断。"
        self._active_process_kind = "preprocess"
        self._refresh_solver_run_panel("一键前处理运行中")
        self._set_status("一键前处理运行中。")
        self._refresh_geometry_panel()

        command = (
            f"source {shlex.quote(status.env_script_path)} >/dev/null 2>&1 && "
            f"cd {shlex.quote(str(self._current_project.case_dir))} && "
            "echo FOAMDESK_STEP:blockMesh && blockMesh && "
            "echo FOAMDESK_STEP:snappyHexMesh && snappyHexMesh -overwrite && "
            "echo FOAMDESK_STEP:checkMesh && checkMesh"
        )
        self._foam_process = QProcess(self)
        self._foam_process.setProgram("bash")
        self._foam_process.setArguments(["-lc", command])
        self._foam_process.readyReadStandardOutput.connect(self._read_process_stdout)
        self._foam_process.readyReadStandardError.connect(self._read_process_stderr)
        self._foam_process.finished.connect(self._on_process_finished)
        self._foam_process.start()
        self._append_log(f"一键前处理已生成配置：{dict_path}")
        self._append_log("执行流程：生成 snappyHexMeshDict -> blockMesh -> snappyHexMesh -overwrite -> checkMesh")

    def _run_simulation_pipeline(self) -> None:
        if self._foam_process and self._foam_process.state() != QProcess.ProcessState.NotRunning:
            self._show_error("已有任务正在运行，请先停止当前任务。")
            return
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return

        status = self._context.environment_detector.detect()
        if not status.is_available or not status.env_script_path:
            self._show_error(f"OpenFOAM 环境不可用：{status.detail}")
            return

        if not self._save_case_parameters():
            return

        selected_name = self._select_stl_asset_for_snappy()
        if not selected_name:
            return

        block_mesh_dict = self._current_project.case_dir / "system" / "blockMeshDict"
        if not block_mesh_dict.exists():
            self._show_error("当前 Case 缺少 system/blockMeshDict，一键仿真需要先有背景网格配置。")
            return

        try:
            dict_path = self._context.geometry_import_service.generate_snappy_hex_mesh_dict(
                self._current_project,
                selected_name,
                self._read_snappy_settings(),
            )
            synced_fields = self._context.project_service.ensure_field_boundaries(
                self._current_project,
                ("movingWall", "fixedWalls", "importedGeometry"),
            )
        except (OSError, ValueError) as error:
            self._show_error(f"一键仿真准备失败：{error}")
            return

        self._workspace_tabs.setCurrentIndex(2)
        self._bottom_tabs.setCurrentIndex(0)
        self._task_text.setPlainText("任务状态：一键仿真流水线运行中")
        self._current_process_output = ""
        self._last_diagnostic_summary = "一键仿真流水线正在运行，暂无失败诊断。"
        self._active_process_kind = "simulationPipeline"
        self._refresh_solver_run_panel("一键仿真流水线运行中")
        self._set_status("一键仿真流水线运行中。")
        self._refresh_geometry_panel()
        if synced_fields:
            relative_fields = [str(path.relative_to(self._current_project.case_dir)) for path in synced_fields]
            self._append_log("已同步 snappy 后求解边界字段：")
            self._append_log("\n".join(f"- {path}" for path in relative_fields))

        command = (
            f"source {shlex.quote(status.env_script_path)} >/dev/null 2>&1 && "
            f"cd {shlex.quote(str(self._current_project.case_dir))} && "
            "echo FOAMDESK_STEP:blockMesh && blockMesh && "
            "echo FOAMDESK_STEP:snappyHexMesh && snappyHexMesh -overwrite && "
            "echo FOAMDESK_STEP:checkMesh && checkMesh && "
            "echo FOAMDESK_STEP:icoFoam && icoFoam"
        )
        self._foam_process = QProcess(self)
        self._foam_process.setProgram("bash")
        self._foam_process.setArguments(["-lc", command])
        self._foam_process.readyReadStandardOutput.connect(self._read_process_stdout)
        self._foam_process.readyReadStandardError.connect(self._read_process_stderr)
        self._foam_process.finished.connect(self._on_process_finished)
        self._foam_process.start()
        self._append_log(f"一键仿真已生成配置：{dict_path}")
        self._append_log(
            "执行流程：生成 snappyHexMeshDict -> blockMesh -> snappyHexMesh -overwrite -> checkMesh -> icoFoam"
        )

    def _preview_imported_stl(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        assets = self._context.geometry_import_service.list_assets(self._current_project)
        stl_assets = [asset for asset in assets if asset.format.upper() == "STL" and asset.stored_path.exists()]
        if not stl_assets:
            self._show_error("当前 Case 没有可预览的 STL，请先导入 STL。")
            return
        names = [asset.name for asset in stl_assets]
        selected_name, ok = QInputDialog.getItem(self, "预览 STL", "选择 STL", names, 0, False)
        if not ok or not selected_name:
            return
        selected_asset = next(asset for asset in stl_assets if asset.name == selected_name)
        self._ensure_vtk_viewer()
        try:
            point_count, face_count = self._vtk_viewer.plot_stl_file(selected_asset.stored_path)
        except RuntimeError as error:
            self._show_error(f"预览 STL 失败：{error}")
            return
        self._append_log(
            f"STL 预览已加载：name={selected_asset.name}, points={point_count}, faces={face_count}"
        )
        self._set_status("STL 预览已加载。")

    def _show_cad_import_limitations(self) -> None:
        QMessageBox.information(
            self,
            "STEP/IGES 支持说明",
            "当前 Sprint 28 先完成 STL 导入 MVP。\n\n"
            "原因：STL 是 OpenFOAM triSurface 最直接支持的几何格式，适合先打通流程。\n\n"
            "STEP、IGES、CATIA、SolidWorks 属于 CAD B-Rep/装配模型，"
            "需要后续接入 OCCT/CAD 内核做读取、修复、三角化和单位处理。",
        )

    def _set_parameter_inputs_enabled(self, enabled: bool) -> None:
        if not hasattr(self, "_end_time_input"):
            return
        self._end_time_input.setEnabled(enabled)
        self._delta_t_input.setEnabled(enabled)
        self._write_interval_input.setEnabled(enabled)
        self._viscosity_input.setEnabled(enabled)

    def _show_parameters(self, parameters: SimulationParameters) -> None:
        self._end_time_input.setText(f"{parameters.end_time:.12g}")
        self._delta_t_input.setText(f"{parameters.delta_t:.12g}")
        self._write_interval_input.setValue(parameters.write_interval)
        self._viscosity_input.setText(f"{parameters.viscosity:.12g}")

    def _read_parameter_inputs(self) -> SimulationParameters:
        return SimulationParameters(
            end_time=float(self._end_time_input.text().strip()),
            delta_t=float(self._delta_t_input.text().strip()),
            write_interval=self._write_interval_input.value(),
            viscosity=float(self._viscosity_input.text().strip()),
        )

    def _load_case_parameters(self) -> None:
        if self._current_project is None:
            self._parameter_status_label.setText("请先新建或打开项目。")
            self._set_parameter_inputs_enabled(False)
            return

        try:
            self._context.project_service.ensure_minimal_case_template(self._current_project)
            parameters = self._context.case_parameter_service.load(self._current_project)
        except (OSError, ValueError) as error:
            self._show_error(f"加载参数失败：{error}")
            return

        self._show_parameters(parameters)
        self._set_parameter_inputs_enabled(True)
        self._parameter_status_label.setText(f"已加载项目参数：{self._current_project.name}")
        self._refresh_solver_run_panel()
        self._set_status("参数已加载。")

    def _save_case_parameters(self) -> bool:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return False

        try:
            parameters = self._read_parameter_inputs()
            self._context.project_service.ensure_minimal_case_template(self._current_project)
            self._context.case_parameter_service.save(self._current_project, parameters)
        except ValueError as error:
            self._show_error(f"参数不合法：{error}")
            return False
        except OSError as error:
            self._show_error(f"保存参数失败：{error}")
            return False

        self._parameter_status_label.setText(
            f"参数已保存到 Case：endTime={parameters.end_time:g}, "
            f"deltaT={parameters.delta_t:g}, writeInterval={parameters.write_interval}, "
            f"nu={parameters.viscosity:g}"
        )
        self._refresh_solver_run_panel()
        self._append_log("参数已写入 system/controlDict 和 constant/physicalProperties。")
        self._set_status("参数保存完成。")
        return True

    def _restore_default_parameters(self) -> None:
        parameters = self._context.case_parameter_service.defaults()
        self._show_parameters(parameters)
        self._parameter_status_label.setText("已恢复默认参数，点击“保存参数到 Case”后生效。")
        self._set_parameter_inputs_enabled(self._current_project is not None)
        self._refresh_solver_run_panel()

    def _refresh_solver_run_panel(self, status_text: str | None = None) -> None:
        if not hasattr(self, "_solver_status_label"):
            return

        self._solver_status_label.setText(f"状态：{status_text or '空闲'}")
        self._refresh_solver_metrics_panel()
        if hasattr(self, "_solver_diagnostic_text"):
            self._solver_diagnostic_text.setPlainText(f"最近诊断：\n{self._last_diagnostic_summary}")
        if self._current_project is None:
            self._solver_project_label.setText("当前项目：未选择")
            self._solver_case_path_label.setText("Case 路径：未选择")
            self._solver_parameter_summary.setPlainText("参数摘要：请先新建或打开项目。")
            return

        self._solver_project_label.setText(f"当前项目：{self._current_project.name}")
        self._solver_case_path_label.setText(f"Case 路径：{self._current_project.case_dir}")
        try:
            parameters = self._context.case_parameter_service.load(self._current_project)
        except (OSError, ValueError) as error:
            self._solver_parameter_summary.setPlainText(f"参数摘要加载失败：{error}")
            return

        self._solver_parameter_summary.setPlainText(
            "参数摘要：\n"
            f"- endTime：{parameters.end_time:g}\n"
            f"- deltaT：{parameters.delta_t:g}\n"
            f"- writeInterval：{parameters.write_interval}\n"
            f"- nu：{parameters.viscosity:g}\n"
            "\n"
            "输出位置：当前 Case 目录下的时间步目录、constant/polyMesh 和日志面板。"
        )

    def _refresh_solver_metrics_panel(self) -> None:
        if not hasattr(self, "_solver_metric_summary"):
            return
        if not self._current_process_output.strip():
            self._solver_metric_summary.setPlainText("关键指标摘要：尚未运行。")
            return
        metrics = self._context.log_metric_service.parse(self._current_process_output)
        self._solver_metric_summary.setPlainText(
            self._context.log_metric_service.format_summary(metrics)
        )

    def _refresh_results_panel(self) -> None:
        if not hasattr(self, "_results_text"):
            return
        if self._current_project is None:
            self._results_text.setPlainText("请先新建或打开项目。")
            self._set_status("结果索引刷新失败：未选择项目。")
            return

        try:
            result_index = self._context.result_index_service.index(self._current_project)
        except OSError as error:
            self._show_error(f"刷新结果索引失败：{error}")
            return

        self._results_text.setPlainText(self._context.result_index_service.format_index(result_index))
        self._refresh_visualization_selectors(show_errors=False)
        self._append_log("结果索引已刷新。")
        self._set_status("结果索引已刷新。")

    def _export_solver_metrics(self) -> bool:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return False
        if not self._current_process_output.strip():
            self._show_error("当前没有可导出的求解日志指标，请先运行一次最小仿真。")
            return False

        metrics = self._context.log_metric_service.parse(self._current_process_output)
        if not metrics.times and not metrics.residuals:
            self._show_error("未从当前日志中识别到可导出的求解指标。")
            return False

        try:
            json_path, csv_path = self._context.metric_export_service.export(
                self._current_project,
                metrics,
            )
        except OSError as error:
            self._show_error(f"导出求解指标失败：{error}")
            return False

        self._append_log(f"求解指标已导出：{json_path}")
        self._append_log(f"残差 CSV 已导出：{csv_path}")
        self._set_status("求解指标导出完成。")
        return True

    def _export_markdown_report(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        try:
            result_index = self._context.result_index_service.index(self._current_project)
            try:
                vtk_info = self._context.openfoam_vtk_service.inspect(self._current_project)
            except (OSError, RuntimeError):
                vtk_info = None
            asset_paths = self._export_report_assets()
            report_path = self._context.report_export_service.export_markdown(
                self._current_project,
                result_index,
                vtk_info,
                asset_paths,
            )
        except OSError as error:
            self._show_error(f"导出 Markdown 报告失败：{error}")
            return
        self._append_log(f"Markdown 报告已导出：{report_path}")
        self._results_text.setPlainText(
            "Markdown 报告已导出\n\n"
            f"路径：{report_path}\n\n"
            "当前报告包含：项目/Case、结果索引、VTK 字段、求解指标、残差数据、可视化能力说明和自动嵌入图片。"
        )
        self._set_status("Markdown 报告导出完成。")

    def _export_report_assets(self) -> list[Path]:
        if self._current_project is None:
            return []
        assets_dir = self._current_project.case_dir / "foamdesk_results" / "report_assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        asset_paths: list[Path] = []
        residual_path = self._export_residual_report_asset(assets_dir)
        if residual_path is not None:
            asset_paths.append(residual_path)
        if self._vtk_viewer is not None:
            asset_paths.extend(self._vtk_viewer.export_all_pngs(assets_dir))
        if asset_paths:
            self._append_log(f"报告图片已导出：{len(asset_paths)} 个")
        return asset_paths

    def _export_residual_report_asset(self, assets_dir: Path) -> Path | None:
        if self._current_project is None:
            return None
        try:
            series = self._context.residual_plot_service.load_series(self._current_project)
        except (OSError, ValueError):
            return None
        figure = Figure(figsize=(8, 3.6), tight_layout=True)
        axes = figure.add_subplot(111)
        for field, points in series.items():
            points = sorted(points, key=lambda item: item[0])
            axes.plot(
                [time for time, _residual in points],
                [residual for _time, residual in points],
                marker="o",
                linewidth=1.4,
                markersize=3,
                label=field,
            )
        axes.set_title("Residual Curve")
        axes.set_xlabel("Time")
        axes.set_ylabel("Final residual")
        axes.set_yscale("log")
        axes.grid(True, which="both", linestyle="--", alpha=0.35)
        axes.legend(loc="best")
        path = assets_dir / "residual_curve.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        return path

    def _plot_residual_curve(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return

        try:
            series = self._context.residual_plot_service.load_series(self._current_project)
        except (OSError, ValueError) as error:
            self._show_error(f"绘制残差曲线失败：{error}")
            return

        self._residual_figure.clear()
        axes = self._residual_figure.add_subplot(111)
        for field, points in series.items():
            points = sorted(points, key=lambda item: item[0])
            axes.plot(
                [time for time, _residual in points],
                [residual for _time, residual in points],
                marker="o",
                linewidth=1.4,
                markersize=3,
                label=field,
            )
        axes.set_title("Residual Curve")
        axes.set_xlabel("Time")
        axes.set_ylabel("Final residual")
        axes.set_yscale("log")
        axes.grid(True, which="both", linestyle="--", alpha=0.35)
        axes.legend(loc="best")
        self._residual_canvas.draw()
        self._append_log("残差曲线已绘制。")
        self._set_status("残差曲线已绘制。")

    def _load_3d_preview_scene(self) -> None:
        self._ensure_vtk_viewer()

        self._show_visualization_feedback(
            "正在加载 3D 技术验证场景",
            "该场景不依赖 OpenFOAM 项目，用于验证三维窗口是否能正常渲染。",
        )
        self._vtk_viewer.plot_cube()
        self._append_log("3D 技术验证场景已加载：单位计算域 + 坐标轴。")
        self._show_visualization_feedback(
            "3D 技术验证场景已加载",
            "你应该能在独立 3D 视图窗口中看到蓝色线框立方体和坐标轴。"
        )
        self._set_status("3D 技术验证场景已加载。")

    def _load_openfoam_3d_case(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        self._ensure_vtk_viewer()

        self._show_visualization_feedback(
            "正在加载真实 OpenFOAM 3D Case",
            f"当前项目：{self._current_project.name}\nCase 路径：{self._current_project.case_dir}",
        )
        try:
            case_info = self._context.openfoam_vtk_service.inspect(self._current_project)
            geometry = self._context.openfoam_vtk_service.build_geometry_filter(self._current_project)
        except (OSError, RuntimeError) as error:
            self._show_error(f"加载 OpenFOAM 3D Case 失败：{error}")
            return

        self._vtk_viewer.plot_polydata_points(geometry.GetOutput(), "真实 OpenFOAM Case 点云")
        self._append_log(
            "真实 OpenFOAM 3D Case 已加载："
            f"blocks={case_info.block_count}, times={len(case_info.time_values)}, "
            f"pointArrays={case_info.point_arrays}, cellArrays={case_info.cell_arrays}, "
            f"marker={case_info.marker_file}"
        )
        self._show_visualization_feedback(
            "真实 OpenFOAM 3D Case 已加载",
            "独立 3D 视图窗口显示当前 case 的采样点云。\n"
            f"blocks：{case_info.block_count}\n"
            f"时间步数量：{len(case_info.time_values)}\n"
            f"点字段：{case_info.point_arrays}\n"
            f"单元字段：{case_info.cell_arrays}",
        )
        self._set_status("真实 OpenFOAM 3D Case 已加载。")

    def _load_pressure_cloud(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        self._ensure_vtk_viewer()

        self._show_visualization_feedback(
            "正在加载压力云图",
            f"当前项目：{self._current_project.name}\n目标字段：p\nCase 路径：{self._current_project.case_dir}",
        )
        try:
            case_info = self._context.openfoam_vtk_service.inspect(self._current_project)
            geometry = self._context.openfoam_vtk_service.build_geometry_filter(self._current_project)
        except (OSError, RuntimeError) as error:
            self._show_error(f"加载压力云图失败：{error}")
            return

        output = geometry.GetOutput()
        point_pressure = output.GetPointData().GetArray("p")
        cell_pressure = output.GetCellData().GetArray("p")
        if point_pressure is None and cell_pressure is None:
            self._show_error(
                "当前 VTK 输出中没有压力字段 p。"
                f"可用点字段：{case_info.point_arrays}；可用单元字段：{case_info.cell_arrays}"
            )
            return

        if point_pressure is not None:
            scalar_range = point_pressure.GetRange()
        else:
            scalar_range = cell_pressure.GetRange()
        if scalar_range[0] == scalar_range[1]:
            scalar_range = (scalar_range[0] - 1.0, scalar_range[1] + 1.0)
        if point_pressure is None:
            self._show_error("当前阶段的稳定 3D 视图先支持点字段压力云图，单元字段压力云图后续接入。")
            return
        self._vtk_viewer.plot_pressure_points(geometry.GetOutput(), point_pressure, scalar_range)
        self._append_log(
            "压力云图已加载："
            f"pRange=({scalar_range[0]:.6g}, {scalar_range[1]:.6g}), "
            f"times={case_info.time_values}"
        )
        self._show_visualization_feedback(
            "压力云图已加载",
            "独立 3D 视图窗口显示字段 p 的点云标量着色结果。\n"
            f"pRange：({scalar_range[0]:.6g}, {scalar_range[1]:.6g})\n"
            f"时间步：{case_info.time_values}\n"
            "如果画面几乎是单色，通常表示当前最小算例的压力场变化很小或为常量。",
        )
        self._set_status("压力云图已加载。")

    def _load_pressure_surface_cloud(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        self._ensure_vtk_viewer()

        self._show_visualization_feedback(
            "正在加载压力表面云图",
            f"当前项目：{self._current_project.name}\n目标字段：p\nCase 路径：{self._current_project.case_dir}",
        )
        try:
            case_info = self._context.openfoam_vtk_service.inspect(self._current_project)
            geometry = self._context.openfoam_vtk_service.build_geometry_filter(self._current_project)
        except (OSError, RuntimeError) as error:
            self._show_error(f"加载压力表面云图失败：{error}")
            return

        output = geometry.GetOutput()
        point_pressure = output.GetPointData().GetArray("p")
        if point_pressure is None:
            self._show_error(
                "当前稳定表面云图 v1 需要点字段 p。"
                f"可用点字段：{case_info.point_arrays}；可用单元字段：{case_info.cell_arrays}"
            )
            return

        scalar_range = point_pressure.GetRange()
        if scalar_range[0] == scalar_range[1]:
            scalar_range = (scalar_range[0] - 1.0, scalar_range[1] + 1.0)
        face_count = self._vtk_viewer.plot_pressure_surface(output, point_pressure, scalar_range)
        self._append_log(
            "压力表面云图已加载："
            f"faces={face_count}, pRange=({scalar_range[0]:.6g}, {scalar_range[1]:.6g}), "
            f"times={case_info.time_values}"
        )
        self._show_visualization_feedback(
            "压力表面云图已加载",
            "独立 3D 视图窗口显示外表面面片，并按字段 p 进行连续表面着色。\n"
            f"面片数量：{face_count}\n"
            f"pRange：({scalar_range[0]:.6g}, {scalar_range[1]:.6g})\n"
            f"时间步：{case_info.time_values}\n"
            "当前最小方盒子算例压力接近常量，所以颜色变化可能仍然不明显。",
        )
        self._set_status("压力表面云图已加载。")

    def _load_velocity_vectors(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        selected_time = self._visual_time_combo.currentText().strip() or "默认"
        time_value = self._selected_visualization_time()
        self._ensure_vtk_viewer()
        self._show_visualization_feedback(
            "正在加载速度箭头",
            f"字段：U\n时间步：{selected_time}\nCase 路径：{self._current_project.case_dir}",
        )
        try:
            case_info = self._context.openfoam_vtk_service.inspect(self._current_project)
            geometry = self._context.openfoam_vtk_service.build_geometry_filter(
                self._current_project,
                time_value=time_value,
            )
        except (OSError, RuntimeError) as error:
            self._show_error(f"加载速度箭头失败：{error}")
            return

        output = geometry.GetOutput()
        velocity_array = output.GetPointData().GetArray("U")
        if velocity_array is None:
            self._show_error(
                "当前速度箭头 v1 需要点字段 U。"
                f"可用点字段：{case_info.point_arrays}；可用单元字段：{case_info.cell_arrays}"
            )
            return

        arrow_count, speed_range = self._vtk_viewer.plot_velocity_vectors(output, velocity_array)
        self._append_log(
            "速度箭头已加载："
            f"time={selected_time}, arrows={arrow_count}, "
            f"speedRange=({speed_range[0]:.6g}, {speed_range[1]:.6g})"
        )
        self._show_visualization_feedback(
            "速度箭头已加载",
            "独立 3D 视图窗口显示字段 U 的方向箭头。\n"
            "箭头方向表示流体速度方向，箭头颜色表示速度大小 |U|。\n"
            f"时间步：{selected_time}\n"
            f"箭头数量：{arrow_count}\n"
            f"速度范围：({speed_range[0]:.6g}, {speed_range[1]:.6g})\n"
            "说明：为避免界面卡顿，当前会对速度点做采样显示。",
        )
        self._set_status("速度箭头已加载。")

    def _load_velocity_slice(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        selected_time = self._visual_time_combo.currentText().strip() or "默认"
        time_value = self._selected_visualization_time()
        requested_axis = self._slice_axis_combo.currentText().strip() if hasattr(self, "_slice_axis_combo") else "自动"
        requested_position = self._slice_position_input.value() if hasattr(self, "_slice_position_input") else 0.5
        slice_mode = "自动选择速度变化最明显的切面" if requested_axis == "自动" else f"{requested_axis} 方向切面"
        self._ensure_vtk_viewer()
        self._show_visualization_feedback(
            "正在加载速度切面",
            f"字段：U\n切面：{slice_mode}\n位置：{requested_position:.2f}\n时间步：{selected_time}\nCase 路径：{self._current_project.case_dir}",
        )
        try:
            case_info = self._context.openfoam_vtk_service.inspect(self._current_project)
            geometry = self._context.openfoam_vtk_service.build_geometry_filter(
                self._current_project,
                time_value=time_value,
            )
        except (OSError, RuntimeError) as error:
            self._show_error(f"加载速度切面失败：{error}")
            return

        output = geometry.GetOutput()
        velocity_array = output.GetPointData().GetArray("U")
        if velocity_array is None:
            self._show_error(
                "当前速度切面 v1 需要点字段 U。"
                f"可用点字段：{case_info.point_arrays}；可用单元字段：{case_info.cell_arrays}"
            )
            return

        point_count, slice_axis, slice_center, speed_range = self._vtk_viewer.plot_velocity_slice(
            output,
            velocity_array,
            None if requested_axis == "自动" else requested_axis,
            requested_position,
        )
        self._append_log(
            "速度切面已加载："
            f"time={selected_time}, requestedAxis={requested_axis}, position={requested_position:.2f}, "
            f"resolvedAxis={slice_axis}, center={slice_center:.6g}, points={point_count}, "
            f"speedRange=({speed_range[0]:.6g}, {speed_range[1]:.6g})"
        )
        self._show_visualization_feedback(
            "速度切面已加载",
            "独立 3D 视图窗口显示速度大小 |U| 的指定切面。\n"
            "可以把它理解成：用一把刀从计算区域中间切开，只看切面附近的流速分布。\n"
            f"请求方向：{requested_axis}\n"
            f"实际方向：{slice_axis}\n"
            f"归一化位置：{requested_position:.2f}\n"
            f"切面位置：{slice_axis}={slice_center:.6g}\n"
            f"时间步：{selected_time}\n"
            f"切面点数：{point_count}\n"
            f"速度范围：({speed_range[0]:.6g}, {speed_range[1]:.6g})\n"
            "说明：位置 0.00 表示该方向最小坐标侧，1.00 表示最大坐标侧；当前仍是点采样切面。",
        )
        self._set_status("速度切面已加载。")

    def _load_velocity_slice_contour(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        selected_time = self._visual_time_combo.currentText().strip() or "默认"
        time_value = self._selected_visualization_time()
        requested_axis = self._slice_axis_combo.currentText().strip() if hasattr(self, "_slice_axis_combo") else "自动"
        requested_position = self._slice_position_input.value() if hasattr(self, "_slice_position_input") else 0.5
        slice_mode = "自动选择速度变化最明显的切面" if requested_axis == "自动" else f"{requested_axis} 方向切面"
        self._ensure_vtk_viewer()
        self._show_visualization_feedback(
            "正在加载连续速度切面",
            f"字段：U\n切面：{slice_mode}\n位置：{requested_position:.2f}\n时间步：{selected_time}\nCase 路径：{self._current_project.case_dir}",
        )
        try:
            case_info = self._context.openfoam_vtk_service.inspect(self._current_project)
            geometry = self._context.openfoam_vtk_service.build_geometry_filter(
                self._current_project,
                time_value=time_value,
            )
        except (OSError, RuntimeError) as error:
            self._show_error(f"加载连续速度切面失败：{error}")
            return

        output = geometry.GetOutput()
        velocity_array = output.GetPointData().GetArray("U")
        if velocity_array is None:
            self._show_error(
                "当前连续速度切面需要点字段 U。"
                f"可用点字段：{case_info.point_arrays}；可用单元字段：{case_info.cell_arrays}"
            )
            return

        point_count, slice_axis, slice_center, speed_range = self._vtk_viewer.plot_velocity_slice_contour(
            output,
            velocity_array,
            None if requested_axis == "自动" else requested_axis,
            requested_position,
        )
        self._append_log(
            "连续速度切面已加载："
            f"time={selected_time}, requestedAxis={requested_axis}, position={requested_position:.2f}, "
            f"resolvedAxis={slice_axis}, center={slice_center:.6g}, sourcePoints={point_count}, "
            f"speedRange=({speed_range[0]:.6g}, {speed_range[1]:.6g})"
        )
        self._show_visualization_feedback(
            "连续速度切面已加载",
            "独立 3D 视图窗口显示速度大小 |U| 的连续插值切面。\n"
            "它比点采样切面更接近专业 CFD 软件里的切面云图。\n"
            f"请求方向：{requested_axis}\n"
            f"实际方向：{slice_axis}\n"
            f"归一化位置：{requested_position:.2f}\n"
            f"切面位置：{slice_axis}={slice_center:.6g}\n"
            f"时间步：{selected_time}\n"
            f"参与插值点数：{point_count}\n"
            f"速度范围：({speed_range[0]:.6g}, {speed_range[1]:.6g})\n"
            "说明：当前 v1 使用 scipy.griddata 对切面点插值，后续可升级为 VTK 原生切平面。",
        )
        self._set_status("连续速度切面已加载。")
    def _load_velocity_streamlines(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        selected_time = self._visual_time_combo.currentText().strip() or "默认"
        time_value = self._selected_visualization_time()
        self._ensure_vtk_viewer()
        seed_count_limit = int(self._streamline_seed_count_input.value())
        length_factor = float(self._streamline_length_factor_input.value())
        step_factor = float(self._streamline_step_percent_input.value()) / 100.0
        self._show_visualization_feedback(
            "正在加载速度流线",
            f"字段：U\n算法：VTK StreamTracer\n时间步：{selected_time}\n"
            f"种子点上限：{seed_count_limit}\n追踪长度：{length_factor:.2f} x 域尺寸\n"
            f"积分步长：{step_factor * 100:.1f}% 域尺寸\nCase 路径：{self._current_project.case_dir}",
        )
        try:
            case_info = self._context.openfoam_vtk_service.inspect(self._current_project)
            geometry = self._context.openfoam_vtk_service.build_geometry_filter(
                self._current_project,
                time_value=time_value,
            )
        except (OSError, RuntimeError) as error:
            self._show_error(f"加载速度流线失败：{error}")
            return

        output = geometry.GetOutput()
        velocity_array = output.GetPointData().GetArray("U")
        if velocity_array is None:
            self._show_error(
                "当前速度流线需要点字段 U。"
                f"可用点字段：{case_info.point_arrays}；可用单元字段：{case_info.cell_arrays}"
            )
            return

        try:
            streamline_output, main_axis, seed_count, speed_range = self._build_vtk_streamlines(
                output,
                velocity_array,
                seed_count_limit=seed_count_limit,
                length_factor=length_factor,
                step_factor=step_factor,
            )
        except RuntimeError as error:
            self._show_error(f"生成 VTK 流线失败：{error}")
            return
        line_count, line_point_count, main_axis, speed_range = self._vtk_viewer.plot_velocity_streamlines(
            output,
            streamline_output,
            main_axis,
            speed_range,
        )
        self._append_log(
            "速度流线已加载："
            f"algorithm=VTK StreamTracer, time={selected_time}, mainAxis={main_axis}, "
            f"seeds={seed_count}, seedLimit={seed_count_limit}, lengthFactor={length_factor:.2f}, "
            f"stepFactor={step_factor:.4f}, lines={line_count}, linePoints={line_point_count}, "
            f"speedRange=({speed_range[0]:.6g}, {speed_range[1]:.6g})"
        )
        self._show_visualization_feedback(
            "速度流线已加载",
            "独立 3D 视图窗口显示 VTK StreamTracer 基于真实 U 字段积分生成的流线。\n"
            "流线可以理解成：把小纸屑放进流体里，它可能走过的路径。\n"
            f"算法：VTK StreamTracer\n"
            f"主流向轴：{main_axis}\n"
            f"时间步：{selected_time}\n"
            f"种子点上限：{seed_count_limit}\n"
            f"实际种子点数量：{seed_count}\n"
            f"追踪长度：{length_factor:.2f} x 域尺寸\n"
            f"积分步长：{step_factor * 100:.1f}% 域尺寸\n"
            f"流线数量：{line_count}\n"
            f"流线点数：{line_point_count}\n"
            f"速度范围：({speed_range[0]:.6g}, {speed_range[1]:.6g})\n"
            "说明：种子点越多流线越密，步长越小越细但计算更慢。",
        )
        self._set_status("速度流线已加载。")

    def _build_vtk_streamlines(
        self,
        poly_data,
        velocity_array,
        seed_count_limit: int = 24,
        length_factor: float = 2.5,
        step_factor: float = 0.02,
    ):
        points = self._vtk_viewer._points(poly_data)
        vectors = vtk_to_numpy(velocity_array)
        if points.size == 0 or vectors.size == 0 or vectors.ndim != 2 or vectors.shape[1] < 3:
            raise RuntimeError("当前 Case 没有可用于流线追踪的速度点字段。")
        count = min(len(points), len(vectors))
        points = points[:count]
        vectors = vectors[:count, :3]
        speeds = np.linalg.norm(vectors, axis=1)
        usable = speeds > 1e-12
        if not np.any(usable):
            raise RuntimeError("速度场全为 0，无法生成流线。")

        mean_vector = vectors[usable].mean(axis=0)
        axis = int(np.argmax(np.abs(mean_vector)))
        main_axis = "XYZ"[axis]
        bounds_min = points.min(axis=0)
        bounds_max = points.max(axis=0)
        direction_sign = 1.0 if mean_vector[axis] >= 0 else -1.0
        seed_plane = bounds_min[axis] if direction_sign >= 0 else bounds_max[axis]
        plane_tolerance = max(float(bounds_max[axis] - bounds_min[axis]) * 0.08, 1e-9)
        seed_mask = np.abs(points[:, axis] - seed_plane) <= plane_tolerance
        seed_points = points[seed_mask & usable]
        seed_count_limit = max(4, min(int(seed_count_limit), 96))
        length_factor = max(0.5, min(float(length_factor), 10.0))
        step_factor = max(0.002, min(float(step_factor), 0.1))
        if len(seed_points) == 0:
            seed_indices = np.argsort(np.abs(points[:, axis] - seed_plane))[:seed_count_limit]
            seed_points = points[seed_indices]
        if len(seed_points) > seed_count_limit:
            indices = np.linspace(0, len(seed_points) - 1, seed_count_limit, dtype=int)
            seed_points = seed_points[indices]

        vtk_seed_points = vtkPoints()
        for point in seed_points:
            vtk_seed_points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
        seed_data = vtkPolyData()
        seed_data.SetPoints(vtk_seed_points)

        domain_size = max(float((bounds_max - bounds_min).max()), 1e-9)
        initial_step = max(domain_size * step_factor, domain_size * 0.002)
        tracer = vtkStreamTracer()
        tracer.SetInputData(poly_data)
        tracer.SetSourceData(seed_data)
        tracer.SetIntegrator(vtkRungeKutta4())
        tracer.SetIntegrationDirectionToForward()
        tracer.SetMaximumPropagation(domain_size * length_factor)
        tracer.SetInitialIntegrationStep(initial_step)
        tracer.SetMinimumIntegrationStep(max(initial_step * 0.1, domain_size * 0.0002))
        tracer.SetMaximumIntegrationStep(max(initial_step * 2.5, domain_size * 0.005))
        tracer.SetComputeVorticity(False)
        tracer.SetInputArrayToProcess(0, 0, 0, 0, "U")
        tracer.Update()
        streamline_output = tracer.GetOutput()
        if streamline_output.GetNumberOfLines() == 0:
            raise RuntimeError("VTK StreamTracer 没有生成有效流线。")
        return streamline_output, main_axis, vtk_seed_points.GetNumberOfPoints(), (
            float(speeds[usable].min()),
            float(speeds[usable].max()),
        )

    def _refresh_visualization_selectors(self, show_errors: bool = True) -> None:
        if not hasattr(self, "_visual_field_combo"):
            return
        self._visual_field_combo.clear()
        self._visual_time_combo.clear()
        if self._current_project is None:
            if show_errors:
                self._show_error("请先新建或打开项目。")
            return
        try:
            case_info = self._context.openfoam_vtk_service.inspect(self._current_project)
        except (OSError, RuntimeError) as error:
            if show_errors:
                self._show_error(f"刷新可视化字段失败：{error}")
            return

        fields = sorted(set(case_info.point_arrays))
        self._visual_field_combo.addItems(fields)
        if "p" in fields:
            self._visual_field_combo.setCurrentText("p")
        times = [f"{time:g}" for time in case_info.time_values]
        self._visual_time_combo.addItems(times)
        if times:
            self._visual_time_combo.setCurrentText(times[-1])
        self._append_log(
            "可视化字段已刷新："
            f"fields={fields}, times={times or ['默认']}"
        )

    def _load_selected_field_surface(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        field_name = self._visual_field_combo.currentText().strip()
        if not field_name:
            self._show_error("请先刷新并选择可视化字段。")
            return
        selected_time = self._visual_time_combo.currentText().strip() or "默认"
        time_value = self._selected_visualization_time()

        self._show_visualization_feedback(
            "正在加载所选字段表面图",
            f"字段：{field_name}\n时间步：{selected_time}\nCase 路径：{self._current_project.case_dir}",
        )
        if not self._render_selected_field_surface(field_name, time_value, selected_time):
            return

    def _render_selected_field_surface(
        self,
        field_name: str,
        time_value: float | None,
        selected_time: str,
    ) -> bool:
        if self._current_project is None:
            return False
        self._ensure_vtk_viewer()
        try:
            case_info = self._context.openfoam_vtk_service.inspect(self._current_project)
            geometry = self._context.openfoam_vtk_service.build_geometry_filter(
                self._current_project,
                time_value=time_value,
            )
        except (OSError, RuntimeError) as error:
            self._show_error(f"加载所选字段表面图失败：{error}")
            return

        output = geometry.GetOutput()
        field_array = output.GetPointData().GetArray(field_name)
        if field_array is None:
            self._show_error(
                f"当前稳定表面图 v1 需要点字段 {field_name}。"
                f"可用点字段：{case_info.point_arrays}；可用单元字段：{case_info.cell_arrays}"
            )
            return False

        values = self._vtk_viewer._scalar_values(field_array)
        if values.size == 0:
            self._show_error(f"字段 {field_name} 没有可显示数据。")
            return False
        scalar_range = (float(values.min()), float(values.max()))
        if scalar_range[0] == scalar_range[1]:
            scalar_range = (scalar_range[0] - 1.0, scalar_range[1] + 1.0)
        face_count = self._vtk_viewer.plot_field_surface(
            output,
            field_array,
            scalar_range,
            field_name if field_array.GetNumberOfComponents() == 1 else f"|{field_name}|",
        )
        self._append_log(
            "所选字段表面图已加载："
            f"field={field_name}, time={selected_time}, faces={face_count}, "
            f"range=({scalar_range[0]:.6g}, {scalar_range[1]:.6g})"
        )
        self._show_visualization_feedback(
            "所选字段表面图已加载",
            f"字段：{field_name}\n"
            f"显示值：{'标量值' if field_array.GetNumberOfComponents() == 1 else '矢量模长'}\n"
            f"时间步：{selected_time}\n"
            f"面片数量：{face_count}\n"
            f"范围：({scalar_range[0]:.6g}, {scalar_range[1]:.6g})\n"
            "说明：当前图像已按所选时间步请求 VTK Reader 读取。",
        )
        self._set_status("所选字段表面图已加载。")
        return True

    def _play_visualization_animation(self) -> None:
        if not hasattr(self, "_visual_time_combo") or self._visual_time_combo.count() == 0:
            self._show_error("请先刷新可视化字段和时间步。")
            return
        if not self._visual_field_combo.currentText().strip():
            self._show_error("请先选择字段。")
            return
        self._visual_animation_timer.start(self._visual_frame_interval_input.value())
        self._append_log("时间步动画已开始播放。")
        self._set_status("时间步动画播放中。")

    def _stop_visualization_animation(self) -> None:
        self._visual_animation_timer.stop()
        self._append_log("时间步动画已暂停。")
        self._set_status("时间步动画已暂停。")

    def _advance_visualization_frame(self) -> None:
        self._step_visualization_frame(1)

    def _step_visualization_frame(self, step: int) -> None:
        if not hasattr(self, "_visual_time_combo") or self._visual_time_combo.count() == 0:
            self._show_error("请先刷新可视化字段和时间步。")
            return
        current_index = self._visual_time_combo.currentIndex()
        next_index = (current_index + step) % self._visual_time_combo.count()
        self._visual_time_combo.setCurrentIndex(next_index)
        field_name = self._visual_field_combo.currentText().strip()
        selected_time = self._visual_time_combo.currentText().strip()
        if not field_name:
            self._show_error("请先选择字段。")
            return
        self._show_visualization_feedback(
            "正在播放时间步动画",
            f"字段：{field_name}\n当前时间步：{selected_time}\n"
            f"帧：{next_index + 1}/{self._visual_time_combo.count()}",
        )
        ok = self._render_selected_field_surface(
            field_name,
            self._selected_visualization_time(),
            selected_time,
        )
        if not ok:
            self._visual_animation_timer.stop()

    def _selected_visualization_time(self) -> float | None:
        if not hasattr(self, "_visual_time_combo"):
            return None
        text = self._visual_time_combo.currentText().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _show_visualization_feedback(self, title: str, detail: str) -> None:
        self._workspace_tabs.setCurrentIndex(5)
        if hasattr(self, "_results_text"):
            self._results_text.setPlainText(f"{title}\n\n{detail}")

    def _ensure_vtk_viewer(self) -> None:
        if self._vtk_viewer is None:
            self._vtk_viewer = VtkViewerDialog(self)
        if self._current_project is not None:
            self._vtk_viewer.set_geometry_assets(
                self._context.geometry_import_service.list_assets(self._current_project)
            )
        self._vtk_viewer.show()
        self._vtk_viewer.raise_()
        self._vtk_viewer.activateWindow()

    def _finalize_vtk_render(self) -> None:
        if self._vtk_viewer is not None:
            self._vtk_viewer.show()
            self._vtk_viewer.raise_()
            self._vtk_viewer.activateWindow()

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
        self._set_theme(self._theme_names[self._theme_index])

    def _set_theme(self, theme_name: str) -> None:
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
            last_project_path=settings.last_project_path,
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
            last_project_path=settings.last_project_path,
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
        if status is None or isinstance(status, bool):
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
            self._refresh_results_panel()
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

        self._activate_project(project, "项目创建完成。")
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

        self._activate_project(project, "项目打开完成。")
        self._append_log(f"已打开项目：{project.path}")
        self._set_status("项目打开完成。")

    def _activate_project(self, project: SimulationProject, status_text: str) -> None:
        self._current_project = project
        self._clear_case_runtime_state()
        self._context.project_service.remember_project(project)
        self._refresh_project_tree()
        self._workspace_tabs.setCurrentIndex(0)
        self._case_label.setText(f"当前 Case: {project.name}/{project.case_name}")
        self._load_case_parameters()
        self._refresh_solver_run_panel()
        self._refresh_results_panel()
        self._refresh_geometry_panel()
        self._restore_project_result_state()
        self._append_log(f"当前项目：{project.path}")
        self._set_status(status_text)

    def _clear_case_runtime_state(self) -> None:
        self._current_process_output = ""
        self._last_diagnostic_summary = "暂无诊断。"
        if hasattr(self, "_solver_metric_summary"):
            self._solver_metric_summary.setPlainText("关键指标摘要：尚未运行。")
        if hasattr(self, "_solver_diagnostic_text"):
            self._solver_diagnostic_text.setPlainText("最近诊断：\n暂无诊断。")
        if hasattr(self, "_residual_figure"):
            self._residual_figure.clear()
            axes = self._residual_figure.add_subplot(111)
            axes.set_title("Residual Curve")
            axes.text(
                0.5,
                0.5,
                "Current case has no residual data.",
                ha="center",
                va="center",
                transform=axes.transAxes,
            )
            axes.set_axis_off()
            self._residual_canvas.draw()

    def _create_case(self) -> None:
        if self._current_project is None:
            self._show_error("请先选择项目。")
            return
        name, ok = QInputDialog.getText(self, "新增 Case", "Case 名称")
        if not ok:
            return
        try:
            project = self._context.project_service.create_case(self._current_project, name)
        except ValueError as error:
            self._show_error(str(error))
            return
        self._activate_project(project, "Case 创建完成。")
        self._append_log(f"已创建 Case：{project.case_dir}")

    def _return_to_project_selection(self) -> None:
        app = QApplication.instance()
        old_quit_on_close = app.quitOnLastWindowClosed() if app else True
        if app:
            app.setQuitOnLastWindowClosed(False)
        current_project = self._current_project
        self.close()
        QApplication.processEvents()
        startup_window = StartupWindow(self._context)
        selected_project = current_project
        status_text = "已取消项目选择。"
        if startup_window.exec() == 1 and startup_window.selected_project is not None:
            selected_project = startup_window.selected_project
            status_text = "已从项目选择页切换项目。"

        if selected_project is None:
            if app:
                app.setQuitOnLastWindowClosed(old_quit_on_close)
            return

        new_window = MainWindow(self._context, initial_project=selected_project)
        if app:
            app._foamdesk_main_window = new_window
        new_window.show()
        new_window.raise_()
        new_window.activateWindow()
        new_window._set_status(status_text)
        if app:
            app.setQuitOnLastWindowClosed(old_quit_on_close)

    def _restore_project_result_state(self) -> None:
        if self._current_project is None:
            return
        try:
            result_index = self._context.result_index_service.index(self._current_project)
        except OSError:
            return

        residuals_csv = self._current_project.case_dir / "foamdesk_results" / "residuals.csv"
        metrics_json = self._current_project.case_dir / "foamdesk_results" / "metrics.json"
        if result_index.latest_time is None and not residuals_csv.exists():
            self._task_text.setPlainText("任务状态：当前项目暂无求解结果")
            self._refresh_solver_run_panel("暂无求解结果")
            self._clear_case_runtime_state()
            return

        lines = [
            "任务状态：已加载项目已有结果",
            f"最新时间步：{result_index.latest_time or '无'}",
            f"网格目录：{'已生成' if result_index.has_mesh else '未生成'}",
            f"残差 CSV：{'已存在' if residuals_csv.exists() else '未生成'}",
            f"指标 JSON：{'已存在' if metrics_json.exists() else '未生成'}",
            "",
            "说明：打开项目只读取已有文件，不会自动重新求解；只有点击“运行”才会重新执行 OpenFOAM。",
        ]
        self._task_text.setPlainText("\n".join(lines))
        self._refresh_solver_run_panel("已加载已有结果")
        if residuals_csv.exists():
            try:
                self._plot_residual_curve()
            except (OSError, ValueError):
                pass

    def _refresh_project_tree(self) -> None:
        if not hasattr(self, "_project_tree"):
            return
        self._project_tree.clear()
        if self._current_project is None:
            empty_item = QTreeWidgetItem(["未选择项目"])
            empty_item.setDisabled(True)
            self._project_tree.addTopLevelItem(empty_item)
            return

        project_item = QTreeWidgetItem([self._current_project.name])
        project_item.setData(0, Qt.ItemDataRole.UserRole, "")
        for case_name in self._context.project_service.list_cases(self._current_project):
            case_item = QTreeWidgetItem([case_name])
            case_item.setData(0, Qt.ItemDataRole.UserRole, case_name)
            if case_name == self._current_project.case_name:
                case_item.setText(0, f"{case_name}  ✓")
            project_item.addChild(case_item)
        self._project_tree.addTopLevelItem(project_item)
        self._project_tree.expandAll()

    def _on_project_tree_item_clicked(self, item: QTreeWidgetItem) -> None:
        case_name = item.data(0, Qt.ItemDataRole.UserRole)
        if not case_name or self._current_project is None:
            return
        try:
            project = self._context.project_service.switch_case(self._current_project, str(case_name))
        except ValueError as error:
            self._show_error(str(error))
            return
        self._activate_project(project, "Case 已切换。")
        self._append_log(f"当前 Case 目录：{project.case_dir}")

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

    def _run_minimal_simulation(self) -> None:
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

        if not self._save_case_parameters():
            return

        repaired_files = self._context.project_service.ensure_minimal_case_template(self._current_project)
        if repaired_files:
            relative_files = [str(path.relative_to(self._current_project.case_dir)) for path in repaired_files]
            self._append_log("已补齐旧项目缺失的最小仿真文件：")
            self._append_log("\n".join(f"- {path}" for path in relative_files))

        block_mesh_dict = self._current_project.case_dir / "system" / "blockMeshDict"
        if not block_mesh_dict.exists():
            self._show_error("当前 case 缺少 system/blockMeshDict。")
            return

        self._workspace_tabs.setCurrentIndex(2)
        self._bottom_tabs.setCurrentIndex(0)
        self._task_text.setPlainText("任务状态：最小仿真运行中")
        self._current_process_output = ""
        self._last_diagnostic_summary = "本次任务正在运行，暂无失败诊断。"
        self._active_process_kind = "minimal"
        self._refresh_solver_run_panel("最小仿真运行中")
        self._set_status("最小仿真运行中。")

        command = (
            f"source {shlex.quote(status.env_script_path)} >/dev/null 2>&1 && "
            f"cd {shlex.quote(str(self._current_project.case_dir))} && "
            "blockMesh && icoFoam"
        )
        self._foam_process = QProcess(self)
        self._foam_process.setProgram("bash")
        self._foam_process.setArguments(["-lc", command])
        self._foam_process.readyReadStandardOutput.connect(self._read_process_stdout)
        self._foam_process.readyReadStandardError.connect(self._read_process_stderr)
        self._foam_process.finished.connect(self._on_process_finished)
        self._foam_process.start()
        self._append_log(f"启动最小仿真：{self._current_project.case_dir}")
        self._append_log("执行流程：blockMesh -> icoFoam")

    def _stop_current_process(self) -> None:
        if not self._foam_process or self._foam_process.state() == QProcess.ProcessState.NotRunning:
            self._task_text.setPlainText("任务状态：空闲")
            self._refresh_solver_run_panel("空闲")
            self._set_status("当前没有正在运行的任务。")
            return
        self._foam_process.terminate()
        if not self._foam_process.waitForFinished(3000):
            self._foam_process.kill()
        self._task_text.setPlainText("任务状态：已停止")
        self._active_process_kind = "idle"
        self._refresh_solver_run_panel("已停止")
        self._set_status("任务已停止。")

    def _read_process_stdout(self) -> None:
        if self._foam_process:
            output = bytes(self._foam_process.readAllStandardOutput()).decode(errors="replace")
            self._current_process_output += output
            self._append_log(output)
            self._refresh_solver_metrics_panel()

    def _read_process_stderr(self) -> None:
        if self._foam_process:
            output = bytes(self._foam_process.readAllStandardError()).decode(errors="replace")
            self._current_process_output += output
            self._append_log(output)
            self._refresh_solver_metrics_panel()

    def _on_process_finished(self, exit_code: int, _exit_status) -> None:
        process_kind = self._active_process_kind
        self._active_process_kind = "idle"
        if exit_code == 0 and process_kind == "simulationPipeline":
            self._task_text.setPlainText("任务状态：一键仿真流水线完成")
            self._last_diagnostic_summary = "一键仿真流水线正常完成，没有失败诊断。"
            if self._export_solver_metrics():
                self._plot_residual_curve()
            self._refresh_geometry_panel()
            self._refresh_results_panel()
            self._refresh_solver_run_panel("一键仿真流水线完成")
            self._set_status("一键仿真流水线完成。")
        elif exit_code == 0 and process_kind == "preprocess":
            summary = self._format_check_mesh_summary(self._current_process_output)
            self._task_text.setPlainText("任务状态：一键前处理完成")
            self._last_diagnostic_summary = "一键前处理完成。\n\n" + summary
            self._problem_text.setPlainText(self._last_diagnostic_summary)
            self._refresh_geometry_panel()
            self._refresh_results_panel()
            self._refresh_solver_run_panel("一键前处理完成")
            self._set_status("一键前处理完成。")
        elif exit_code == 0 and process_kind == "checkMesh":
            summary = self._format_check_mesh_summary(self._current_process_output)
            self._task_text.setPlainText("任务状态：checkMesh 完成")
            self._last_diagnostic_summary = summary
            self._problem_text.setPlainText(summary)
            self._refresh_results_panel()
            self._refresh_solver_run_panel("checkMesh 完成")
            self._set_status("checkMesh 完成。")
        elif exit_code == 0 and process_kind == "snappyHexMesh":
            self._task_text.setPlainText("任务状态：snappyHexMesh 完成")
            self._last_diagnostic_summary = "snappyHexMesh 正常完成，没有失败诊断。"
            self._refresh_geometry_panel()
            self._refresh_results_panel()
            self._refresh_solver_run_panel("snappyHexMesh 完成")
            self._set_status("snappyHexMesh 完成。")
        elif exit_code == 0:
            self._task_text.setPlainText("任务状态：最小仿真完成")
            self._last_diagnostic_summary = "本次任务正常完成，没有失败诊断。"
            if self._export_solver_metrics():
                self._plot_residual_curve()
            self._refresh_results_panel()
            self._refresh_solver_run_panel("最小仿真完成")
            self._set_status("最小仿真完成。")
        else:
            self._update_diagnostics(exit_code)
            label = self._process_label(process_kind)
            failed_step = self._detect_failed_pipeline_step(process_kind, self._current_process_output)
            if failed_step:
                advice = self._pipeline_step_advice(failed_step)
                self._last_diagnostic_summary = (
                    f"失败步骤：{failed_step}\n\n{advice}\n\n{self._last_diagnostic_summary}"
                )
                self._problem_text.setPlainText(self._last_diagnostic_summary)
                self._bottom_tabs.setCurrentIndex(2)
            self._task_text.setPlainText(f"任务状态：{label}失败，退出码 {exit_code}")
            self._refresh_solver_run_panel(f"{label}失败，退出码 {exit_code}")
            self._set_status(f"{label}失败，退出码 {exit_code}。")

    def _process_label(self, process_kind: str) -> str:
        labels = {
            "simulationPipeline": "一键仿真流水线",
            "preprocess": "一键前处理",
            "checkMesh": "checkMesh",
            "snappyHexMesh": "snappyHexMesh",
            "minimal": "最小仿真",
        }
        return labels.get(process_kind, "OpenFOAM 任务")

    def _detect_failed_pipeline_step(self, process_kind: str, output: str) -> str | None:
        if process_kind not in {"preprocess", "simulationPipeline"}:
            return None
        step_names = {
            "blockMesh": "blockMesh 背景网格生成",
            "snappyHexMesh": "snappyHexMesh 贴体网格生成",
            "checkMesh": "checkMesh 网格质量检查",
            "icoFoam": "icoFoam 最小求解",
        }
        matches = re.findall(r"FOAMDESK_STEP:([A-Za-z0-9_]+)", output)
        if not matches:
            return "未知步骤，未识别到 FoamDesk 步骤标记"
        return step_names.get(matches[-1], matches[-1])

    def _pipeline_step_advice(self, failed_step: str) -> str:
        if "blockMesh" in failed_step:
            return (
                "修复建议：\n"
                "- 检查 `system/blockMeshDict` 是否存在并且语法正确。\n"
                "- 确认背景网格区域要包住 STL 几何，否则 snappyHexMesh 后续无法贴体。\n"
                "- 如果你刚新建 Case，可以先运行最小仿真验证 blockMesh 是否能单独通过。"
            )
        if "snappyHexMesh" in failed_step:
            return (
                "修复建议：\n"
                "- 检查 STL 是否已经导入到 `constant/triSurface`，文件名是否和 `snappyHexMeshDict` 一致。\n"
                "- 检查 `locationInMesh` 是否位于流体区域内部；这个点选错会导致网格区域判断失败。\n"
                "- 先降低最大加密等级，例如从 4 降到 2，减少网格生成压力。\n"
                "- 如果启用了边界层 addLayers，先关闭边界层再试。"
            )
        if "checkMesh" in failed_step:
            return (
                "修复建议：\n"
                "- 查看日志中的 `Failed`、`severely non-orthogonal`、`skewness` 等关键词。\n"
                "- 降低 snappy 加密等级或关闭边界层，先得到可用网格。\n"
                "- 如果非正交角或扭曲度过高，需要调整背景网格、STL 几何质量或 snappy 参数。"
            )
        if "icoFoam" in failed_step:
            return (
                "修复建议：\n"
                "- 检查 `0/U` 和 `0/p` 的边界名称是否和网格 boundary 文件一致。\n"
                "- 如果 snappyHexMesh 生成了新的 patch，求解场文件也必须包含对应边界条件。\n"
                "- 检查 `system/controlDict`、`fvSchemes`、`fvSolution` 是否完整。\n"
                "- 先确认 checkMesh 通过，再运行求解器。"
            )
        return (
            "修复建议：\n"
            "- 查看底部日志中最后一个 OpenFOAM 报错块。\n"
            "- 优先检查当前 Case 的 system、constant、0 目录是否完整。"
        )

    def _format_check_mesh_summary(self, output: str) -> str:
        lines = ["checkMesh 网格质量检查摘要", ""]
        if "Mesh OK." in output:
            lines.append("总体结论：通过，OpenFOAM 输出 Mesh OK。")
        elif "Failed" in output or "failed" in output:
            lines.append("总体结论：存在失败检查，需要查看日志中的 Failed 项。")
        else:
            lines.append("总体结论：未识别到明确 Mesh OK，请查看完整日志。")

        checks = [
            ("点数量", r"points:\s+([0-9]+)"),
            ("面数量", r"faces:\s+([0-9]+)"),
            ("单元数量", r"cells:\s+([0-9]+)"),
            ("边界 patch 数量", r"boundary patches:\s+([0-9]+)"),
            ("最大长宽比", r"Max aspect ratio\s*=\s*([0-9.eE+-]+)"),
            ("最大非正交角", r"Mesh non-orthogonality Max:\s*([0-9.eE+-]+)"),
            ("最大扭曲度", r"Max skewness\s*=\s*([0-9.eE+-]+)"),
        ]
        for label, pattern in checks:
            match = re.search(pattern, output)
            if match:
                lines.append(f"{label}：{match.group(1)}")

        failed_lines = [
            line.strip()
            for line in output.splitlines()
            if "Failed" in line or "failed" in line or "Error" in line
        ]
        if failed_lines:
            lines.extend(["", "需要关注："])
            lines.extend(f"- {line}" for line in failed_lines[:8])

        lines.extend(
            [
                "",
                "说明：checkMesh 是 OpenFOAM 的网格体检工具。它不是求解流体，而是检查当前网格是否适合后续计算。",
            ]
        )
        return "\n".join(lines)

    def _update_diagnostics(self, exit_code: int) -> None:
        diagnostics = self._context.log_diagnostic_service.diagnose(self._current_process_output)
        summary = self._context.log_diagnostic_service.format_diagnostics(diagnostics)
        self._last_diagnostic_summary = f"退出码：{exit_code}\n\n{summary}"
        self._problem_text.setPlainText(self._last_diagnostic_summary)
        self._bottom_tabs.setCurrentIndex(2)
        self._append_log("已生成失败诊断。")

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
        if self._tutorial_overlay:
            self._tutorial_overlay.show_overlay()
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
