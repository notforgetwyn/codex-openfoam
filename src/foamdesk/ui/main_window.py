from __future__ import annotations

import shlex
import re
from pathlib import Path

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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        action_row = QHBoxLayout()
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

    def closeEvent(self, event) -> None:  # noqa: N802
        self._clear_tabs()
        super().closeEvent(event)

    def plot_cube(self) -> None:
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
        points = self._points(poly_data)
        axes = self._reset_axes(title)
        if points.size == 0:
            axes.text2D(0.08, 0.5, "当前 case 没有可显示点数据。", color="#d4d4d4")
            self._show()
            return
        points = self._sample_points(points)
        axes.scatter(points[:, 0], points[:, 1], points[:, 2], s=6, c="#4fc3ff", alpha=0.85)
        self._finish_axes(axes, points)

    def plot_stl_file(self, path: Path) -> tuple[int, int]:
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
        points = self._points(poly_data)
        faces = self._faces(poly_data)
        values = self._scalar_values(field_array)
        axes = self._reset_axes(f"{field_name} Surface")
        if points.size == 0 or faces.size == 0 or values.size == 0:
            axes.text2D(0.08, 0.5, "No surface data to display.", color="#d4d4d4")
            self._show()
            return 0

        faces = self._sample_faces(faces)
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
        points = self._points(poly_data)
        vectors = vtk_to_numpy(vector_array)
        axes = self._reset_axes("Velocity Slice Contour |U|")
        if points.size == 0 or vectors.size == 0 or vectors.ndim != 2 or vectors.shape[1] < 3:
            axes.text2D(0.08, 0.5, "No velocity slice data to display.", color="#d4d4d4")
            self._show()
            return 0, "X", 0.0, (0.0, 0.0)

        count = min(len(points), len(vectors))
        points = points[:count]
        speeds = np.linalg.norm(vectors[:count, :3], axis=1)
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
            total_line_points += count

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
        self._finish_axes(axes, source_points)
        return line_count, total_line_points, main_axis, speed_range

    def plot_pressure_points(self, poly_data, pressure_array, scalar_range: tuple[float, float]) -> None:
        points = self._points(poly_data)
        pressure = vtk_to_numpy(pressure_array)
        axes = self._reset_axes("Pressure Point Cloud")
        if points.size == 0 or pressure.size == 0:
            axes.text2D(0.08, 0.5, "当前 case 没有可显示压力点数据。", color="#d4d4d4")
            self._show()
            return
        count = min(len(points), len(pressure))
        points = points[:count]
        pressure = pressure[:count]
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
        self._tabs.removeTab(index)
        if widget is not None:
            widget.deleteLater()

    def _clear_tabs(self) -> None:
        while self._tabs.count():
            self._close_tab(0)
        self.figure = None
        self.canvas = None

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
        layout = QVBoxLayout(wrapper)
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
        import_button = QPushButton("导入 STL")
        refresh_button = QPushButton("刷新几何清单")
        snappy_button = QPushButton("生成 snappyHexMeshDict")
        run_snappy_button = QPushButton("运行 snappyHexMesh")
        check_mesh_button = QPushButton("运行 checkMesh")
        preview_button = QPushButton("预览已导入 STL")
        limitation_button = QPushButton("STEP/IGES 支持说明")
        import_button.clicked.connect(lambda _checked=False: self._import_stl_geometry())
        refresh_button.clicked.connect(lambda _checked=False: self._refresh_geometry_panel())
        snappy_button.clicked.connect(lambda _checked=False: self._generate_snappy_hex_mesh_dict())
        run_snappy_button.clicked.connect(lambda _checked=False: self._run_snappy_hex_mesh())
        check_mesh_button.clicked.connect(lambda _checked=False: self._run_check_mesh())
        preview_button.clicked.connect(lambda _checked=False: self._preview_imported_stl())
        limitation_button.clicked.connect(lambda _checked=False: self._show_cad_import_limitations())
        action_row.addWidget(import_button)
        action_row.addWidget(refresh_button)
        action_row.addWidget(snappy_button)
        action_row.addWidget(run_snappy_button)
        action_row.addWidget(check_mesh_button)
        action_row.addWidget(preview_button)
        action_row.addWidget(limitation_button)
        action_row.addStretch(1)

        self._geometry_text = QTextEdit()
        self._geometry_text.setReadOnly(True)
        self._geometry_text.setPlainText("请先新建或打开项目，然后导入 STL 几何。")

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addLayout(action_row)
        layout.addWidget(self._geometry_text, 1)
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
        try:
            asset = self._context.geometry_import_service.import_stl(
                self._current_project,
                Path(file_path),
            )
        except (OSError, ValueError) as error:
            self._show_error(f"导入 STL 失败：{error}")
            return
        self._append_log(f"STL 几何已导入：{asset.stored_path}")
        self._refresh_geometry_panel()
        self._workspace_tabs.setCurrentIndex(6)
        self._set_status("STL 几何导入完成。")

    def _refresh_geometry_panel(self) -> None:
        if not hasattr(self, "_geometry_text"):
            return
        if self._current_project is None:
            self._geometry_text.setPlainText("请先新建或打开项目，然后导入 STL 几何。")
            return
        self._geometry_text.setPlainText(self._context.geometry_import_service.format_assets(self._current_project))

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
            )
        except (OSError, ValueError) as error:
            self._show_error(f"生成 snappyHexMeshDict 失败：{error}")
            return

        self._append_log(f"snappyHexMeshDict 已生成：{dict_path}")
        self._refresh_geometry_panel()
        self._workspace_tabs.setCurrentIndex(6)
        self._set_status("snappyHexMeshDict 生成完成。")

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
        if exit_code == 0 and process_kind == "checkMesh":
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
            self._task_text.setPlainText(f"任务状态：{label}失败，退出码 {exit_code}")
            self._refresh_solver_run_panel(f"{label}失败，退出码 {exit_code}")
            self._set_status(f"{label}失败，退出码 {exit_code}。")

    def _process_label(self, process_kind: str) -> str:
        labels = {
            "checkMesh": "checkMesh",
            "snappyHexMesh": "snappyHexMesh",
            "minimal": "最小仿真",
        }
        return labels.get(process_kind, "OpenFOAM 任务")

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
