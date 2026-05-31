from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import vtk
from matplotlib import colormaps
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def _capability_value(capabilities: str, key: str) -> str:
    for line in capabilities.splitlines():
        if key.lower() in line.lower():
            return line.split(":", 1)[-1].strip() if ":" in line else line.strip()
    return "unknown"


def _log_vtk_render_backend(render_window, label: str) -> None:
    try:
        capabilities = render_window.ReportCapabilities()
    except Exception as error:  # noqa: BLE001
        print(f"[FoamDesk VTK] {label}: render backend unknown ({error})", flush=True)
        return
    renderer = _capability_value(capabilities, "OpenGL renderer string")
    vendor = _capability_value(capabilities, "OpenGL vendor string")
    version = _capability_value(capabilities, "OpenGL version string")
    lower_renderer = renderer.lower()
    mode = "CPU software rendering" if any(token in lower_renderer for token in ("llvmpipe", "softpipe", "software")) else "GPU hardware rendering"
    print(
        f"[FoamDesk VTK] {label}: {mode}; renderer={renderer}; vendor={vendor}; OpenGL={version}",
        flush=True,
    )


class NativeVtkPreviewWidget(QWidget):
    def __init__(self, parent: QWidget | None = None, background: tuple[float, float, float] = (0.12, 0.12, 0.12)) -> None:
        super().__init__(parent)
        self._background = background
        self._interactor_initialized = False
        self._render_backend_logged = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._vtk_widget = QVTKRenderWindowInteractor(self)
        self._vtk_widget.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        layout.addWidget(self._vtk_widget, 1)
        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(*background)
        self._vtk_widget.GetRenderWindow().AddRenderer(self._renderer)
        self._interactor = self._vtk_widget.GetRenderWindow().GetInteractor()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        QTimer.singleShot(0, self._initialize_interactor)

    def _initialize_interactor(self) -> None:
        if self._interactor_initialized or not self._vtk_widget.isVisible():
            return
        self._interactor.Initialize()
        self._interactor_initialized = True

    def clear(self, background: tuple[float, float, float] | None = None) -> None:
        if background is not None:
            self._background = background
        self._renderer.RemoveAllViewProps()
        self._renderer.SetBackground(*self._background)

    def add_polydata(
        self,
        poly_data,
        color: tuple[float, float, float] = (0.25, 0.74, 1.0),
        opacity: float = 0.72,
        edge_color: tuple[float, float, float] | None = (0.03, 0.18, 0.28),
        line_width: float = 0.25,
    ) -> vtk.vtkActor:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetInterpolationToPhong()
        if edge_color is not None:
            actor.GetProperty().EdgeVisibilityOn()
            actor.GetProperty().SetEdgeColor(*edge_color)
            actor.GetProperty().SetLineWidth(line_width)
        self._renderer.AddActor(actor)
        return actor

    def add_polyline(
        self,
        points: np.ndarray,
        color: tuple[float, float, float] = (0.31, 0.76, 1.0),
        width: float = 2.0,
        opacity: float = 1.0,
    ) -> None:
        if len(points) < 2:
            return
        vtk_points = vtk.vtkPoints()
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(points))
        for index, point in enumerate(points):
            vtk_points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
            polyline.GetPointIds().SetId(index, index)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyline)
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)
        poly_data.SetLines(cells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(width)
        actor.GetProperty().SetOpacity(opacity)
        self._renderer.AddActor(actor)

    def add_polygon(
        self,
        vertices: np.ndarray,
        color: tuple[float, float, float] = (0.31, 0.76, 1.0),
        opacity: float = 0.18,
        edge_color: tuple[float, float, float] | None = (0.31, 0.76, 1.0),
    ) -> None:
        if len(vertices) < 3:
            return
        points = vtk.vtkPoints()
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(vertices))
        for index, vertex in enumerate(vertices):
            points.InsertNextPoint(float(vertex[0]), float(vertex[1]), float(vertex[2]))
            polygon.GetPointIds().SetId(index, index)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polygon)
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(cells)
        self.add_polydata(poly_data, color=color, opacity=opacity, edge_color=edge_color, line_width=1.0)

    def add_arrow(
        self,
        start: np.ndarray,
        direction: np.ndarray,
        length: float,
        color: tuple[float, float, float] = (0.15, 0.45, 1.0),
        opacity: float = 0.92,
    ) -> None:
        direction = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12 or length <= 0:
            return
        unit = direction / norm
        arrow = vtk.vtkArrowSource()
        arrow.SetShaftRadius(0.025)
        arrow.SetTipRadius(0.075)
        arrow.SetTipLength(0.28)
        arrow.Update()

        transform = vtk.vtkTransform()
        matrix = vtk.vtkMatrix4x4()
        side_values = [0.0, 0.0, 0.0]
        up_values = [0.0, 0.0, 0.0]
        vtk.vtkMath.Perpendiculars(unit.tolist(), side_values, up_values, 0)
        side = np.asarray(side_values, dtype=float)
        up = np.cross(unit, side)
        matrix.Identity()
        for row, value in enumerate(unit):
            matrix.SetElement(row, 0, float(value))
        for row, value in enumerate(side):
            matrix.SetElement(row, 1, float(value))
        for row, value in enumerate(up):
            matrix.SetElement(row, 2, float(value))
        transform.Translate(float(start[0]), float(start[1]), float(start[2]))
        transform.Concatenate(matrix)
        transform.Scale(float(length), float(length), float(length))

        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputConnection(arrow.GetOutputPort())
        tf.SetTransform(transform)
        tf.Update()
        actor = self.add_polydata(tf.GetOutput(), color=color, opacity=opacity, edge_color=None)
        actor.GetProperty().SetSpecular(0.25)

    def add_text(
        self,
        text: str,
        position: tuple[float, float, float],
        color: tuple[float, float, float] = (0.85, 0.85, 0.85),
        size: int = 16,
    ) -> None:
        actor = vtk.vtkBillboardTextActor3D()
        actor.SetInput(text)
        actor.SetPosition(float(position[0]), float(position[1]), float(position[2]))
        actor.GetTextProperty().SetColor(*color)
        actor.GetTextProperty().SetFontSize(size)
        actor.GetTextProperty().SetBold(True)
        actor.GetTextProperty().ShadowOff()
        self._renderer.AddActor(actor)

    def add_message(self, message: str) -> None:
        actor = vtk.vtkTextActor()
        actor.SetInput(message)
        actor.SetPosition(24, 34)
        actor.GetTextProperty().SetColor(0.85, 0.85, 0.85)
        actor.GetTextProperty().SetFontSize(18)
        actor.GetTextProperty().ShadowOff()
        self._renderer.AddActor2D(actor)

    def finish(self, points: np.ndarray | None = None) -> None:
        self._renderer.ResetCamera()
        camera = self._renderer.GetActiveCamera()
        camera.Azimuth(-35)
        camera.Elevation(22)
        camera.Zoom(1.12)
        self._renderer.ResetCameraClippingRange()
        QTimer.singleShot(0, self.render)

    def render(self) -> None:
        if not self.isVisible() or not self._vtk_widget.isVisible():
            return
        self._initialize_interactor()
        render_window = self._vtk_widget.GetRenderWindow()
        render_window.Render()
        if not self._render_backend_logged:
            _log_vtk_render_backend(render_window, "preview widget")
            self._render_backend_logged = True

    def save_png(self, path: Path) -> None:
        self.render()
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(self._vtk_widget.GetRenderWindow())
        window_to_image.SetInputBufferTypeToRGBA()
        window_to_image.ReadFrontBufferOff()
        window_to_image.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(window_to_image.GetOutputPort())
        writer.Write()


class VtkViewerDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("FoamDesk 3D 视图")
        self.resize(920, 680)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        action_row = QHBoxLayout()
        action_row.addWidget(QLabel("STL 预览窗口"))
        action_row.addStretch(1)
        clear_button = QPushButton("清空视图")
        clear_button.clicked.connect(self._clear_tabs)
        export_button = QPushButton("导出 PNG")
        export_button.clicked.connect(self._export_png)
        action_row.addWidget(clear_button)
        action_row.addWidget(export_button)
        layout.addLayout(action_row)

        self._last_plot_title = "foamdesk_visualization"
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setTabsClosable(True)
        self._tabs.tabCloseRequested.connect(self._close_tab)
        layout.addWidget(self._tabs)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._clear_tabs()
        super().closeEvent(event)

    def plot_stl_file(self, path: Path) -> tuple[int, int]:
        reader = vtkSTLReader()
        reader.SetFileName(str(path))
        reader.Update()
        poly_data = reader.GetOutput()
        points = self._points(poly_data)
        faces = self._faces(poly_data)
        canvas = self._reset_view(f"STL {path.name}")
        if points.size == 0 or faces.size == 0:
            canvas.add_message("No STL surface data to display.")
            canvas.finish()
            self._show()
            return 0, 0
        canvas.add_polydata(poly_data, color=(0.58, 0.62, 0.66), opacity=0.92, edge_color=(0.12, 0.12, 0.12))
        canvas.finish(points)
        self._show()
        return len(points), len(faces)

    def _reset_view(self, title: str) -> NativeVtkPreviewWidget:
        self._last_plot_title = title
        canvas = NativeVtkPreviewWidget(self, background=(1.0, 1.0, 1.0))
        self._tabs.addTab(canvas, self._tab_title(title))
        self._tabs.setCurrentWidget(canvas)
        return canvas

    def _show(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def _export_png(self) -> None:
        current_canvas = self._tabs.currentWidget()
        if not isinstance(current_canvas, NativeVtkPreviewWidget):
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
            current_canvas.save_png(Path(file_path))
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
            if not isinstance(canvas, NativeVtkPreviewWidget):
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
            canvas.save_png(path)
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

    def _points(self, poly_data) -> np.ndarray:
        vtk_points = poly_data.GetPoints()
        if vtk_points is None:
            return np.empty((0, 3), dtype=float)
        return vtk_to_numpy(vtk_points.GetData())

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

class NativeVtkViewerDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle("FoamDesk Native VTK 视图")
        self.resize(1120, 760)
        self._interactor_initialized = False
        self._render_backend_logged = False
        self._animation_render_callback: Callable[[int], None] | None = None
        self._animation_frame_count = 0
        self._animation_frame_index = 0
        self._animation_interval_ms = 800
        self._animation_loop = True
        self._geometry_assets = []
        self._slice_data = None
        self._combined_stl_polydata = None
        self._last_plot_fn = None
        self._current_plot_mode = ""
        self._streamline_display_state = None
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._advance_animation_frame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        toolbar = QHBoxLayout()
        self._near_stl_check = QCheckBox("仅显示STL附近")
        self._near_stl_check.setToolTip("仅显示 STL 几何表面附近的场数据")
        self._near_stl_check.toggled.connect(self._on_near_stl_toggled)
        toolbar.addWidget(self._near_stl_check)
        toolbar.addSpacing(8)
        self._streamline_count_label = QLabel("基线数")
        self._streamline_count_label.setVisible(False)
        toolbar.addWidget(self._streamline_count_label)
        self._streamline_count_spin = QSpinBox()
        self._streamline_count_spin.setRange(50, 12000)
        self._streamline_count_spin.setSingleStep(100)
        self._streamline_count_spin.setValue(2600)
        self._streamline_count_spin.setVisible(False)
        toolbar.addWidget(self._streamline_count_spin)
        self._streamline_count_apply_button = QPushButton("确定")
        self._streamline_count_apply_button.clicked.connect(self._on_streamline_count_confirmed)
        self._streamline_count_apply_button.setVisible(False)
        toolbar.addWidget(self._streamline_count_apply_button)
        toolbar.addSpacing(8)
        self._status_label = QLabel("原生 VTK 高质量渲染窗口")
        toolbar.addWidget(self._status_label)
        toolbar.addStretch(1)
        self._play_animation_button = QPushButton("播放动画")
        self._pause_animation_button = QPushButton("暂停动画")
        self._play_animation_button.clicked.connect(self.play_animation)
        self._pause_animation_button.clicked.connect(self.pause_animation)
        self._play_animation_button.setEnabled(False)
        self._pause_animation_button.setEnabled(False)
        toolbar.addWidget(self._play_animation_button)
        toolbar.addWidget(self._pause_animation_button)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        toolbar.addWidget(close_button)
        # slice controls in toolbar (hidden by default, shown when Slice mode is active)
        toolbar.addSpacing(8)
        self._slice_lbl = QLabel("切片|")
        self._slice_lbl.setVisible(False)
        toolbar.addWidget(self._slice_lbl)
        self._slice_axis_combo = QComboBox()
        self._slice_axis_combo.addItems(["自动", "X", "Y", "Z"])
        self._slice_axis_combo.currentTextChanged.connect(self._on_slice_ctrl_changed)
        self._slice_axis_combo.setVisible(False)
        toolbar.addWidget(self._slice_axis_combo)
        self._slice_pos_spin = QDoubleSpinBox()
        self._slice_pos_spin.setRange(0.0, 1.0)
        self._slice_pos_spin.setSingleStep(0.05)
        self._slice_pos_spin.setDecimals(2)
        self._slice_pos_spin.setValue(0.5)
        self._slice_pos_spin.valueChanged.connect(self._on_slice_ctrl_changed)
        self._slice_pos_spin.setVisible(False)
        toolbar.addWidget(self._slice_pos_spin)
        layout.addLayout(toolbar)

        self._vtk_widget = QVTKRenderWindowInteractor(self)
        self._vtk_widget.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        layout.addWidget(self._vtk_widget, 1)
        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(1.0, 1.0, 1.0)
        self._vtk_widget.GetRenderWindow().AddRenderer(self._renderer)
        self._interactor = self._vtk_widget.GetRenderWindow().GetInteractor()

    def _hide_slice_ctrls(self) -> None:
        self._slice_lbl.setVisible(False)
        self._slice_axis_combo.setVisible(False)
        self._slice_pos_spin.setVisible(False)

    def _show_slice_ctrls(self) -> None:
        self._slice_lbl.setVisible(True)
        self._slice_axis_combo.setVisible(True)
        self._slice_pos_spin.setVisible(True)

    def _set_near_stl_control_visible(self, visible: bool) -> None:
        self._near_stl_check.setVisible(visible)

    def _set_streamline_controls_visible(self, visible: bool) -> None:
        self._streamline_count_label.setVisible(visible)
        self._streamline_count_spin.setVisible(visible)
        self._streamline_count_apply_button.setVisible(visible)

    def set_geometry_assets(self, assets) -> None:
        self._geometry_assets = [
            asset
            for asset in assets
            if getattr(asset, "format", "").upper() == "STL"
            and getattr(asset, "stored_path", None) is not None
            and asset.stored_path.exists()
        ]

    def closeEvent(self, event) -> None:  # noqa: N802
        self.pause_animation()
        self._animation_render_callback = None
        self._renderer.RemoveAllViewProps()
        self._interactor_initialized = False
        self._render_backend_logged = False
        super().closeEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        QTimer.singleShot(0, self._initialize_interactor)

    def _initialize_interactor(self) -> None:
        if self._interactor_initialized:
            return
        if not self._vtk_widget.isVisible():
            return
        self._interactor.Initialize()
        self._interactor_initialized = True

    def set_animation_source(
        self,
        frame_count: int,
        render_callback: Callable[[int], None] | None,
        interval_ms: int = 800,
        loop: bool = True,
    ) -> None:
        self.pause_animation()
        self._animation_frame_count = max(0, int(frame_count))
        self._animation_frame_index = 0
        self._animation_interval_ms = max(16, int(interval_ms))
        self._animation_loop = bool(loop)
        self._animation_render_callback = render_callback if self._animation_frame_count > 1 else None
        enabled = self._animation_render_callback is not None
        self._play_animation_button.setEnabled(enabled)
        self._pause_animation_button.setEnabled(enabled)

    def play_animation(self) -> None:
        if self._animation_render_callback is None or self._animation_frame_count <= 1:
            return
        near_stl_checked = self._near_stl_check.isChecked()
        if self._animation_frame_index >= self._animation_frame_count - 1:
            self._animation_frame_index = 0
        first_frame = max(1, self._animation_frame_index)
        self.render_animation_frame(first_frame)
        self._restore_near_stl_checked(near_stl_checked)
        self._animation_frame_index = min(first_frame + 1, self._animation_frame_count - 1)
        self._animation_timer.start(self._animation_interval_ms)

    def pause_animation(self) -> None:
        self._animation_timer.stop()

    def _advance_animation_frame(self) -> None:
        if self._animation_render_callback is None or self._animation_frame_count <= 1:
            self.pause_animation()
            return
        self.render_animation_frame(self._animation_frame_index)
        if self._animation_loop:
            self._animation_frame_index = (self._animation_frame_index + 1) % self._animation_frame_count
        else:
            self._animation_frame_index = min(self._animation_frame_index + 1, self._animation_frame_count - 1)

    def render_animation_frame(self, frame_index: int) -> None:
        if self._animation_render_callback is not None:
            self._animation_render_callback(frame_index)

    def plot_surface(self, poly_data, field_array, scalar_range: tuple[float, float], label: str) -> None:
        self._current_plot_mode = "surface"
        self._reset_scene(f"Surface 表面云图：{label}")
        self._set_near_stl_control_visible(True)
        self._set_streamline_controls_visible(False)
        display_data = self._maybe_clip_near_stl(poly_data)
        array_name = self._scalar_array_name(display_data, field_array, label)
        display_data.GetPointData().SetActiveScalars(array_name)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(display_data)
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(array_name)
        mapper.SetScalarRange(*scalar_range)
        mapper.SetLookupTable(self._lookup_table(scalar_range))
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetProperty().SetSpecular(0.25)
        actor.GetProperty().SetSpecularPower(18)
        actor.GetProperty().SetOpacity(0.62)
        self._renderer.AddActor(actor)
        self._add_outline(display_data)
        self._add_flow_labels(display_data)
        self._add_scalar_bar(mapper.GetLookupTable(), label)
        self._last_plot_fn = (self.plot_surface, (poly_data, field_array, scalar_range, label))
        self._finish_scene(display_data)

    def _on_slice_ctrl_changed(self) -> None:
        if self._slice_data is None:
            return
        data = self._slice_data
        axis_name = self._slice_axis_combo.currentText().strip()
        pos = self._slice_pos_spin.value()
        self.plot_slice(
            data["poly_data"], data["field_array"], data["scalar_range"],
            data["label"], None if axis_name == "自动" else axis_name, pos,
        )

    def plot_slice(
        self,
        poly_data,
        field_array,
        scalar_range: tuple[float, float],
        label: str,
        axis_name: str | None,
        normalized_position: float,
    ) -> tuple[str, float]:
        self._current_plot_mode = "slice"
        self._slice_data = {
            "poly_data": poly_data, "field_array": field_array,
            "scalar_range": scalar_range, "label": label,
        }
        self._show_slice_ctrls()
        # update controls without triggering re-render
        self._slice_axis_combo.blockSignals(True)
        self._slice_pos_spin.blockSignals(True)
        if axis_name:
            idx = self._slice_axis_combo.findText(axis_name)
            if idx >= 0: self._slice_axis_combo.setCurrentIndex(idx)
        self._slice_pos_spin.setValue(normalized_position)
        self._slice_axis_combo.blockSignals(False)
        self._slice_pos_spin.blockSignals(False)
        self._reset_scene(f"Slice 切片：{label}")
        self._set_near_stl_control_visible(False)
        self._set_streamline_controls_visible(False)
        array_name = self._scalar_array_name(poly_data, field_array, label)
        axis, center = self._axis_and_center(poly_data, axis_name, normalized_position)
        plane = vtk.vtkPlane()
        origin = [(poly_data.GetBounds()[0] + poly_data.GetBounds()[1]) * 0.5,
                  (poly_data.GetBounds()[2] + poly_data.GetBounds()[3]) * 0.5,
                  (poly_data.GetBounds()[4] + poly_data.GetBounds()[5]) * 0.5]
        origin[axis] = center
        normal = [0.0, 0.0, 0.0]
        normal[axis] = 1.0
        plane.SetOrigin(*origin)
        plane.SetNormal(*normal)
        cutter = vtk.vtkCutter()
        cutter.SetInputDataObject(poly_data)
        cutter.SetCutFunction(plane)
        cutter.Update()
        slice_data = cutter.GetOutput()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(slice_data)
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(array_name)
        mapper.SetScalarRange(*scalar_range)
        mapper.SetLookupTable(self._lookup_table(scalar_range))
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetProperty().SetOpacity(0.98)
        self._renderer.AddActor(actor)
        self._add_outline(poly_data)
        self._add_flow_labels(poly_data)
        self._add_scalar_bar(mapper.GetLookupTable(), label)
        self._last_plot_fn = (self.plot_slice, (poly_data, field_array, scalar_range, label, axis_name, normalized_position))
        self._finish_scene(poly_data)
        return "XYZ"[axis], center

    def plot_iso_surface(self, poly_data, field_array, scalar_range: tuple[float, float], label: str) -> None:
        self._current_plot_mode = "iso_surface"
        self._reset_scene(f"Iso-surface 等值面：{label}")
        self._set_near_stl_control_visible(False)
        self._set_streamline_controls_visible(False)
        display_data = self._maybe_clip_near_stl(poly_data)
        array_name = self._scalar_array_name(display_data, field_array, label)
        display_data.GetPointData().SetActiveScalars(array_name)
        value_min, value_max = scalar_range
        if value_max <= value_min:
            value_max = value_min + 1.0
        contour = vtk.vtkContourFilter()
        contour.SetInputData(display_data)
        contour.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, array_name)
        for index, value in enumerate(np.linspace(value_min, value_max, 7)):
            contour.SetValue(index, float(value))
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(contour.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(array_name)
        mapper.SetScalarRange(value_min, value_max)
        mapper.SetLookupTable(self._lookup_table((value_min, value_max)))
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.72)
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetProperty().SetSpecular(0.35)
        actor.GetProperty().SetSpecularPower(24)
        self._renderer.AddActor(actor)
        self._add_outline(display_data)
        self._add_flow_labels(display_data)
        self._add_scalar_bar(mapper.GetLookupTable(), label)
        self._last_plot_fn = (self.plot_iso_surface, (poly_data, field_array, scalar_range, label))
        self._finish_scene(display_data)

    def plot_volume(self, poly_data, field_array, scalar_range: tuple[float, float], label: str) -> None:
        self._current_plot_mode = "volume"
        self._reset_scene(f"Volume 体渲染：{label}")
        self._set_near_stl_control_visible(False)
        self._set_streamline_controls_visible(False)
        display_data = self._maybe_clip_near_stl(poly_data)
        array_name = self._scalar_array_name(display_data, field_array, label)
        vmin, vmax = scalar_range
        if vmax <= vmin:
            vmax = vmin + 1.0
        lut = self._lookup_table(scalar_range)
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputDataObject(display_data)
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(array_name)
        mapper.SetScalarRange(vmin, vmax)
        mapper.SetLookupTable(lut)
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.6)
        actor.GetProperty().SetInterpolationToPhong()
        self._renderer.AddActor(actor)
        self._add_scalar_bar(lut, label)
        self._last_plot_fn = (self.plot_volume, (poly_data, field_array, scalar_range, label))
        self._finish_scene(display_data)

    def plot_streamlines(
        self,
        source_poly_data,
        streamline_poly_data,
        speed_range: tuple[float, float],
        inlet_label: str = "",
        stl_streamline_data=None,
    ) -> None:
        self._current_plot_mode = "streamline"
        self._streamline_display_state = {
            "source_poly_data": source_poly_data,
            "inlet": (streamline_poly_data, speed_range, inlet_label),
            "stl": stl_streamline_data,
        }
        self._plot_current_streamline_dataset()

    def _plot_current_streamline_dataset(self) -> None:
        if not self._streamline_display_state:
            return
        source_poly_data = self._streamline_display_state["source_poly_data"]
        use_stl = self._near_stl_check.isChecked() and self._streamline_display_state.get("stl") is not None
        streamline_poly_data, speed_range, label = self._streamline_display_state["stl" if use_stl else "inlet"]
        title_prefix = "几何体表面" if use_stl else "入口面"
        title = f"Streamline 流线：{title_prefix}={label}，按 |U| 着色"
        self._reset_scene(title)
        self._set_near_stl_control_visible(True)
        self._set_streamline_controls_visible(True)
        speed_array_name = self._streamline_speed_array_name(streamline_poly_data)
        value_min, value_max = speed_range
        if value_max <= value_min:
            value_max = value_min + 1.0
        lookup_table = self._lookup_table((value_min, value_max))
        self._configure_streamline_growth_animation(
            streamline_poly_data,
            source_poly_data,
            speed_array_name,
            lookup_table,
            (value_min, value_max),
        )
        self._add_scalar_bar(lookup_table, "|U| (m/s)")
        camera_target = self._combined_stl_polydata if self._combined_stl_polydata is not None else source_poly_data
        self._last_plot_fn = (self._plot_current_streamline_dataset, ())
        self._finish_scene(camera_target, zoom=1.55)

    def _reset_scene(self, status: str) -> None:
        self._status_label.setText(status)
        self._renderer.RemoveAllViewProps()
        self._renderer.SetBackground(1.0, 1.0, 1.0)
        self._add_geometry_assets()
        # Only hide slice controls if switching to non-slice mode
        if not status.startswith("Slice"):
            self._hide_slice_ctrls()
            self._slice_data = None

    def _finish_scene(self, poly_data, zoom: float = 1.25) -> None:
        bounds = poly_data.GetBounds() if poly_data is not None else None
        if bounds is not None and all(np.isfinite(bounds)):
            self._renderer.ResetCamera(*bounds)
        else:
            self._renderer.ResetCamera()
        camera = self._renderer.GetActiveCamera()
        camera.Azimuth(-35)
        camera.Elevation(18)
        camera.Zoom(zoom)
        self._renderer.ResetCameraClippingRange()
        self.show()
        self.raise_()
        self.activateWindow()
        QTimer.singleShot(0, self._render_window)

    def _render_window(self) -> None:
        if not self.isVisible() or not self._vtk_widget.isVisible():
            return
        self._initialize_interactor()
        render_window = self._vtk_widget.GetRenderWindow()
        render_window.Render()
        if not self._render_backend_logged:
            _log_vtk_render_backend(render_window, "native result viewer")
            self._render_backend_logged = True

    def _lookup_table(self, scalar_range: tuple[float, float]):
        table = vtk.vtkLookupTable()
        table.SetNumberOfTableValues(256)
        table.SetRange(*scalar_range)
        cmap = colormaps["turbo"]
        for index in range(256):
            r, g, b, a = cmap(index / 255.0)
            table.SetTableValue(index, float(r), float(g), float(b), float(a))
        table.Build()
        return table

    def _add_outline(self, poly_data) -> None:
        outline = vtk.vtkOutlineFilter()
        outline.SetInputData(poly_data)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(outline.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.05, 0.10, 0.18)
        actor.GetProperty().SetOpacity(0.55)
        actor.GetProperty().SetLineWidth(1.4)
        self._renderer.AddActor(actor)

    def _add_geometry_assets(self) -> None:
        stl_pieces = []
        for asset in self._geometry_assets:
            try:
                reader = vtkSTLReader()
                reader.SetFileName(str(asset.stored_path))
                reader.Update()
                poly_data = reader.GetOutput()
                if poly_data is None or poly_data.GetNumberOfPoints() == 0:
                    continue
                stl_pieces.append(poly_data)
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(poly_data)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(0.58, 0.62, 0.66)
                actor.GetProperty().SetOpacity(0.35)
                actor.GetProperty().SetInterpolationToPhong()
                actor.GetProperty().SetSpecular(0.2)
                actor.GetProperty().SetSpecularPower(12)
                actor.GetProperty().EdgeVisibilityOn()
                actor.GetProperty().SetEdgeColor(0.12, 0.12, 0.12)
                actor.GetProperty().SetLineWidth(0.7)
                self._renderer.AddActor(actor)
            except (OSError, ValueError, RuntimeError):
                continue
        if stl_pieces:
            append_filter = vtk.vtkAppendPolyData()
            for pd in stl_pieces:
                append_filter.AddInputData(pd)
            append_filter.Update()
            self._combined_stl_polydata = append_filter.GetOutput()
        else:
            self._combined_stl_polydata = None

    def _on_near_stl_toggled(self, _checked: bool) -> None:
        """Re-render with current data to apply/remove STL proximity clip."""
        if self._current_plot_mode == "slice":
            return
        if self._current_plot_mode == "streamline":
            self._on_streamline_near_stl_toggled(_checked)
            return
        if self._last_plot_fn is not None:
            fn, args = self._last_plot_fn
            fn(*args)

    def _on_streamline_near_stl_toggled(self, _checked: bool) -> None:
        """Switch Streamline mode between inlet seeds and STL-near seeds."""
        self._plot_current_streamline_dataset()

    def _restore_near_stl_checked(self, checked: bool) -> None:
        if self._near_stl_check.isChecked() == checked:
            return
        self._near_stl_check.blockSignals(True)
        self._near_stl_check.setChecked(checked)
        self._near_stl_check.blockSignals(False)

    def _on_streamline_count_confirmed(self) -> None:
        if self._current_plot_mode != "streamline":
            return
        near_stl_checked = self._near_stl_check.isChecked()
        was_running = self._animation_timer.isActive()
        self._restore_near_stl_checked(near_stl_checked)
        self._plot_current_streamline_dataset()
        self._restore_near_stl_checked(near_stl_checked)
        if was_running:
            self.play_animation()

    def _maybe_clip_near_stl(self, input_data):
        """If '仅显示STL附近' is checked, clip input to STL proximity."""
        if not self._near_stl_check.isChecked() or self._combined_stl_polydata is None:
            return input_data
        try:
            stl_bounds = self._combined_stl_polydata.GetBounds()
            stl_diag = np.sqrt(
                (stl_bounds[1] - stl_bounds[0]) ** 2 +
                (stl_bounds[3] - stl_bounds[2]) ** 2 +
                (stl_bounds[5] - stl_bounds[4]) ** 2
            )
            # threshold: 5% of STL diagonal, min 0.005
            threshold = max(stl_diag * 0.05, 0.005)
            implicit = vtk.vtkImplicitPolyDataDistance()
            implicit.SetInput(self._combined_stl_polydata)
            # Convert to polydata if needed
            if hasattr(input_data, 'GetPolys'):
                pd_input = input_data
            else:
                geom = vtk.vtkGeometryFilter()
                geom.SetInputDataObject(input_data)
                geom.Update()
                pd_input = geom.GetOutput()
                if pd_input is None or pd_input.GetNumberOfPoints() == 0:
                    return input_data
            clip = vtk.vtkClipPolyData()
            clip.SetInputData(pd_input)
            clip.SetClipFunction(implicit)
            clip.SetValue(threshold)
            clip.SetInsideOut(True)   # keep inside (distance < threshold = near STL)
            clip.Update()
            clipped = clip.GetOutput()
            if clipped is None or clipped.GetNumberOfPoints() == 0:
                return input_data
            return clipped
        except Exception:
            return input_data

    def _add_scalar_bar(self, lookup_table, label: str) -> None:
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(lookup_table)
        scalar_bar.SetTitle(label)
        scalar_bar.SetOrientationToHorizontal()
        scalar_bar.SetNumberOfLabels(6)
        scalar_bar.SetWidth(0.62)
        scalar_bar.SetHeight(0.10)
        scalar_bar.SetPosition(0.20, 0.03)
        scalar_bar.GetTitleTextProperty().SetColor(0.0, 0.0, 0.0)
        scalar_bar.GetTitleTextProperty().SetBold(True)
        scalar_bar.GetLabelTextProperty().SetColor(0.0, 0.0, 0.0)
        self._renderer.AddActor2D(scalar_bar)

    def set_inlet_outlet_positions(self, inlet_centers: list, outlet_centers: list) -> None:
        self._inlet_positions = inlet_centers
        self._outlet_positions = outlet_centers

    def _add_flow_labels(self, poly_data=None) -> None:
        inlet_centers = getattr(self, '_inlet_positions', None) or []
        outlet_centers = getattr(self, '_outlet_positions', None) or []
        if inlet_centers or outlet_centers:
            for pos in inlet_centers:
                self._add_world_label("INLET", pos, (0.0, 0.18, 1.0))
            for pos in outlet_centers:
                self._add_world_label("OUTLET", pos, (1.0, 0.0, 0.0))
            return
        # fallback: longest-axis heuristic
        if poly_data is None:
            return
        bounds = poly_data.GetBounds()
        if bounds is None or not all(np.isfinite(bounds)):
            return
        spans = np.array(
            [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]],
            dtype=float,
        )
        axis = int(np.argmax(spans))
        center = np.array(
            [
                (bounds[0] + bounds[1]) * 0.5,
                (bounds[2] + bounds[3]) * 0.5,
                (bounds[4] + bounds[5]) * 0.5,
            ],
            dtype=float,
        )
        label_offset_axis = int(np.argmax([spans[index] if index != axis else -1.0 for index in range(3)]))
        offset = max(float(spans[label_offset_axis]) * 0.08, float(spans.max()) * 0.025, 1e-6)
        inlet_position = center.copy()
        outlet_position = center.copy()
        inlet_position[axis] = bounds[axis * 2]
        outlet_position[axis] = bounds[axis * 2 + 1]
        inlet_position[label_offset_axis] -= offset
        outlet_position[label_offset_axis] -= offset
        self._add_world_label("INLET", inlet_position, (0.0, 0.18, 1.0))
        self._add_world_label("OUTLET", outlet_position, (1.0, 0.0, 0.0))

    def _add_world_label(self, text: str, position: np.ndarray, color: tuple[float, float, float]) -> None:
        actor = vtk.vtkBillboardTextActor3D()
        actor.SetInput(text)
        actor.SetPosition(float(position[0]), float(position[1]), float(position[2]))
        actor.GetTextProperty().SetColor(*color)
        actor.GetTextProperty().SetBold(True)
        actor.GetTextProperty().SetFontSize(24)
        actor.GetTextProperty().ShadowOff()
        self._renderer.AddActor(actor)

    def _axis_and_center(self, poly_data, axis_name: str | None, normalized_position: float) -> tuple[int, float]:
        bounds = poly_data.GetBounds()
        axis = "XYZ".find(axis_name or "")
        if axis < 0:
            spans = [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]
            axis = int(np.argmin(spans))
        normalized_position = min(max(float(normalized_position), 0.0), 1.0)
        axis_min = bounds[axis * 2]
        axis_max = bounds[axis * 2 + 1]
        return axis, axis_min + (axis_max - axis_min) * normalized_position

    def _scalar_array_name(self, poly_data, field_array, label: str) -> str:
        array_name = field_array.GetName() or label
        if field_array.GetNumberOfComponents() == 1:
            return array_name
        values = vtk_to_numpy(field_array)
        if values.ndim == 1:
            scalar_values = values
        else:
            scalar_values = np.linalg.norm(values[:, : min(values.shape[1], 3)], axis=1)
        scalar_name = f"mag({array_name})"
        scalar_array = numpy_to_vtk(scalar_values.astype(float), deep=True)
        scalar_array.SetName(scalar_name)
        poly_data.GetPointData().AddArray(scalar_array)
        return scalar_name

    def _add_streamline_tubes(
        self,
        streamline_poly_data,
        source_poly_data,
        speed_array_name: str,
        lookup_table,
        scalar_range: tuple[float, float],
    ) -> None:
        if streamline_poly_data is None or streamline_poly_data.GetNumberOfLines() == 0:
            return
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(streamline_poly_data)
        tube.SetRadius(max(self._streamline_tube_radius(source_poly_data) * 1.45, 1e-6))
        tube.SetNumberOfSides(12)
        tube.CappingOn()
        tube.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(speed_array_name)
        mapper.SetScalarRange(*scalar_range)
        mapper.SetLookupTable(lookup_table)
        mapper.ScalarVisibilityOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.92)
        actor.GetProperty().SetAmbient(0.28)
        actor.GetProperty().SetDiffuse(0.78)
        actor.GetProperty().SetSpecular(0.32)
        actor.GetProperty().SetSpecularPower(18)
        actor.GetProperty().SetInterpolationToPhong()
        self._renderer.AddActor(actor)

    def _configure_streamline_growth_animation(
        self,
        streamline_poly_data,
        source_poly_data,
        speed_array_name: str,
        lookup_table,
        scalar_range: tuple[float, float],
    ) -> None:
        paths = self._streamline_path_records(streamline_poly_data, source_poly_data)
        if not paths:
            self.set_animation_source(0, None)
            return

        animated_points = vtk.vtkPoints()
        animated_lines = vtk.vtkCellArray()
        animated_speeds = vtk.vtkFloatArray()
        animated_speeds.SetName(speed_array_name or "U_mag")
        animated_poly_data = vtk.vtkPolyData()
        animated_poly_data.SetPoints(animated_points)
        animated_poly_data.SetLines(animated_lines)
        animated_poly_data.GetPointData().AddArray(animated_speeds)
        animated_poly_data.GetPointData().SetActiveScalars(animated_speeds.GetName())

        tube = vtk.vtkTubeFilter()
        tube.SetInputData(animated_poly_data)
        tube.SetRadius(max(self._streamline_tube_radius(source_poly_data) * 1.45, 1e-6))
        tube.SetNumberOfSides(12)
        tube.CappingOn()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(animated_speeds.GetName())
        mapper.SetScalarRange(*scalar_range)
        mapper.SetLookupTable(lookup_table)
        mapper.ScalarVisibilityOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.92)
        actor.GetProperty().SetAmbient(0.28)
        actor.GetProperty().SetDiffuse(0.78)
        actor.GetProperty().SetSpecular(0.32)
        actor.GetProperty().SetSpecularPower(18)
        actor.GetProperty().SetInterpolationToPhong()
        self._renderer.AddActor(actor)

        frame_count = 260

        def render_growth_frame(frame_index: int) -> None:
            progress = min(max(float(frame_index) / float(frame_count - 1), 0.0), 1.0)
            self._build_partial_streamline_polydata(paths, progress, animated_points, animated_lines, animated_speeds)
            animated_points.Modified()
            animated_lines.Modified()
            animated_speeds.Modified()
            animated_poly_data.Modified()
            tube.Modified()
            self._render_window()

        self.set_animation_source(frame_count, render_growth_frame, interval_ms=35, loop=False)

    def _build_partial_streamline_polydata(
        self,
        paths: list[dict[str, np.ndarray | float]],
        progress: float,
        vtk_points,
        vtk_lines,
        vtk_speeds,
    ) -> None:
        vtk_points.Reset()
        vtk_lines.Reset()
        vtk_speeds.Reset()
        if progress <= 0.0:
            return
        for path in paths:
            points = path["points"]
            speeds = path["speeds"]
            total_length = float(path["total_length"])
            if not isinstance(points, np.ndarray) or not isinstance(speeds, np.ndarray):
                continue
            if len(points) < 2 or total_length <= 1e-12:
                continue
            reveal_distance = total_length * progress
            partial_points: list[np.ndarray] = [points[0]]
            partial_speeds: list[float] = [float(speeds[0]) if speeds.size else 0.0]
            walked = 0.0
            for index in range(len(points) - 1):
                start = points[index]
                end = points[index + 1]
                segment_length = float(np.linalg.norm(end - start))
                if segment_length <= 1e-12:
                    continue
                next_walked = walked + segment_length
                if next_walked <= reveal_distance:
                    partial_points.append(end)
                    partial_speeds.append(float(speeds[min(index + 1, len(speeds) - 1)]))
                    walked = next_walked
                    continue
                local_phase = max(min((reveal_distance - walked) / segment_length, 1.0), 0.0)
                if local_phase > 1e-5:
                    partial_points.append(start * (1.0 - local_phase) + end * local_phase)
                    start_speed = float(speeds[min(index, len(speeds) - 1)])
                    end_speed = float(speeds[min(index + 1, len(speeds) - 1)])
                    partial_speeds.append(start_speed * (1.0 - local_phase) + end_speed * local_phase)
                break
            if len(partial_points) < 2:
                continue
            polyline = vtk.vtkPolyLine()
            polyline.GetPointIds().SetNumberOfIds(len(partial_points))
            for point_index, point in enumerate(partial_points):
                new_id = vtk_points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
                vtk_speeds.InsertNextValue(float(partial_speeds[point_index]))
                polyline.GetPointIds().SetId(point_index, new_id)
            vtk_lines.InsertNextCell(polyline)

    def _streamline_speed_array_name(self, streamline_poly_data) -> str:
        existing = streamline_poly_data.GetPointData().GetArray("U_mag")
        if existing is not None:
            streamline_poly_data.GetPointData().SetActiveScalars("U_mag")
            return "U_mag"
        vector_array = streamline_poly_data.GetPointData().GetArray("U")
        if vector_array is None:
            scalars = streamline_poly_data.GetPointData().GetScalars()
            if scalars is not None and scalars.GetName():
                return scalars.GetName()
            fallback_values = np.zeros(streamline_poly_data.GetNumberOfPoints(), dtype=float)
            fallback_array = numpy_to_vtk(fallback_values, deep=True)
            fallback_array.SetName("U_mag")
            streamline_poly_data.GetPointData().AddArray(fallback_array)
            streamline_poly_data.GetPointData().SetActiveScalars("U_mag")
            return "U_mag"
        vectors = vtk_to_numpy(vector_array)
        if vectors.ndim == 1:
            speed_values = np.abs(vectors.astype(float))
        else:
            speed_values = np.linalg.norm(vectors[:, : min(vectors.shape[1], 3)], axis=1)
        speed_array = numpy_to_vtk(speed_values.astype(float), deep=True)
        speed_array.SetName("U_mag")
        streamline_poly_data.GetPointData().AddArray(speed_array)
        streamline_poly_data.GetPointData().SetActiveScalars("U_mag")
        return "U_mag"

    def _streamline_tube_radius(self, poly_data) -> float:
        bounds = poly_data.GetBounds()
        if bounds is None or not all(np.isfinite(bounds)):
            return 0.004
        spans = np.array(
            [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]],
            dtype=float,
        )
        max_span = max(float(spans.max()), 1e-9)
        usable_spans = np.array([span for span in spans if span > max_span * 1e-5], dtype=float)
        if usable_spans.size == 0:
            return max_span * 0.0008
        section_spans = usable_spans[usable_spans >= max_span * 0.02]
        section_scale = float(section_spans.min() if section_spans.size else usable_spans.min())
        return max(min(section_scale * 0.006, max_span * 0.0014), max_span * 0.00035)

    def _streamline_path_records(self, streamline_poly_data, source_poly_data=None) -> list[dict[str, np.ndarray | float]]:
        vtk_points = streamline_poly_data.GetPoints()
        lines = streamline_poly_data.GetLines()
        if vtk_points is None or lines is None:
            return []
        points = vtk_to_numpy(vtk_points.GetData())
        raw_lines = vtk_to_numpy(lines.GetData())
        speed_array = streamline_poly_data.GetPointData().GetArray("U_mag")
        if speed_array is not None:
            all_speeds = vtk_to_numpy(speed_array).astype(float)
        else:
            vector_array = streamline_poly_data.GetPointData().GetArray("U")
            if vector_array is not None:
                vectors = vtk_to_numpy(vector_array)
                all_speeds = np.linalg.norm(vectors[:, : min(vectors.shape[1], 3)], axis=1)
            else:
                all_speeds = np.zeros(streamline_poly_data.GetNumberOfPoints(), dtype=float)
        paths: list[dict[str, np.ndarray | float]] = []
        index = 0
        while index < len(raw_lines):
            count = int(raw_lines[index])
            index += 1
            ids = raw_lines[index : index + count].astype(int)
            index += count
            if count < 2:
                continue
            path_speeds = np.nan_to_num(all_speeds[ids], nan=0.0, posinf=0.0, neginf=0.0)
            path_points, path_speeds = self._orient_streamline_path_to_outlet(
                points[ids],
                np.maximum(path_speeds.astype(float), 0.0),
                streamline_poly_data,
                ids,
            )
            path_points, path_speeds = self._extend_streamline_tail(path_points, path_speeds, source_poly_data)
            total_length = self._path_length(path_points)
            if total_length <= 1e-12:
                continue
            paths.append(
                {
                    "points": path_points,
                    "speeds": path_speeds,
                    "total_length": float(total_length),
                }
            )
        filtered_paths = self._filter_visible_streamline_paths(paths)
        if filtered_paths:
            paths = filtered_paths
        max_paths = self._selected_streamline_count()
        if len(paths) <= max_paths:
            return paths
        selected = np.linspace(0, len(paths) - 1, max_paths, dtype=int)
        return [paths[int(path_index)] for path_index in selected]

    def _selected_streamline_count(self) -> int:
        if hasattr(self, "_streamline_count_spin"):
            return max(1, int(self._streamline_count_spin.value()))
        return 2600

    def _extend_streamline_tail(
        self,
        path_points: np.ndarray,
        path_speeds: np.ndarray,
        source_poly_data=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(path_points) < 3:
            return path_points, path_speeds
        tail_direction = path_points[-1] - path_points[-3]
        norm = float(np.linalg.norm(tail_direction))
        if norm <= 1e-12:
            return path_points, path_speeds
        tail_direction = tail_direction / norm
        domain_length = self._bounds_length(source_poly_data) if source_poly_data is not None else float(self._path_length(path_points))
        extension_length = max(domain_length * 0.22, norm * 0.8)
        extension_steps = 10
        extra_points = [
            path_points[-1] + tail_direction * extension_length * (step / extension_steps)
            for step in range(1, extension_steps + 1)
        ]
        extended_points = np.vstack([path_points, np.array(extra_points, dtype=float)])
        tail_speed = float(path_speeds[-1]) if path_speeds.size else 0.0
        extra_speeds = np.full(extension_steps, tail_speed, dtype=float)
        extended_speeds = np.concatenate([path_speeds, extra_speeds])
        return extended_points, extended_speeds

    def _filter_visible_streamline_paths(
        self,
        paths: list[dict[str, np.ndarray | float]],
    ) -> list[dict[str, np.ndarray | float]]:
        if not paths:
            return []
        lengths = np.array([float(path["total_length"]) for path in paths], dtype=float)
        longest = float(lengths.max()) if lengths.size else 0.0
        if longest <= 1e-12:
            return []
        min_length = max(longest * 0.18, 1e-6)
        filtered = []
        for path in paths:
            points = path["points"]
            speeds = path["speeds"]
            total_length = float(path["total_length"])
            if not isinstance(points, np.ndarray) or not isinstance(speeds, np.ndarray):
                continue
            if len(points) < 4 or total_length < min_length:
                continue
            displacement = float(np.linalg.norm(points[-1] - points[0]))
            if displacement < min_length * 0.35:
                continue
            if speeds.size and float(np.nanmax(speeds)) <= 1e-9:
                continue
            filtered.append(path)
        return filtered

    def _orient_streamline_path_to_outlet(
        self,
        path_points: np.ndarray,
        path_speeds: np.ndarray,
        streamline_poly_data,
        ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(path_points) < 2:
            return path_points, path_speeds
        outlet_target = self._streamline_outlet_target()
        if outlet_target is not None:
            start_distance = float(np.linalg.norm(path_points[0] - outlet_target))
            end_distance = float(np.linalg.norm(path_points[-1] - outlet_target))
            if end_distance > start_distance:
                return path_points[::-1].copy(), path_speeds[::-1].copy()
            return path_points, path_speeds

        vector_array = streamline_poly_data.GetPointData().GetArray("U")
        if vector_array is None:
            return path_points, path_speeds
        vectors = vtk_to_numpy(vector_array)
        if vectors.ndim != 2 or vectors.shape[1] < 3:
            return path_points, path_speeds
        line_vectors = vectors[ids, :3]
        valid = np.linalg.norm(line_vectors, axis=1) > 1e-12
        if not np.any(valid):
            return path_points, path_speeds
        mean_vector = line_vectors[valid].mean(axis=0)
        if float(np.dot(path_points[-1] - path_points[0], mean_vector)) < 0.0:
            return path_points[::-1].copy(), path_speeds[::-1].copy()
        return path_points, path_speeds

    def _streamline_outlet_target(self) -> np.ndarray | None:
        outlet_centers = getattr(self, "_outlet_positions", None) or []
        if not outlet_centers:
            return None
        return np.array(outlet_centers, dtype=float).mean(axis=0)

    def _point_and_speed_on_streamline_path(
        self,
        path_record: dict[str, np.ndarray | float],
        distance: float,
    ) -> tuple[np.ndarray, float]:
        points = path_record["points"]
        speeds = path_record["speeds"]
        if not isinstance(points, np.ndarray) or not isinstance(speeds, np.ndarray):
            return np.zeros(3, dtype=float), 0.0
        if len(points) == 1:
            return points[0], float(speeds[0]) if speeds.size else 0.0
        segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
        total = float(path_record["total_length"])
        if total <= 1e-12:
            return points[0], 0.0
        clamped_distance = min(max(float(distance), 0.0), total)
        cumulative = np.cumsum(segments)
        segment_index = int(np.searchsorted(cumulative, clamped_distance, side="right"))
        segment_index = min(segment_index, len(segments) - 1)
        previous = 0.0 if segment_index == 0 else float(cumulative[segment_index - 1])
        local_length = max(float(segments[segment_index]), 1e-12)
        local_phase = (clamped_distance - previous) / local_length
        point = points[segment_index] * (1.0 - local_phase) + points[segment_index + 1] * local_phase
        speed = float(speeds[segment_index] * (1.0 - local_phase) + speeds[segment_index + 1] * local_phase)
        return point, max(speed, 0.0)

    def _path_length(self, path: np.ndarray) -> float:
        if len(path) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

    def _bounds_length(self, poly_data) -> float:
        bounds = poly_data.GetBounds()
        if bounds is None or not all(np.isfinite(bounds)):
            return 1.0
        spans = np.array(
            [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]],
            dtype=float,
        )
        return max(float(spans.max()), 1.0)
