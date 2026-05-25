from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import vtk
from matplotlib import colormaps
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.util.numpy_support import vtk_to_numpy


class NativeVtkPreviewWidget(QWidget):
    def __init__(self, parent: QWidget | None = None, background: tuple[float, float, float] = (0.12, 0.12, 0.12)) -> None:
        super().__init__(parent)
        self._background = background
        self._interactor_initialized = False
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
        self._vtk_widget.GetRenderWindow().Render()

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
        self._animation_render_callback: Callable[[int], None] | None = None
        self._animation_frame_count = 0
        self._animation_frame_index = 0
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._advance_animation_frame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        toolbar = QHBoxLayout()
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
        layout.addLayout(toolbar)

        self._vtk_widget = QVTKRenderWindowInteractor(self)
        self._vtk_widget.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        layout.addWidget(self._vtk_widget, 1)
        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(1.0, 1.0, 1.0)
        self._vtk_widget.GetRenderWindow().AddRenderer(self._renderer)
        self._interactor = self._vtk_widget.GetRenderWindow().GetInteractor()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.pause_animation()
        self._animation_render_callback = None
        self._renderer.RemoveAllViewProps()
        self._interactor_initialized = False
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

    def set_animation_source(self, frame_count: int, render_callback: Callable[[int], None] | None) -> None:
        self.pause_animation()
        self._animation_frame_count = max(0, int(frame_count))
        self._animation_frame_index = 0
        self._animation_render_callback = render_callback if self._animation_frame_count > 1 else None
        enabled = self._animation_render_callback is not None
        self._play_animation_button.setEnabled(enabled)
        self._pause_animation_button.setEnabled(enabled)

    def play_animation(self) -> None:
        if self._animation_render_callback is None or self._animation_frame_count <= 1:
            return
        self._animation_frame_index = 0
        self.render_animation_frame(self._animation_frame_index)
        self._animation_frame_index = 1
        self._animation_timer.start(800)

    def pause_animation(self) -> None:
        self._animation_timer.stop()

    def _advance_animation_frame(self) -> None:
        if self._animation_render_callback is None or self._animation_frame_count <= 1:
            self.pause_animation()
            return
        self.render_animation_frame(self._animation_frame_index)
        self._animation_frame_index = (self._animation_frame_index + 1) % self._animation_frame_count

    def render_animation_frame(self, frame_index: int) -> None:
        if self._animation_render_callback is not None:
            self._animation_render_callback(frame_index)

    def plot_surface(self, poly_data, field_array, scalar_range: tuple[float, float], label: str) -> None:
        self._reset_scene(f"Surface 表面云图：{label}")
        array_name = self._scalar_array_name(poly_data, field_array, label)
        poly_data.GetPointData().SetActiveScalars(array_name)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(array_name)
        mapper.SetScalarRange(*scalar_range)
        mapper.SetLookupTable(self._lookup_table(scalar_range))
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetProperty().SetSpecular(0.25)
        actor.GetProperty().SetSpecularPower(18)
        self._renderer.AddActor(actor)
        self._add_outline(poly_data)
        self._add_flow_labels()
        self._add_scalar_bar(mapper.GetLookupTable(), label)
        self._finish_scene(poly_data)

    def plot_slice(
        self,
        poly_data,
        field_array,
        scalar_range: tuple[float, float],
        label: str,
        axis_name: str | None,
        normalized_position: float,
    ) -> tuple[str, float]:
        self._reset_scene(f"Slice 切片：{label}")
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
        cutter.SetInputData(poly_data)
        cutter.SetCutFunction(plane)
        cutter.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cutter.GetOutputPort())
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
        self._add_flow_labels()
        self._add_scalar_bar(mapper.GetLookupTable(), label)
        self._finish_scene(poly_data)
        return "XYZ"[axis], center

    def plot_contour(
        self,
        poly_data,
        field_array,
        scalar_range: tuple[float, float],
        label: str,
        axis_name: str | None,
        normalized_position: float,
    ) -> tuple[str, float]:
        self._reset_scene(f"Contour 等值线：{label}")
        array_name = self._scalar_array_name(poly_data, field_array, label)
        axis, center = self._axis_and_center(poly_data, axis_name, normalized_position)
        bounds = poly_data.GetBounds()
        plane = vtk.vtkPlane()
        origin = [(bounds[0] + bounds[1]) * 0.5, (bounds[2] + bounds[3]) * 0.5, (bounds[4] + bounds[5]) * 0.5]
        origin[axis] = center
        normal = [0.0, 0.0, 0.0]
        normal[axis] = 1.0
        plane.SetOrigin(*origin)
        plane.SetNormal(*normal)
        cutter = vtk.vtkCutter()
        cutter.SetInputData(poly_data)
        cutter.SetCutFunction(plane)
        contour = vtk.vtkContourFilter()
        contour.SetInputConnection(cutter.GetOutputPort())
        contour.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, array_name)
        value_min, value_max = scalar_range
        if value_max <= value_min:
            value_max = value_min + 1.0
        contour.GenerateValues(22, value_min, value_max)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(contour.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(array_name)
        mapper.SetScalarRange(value_min, value_max)
        mapper.SetLookupTable(self._lookup_table((value_min, value_max)))
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(2.0)
        self._renderer.AddActor(actor)
        self._add_outline(poly_data)
        self._add_flow_labels()
        self._add_scalar_bar(mapper.GetLookupTable(), label)
        self._finish_scene(poly_data)
        return "XYZ"[axis], center

    def plot_vectors(self, poly_data, vector_array, limit: int = 900) -> None:
        self._reset_scene("Vector 矢量箭头：|U|")
        vector_name = vector_array.GetName()
        poly_data.GetPointData().SetActiveVectors(vector_name)
        speeds = vtk_to_numpy(vector_array)
        speed_values = np.linalg.norm(speeds[:, :3], axis=1) if speeds.ndim == 2 and speeds.shape[1] >= 3 else np.array([0.0, 1.0])
        speed_range = (float(speed_values.min()), float(speed_values.max()))
        if speed_range[0] == speed_range[1]:
            speed_range = (speed_range[0] - 1.0, speed_range[1] + 1.0)

        mask = vtk.vtkMaskPoints()
        mask.SetInputData(poly_data)
        mask.SetMaximumNumberOfPoints(limit)
        mask.RandomModeOn()
        arrow = vtk.vtkArrowSource()
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputConnection(mask.GetOutputPort())
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetVectorModeToUseVector()
        glyph.SetScaleModeToScaleByVector()
        glyph.SetScaleFactor(0.08)
        glyph.OrientOn()
        glyph.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.SetScalarRange(*speed_range)
        mapper.SetLookupTable(self._lookup_table(speed_range))
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToPhong()
        self._renderer.AddActor(actor)
        self._add_outline(poly_data)
        self._add_flow_labels()
        self._add_scalar_bar(mapper.GetLookupTable(), "|U| (m/s)")
        self._finish_scene(poly_data)

    def plot_iso_surface(self, poly_data, field_array, scalar_range: tuple[float, float], label: str) -> None:
        self._reset_scene(f"Iso-surface 等值面：{label}")
        array_name = self._scalar_array_name(poly_data, field_array, label)
        poly_data.GetPointData().SetActiveScalars(array_name)
        value_min, value_max = scalar_range
        if value_max <= value_min:
            value_max = value_min + 1.0
        contour = vtk.vtkContourFilter()
        contour.SetInputData(poly_data)
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
        self._add_outline(poly_data)
        self._add_flow_labels()
        self._add_scalar_bar(mapper.GetLookupTable(), label)
        self._finish_scene(poly_data)

    def plot_streamlines(self, source_poly_data, streamline_poly_data, speed_range: tuple[float, float]) -> None:
        self._reset_scene("Streamline 流线：|U|")
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(streamline_poly_data)
        tube.SetRadius(0.018)
        tube.SetNumberOfSides(14)
        tube.CappingOn()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        mapper.SetScalarRange(*speed_range)
        mapper.SetLookupTable(self._lookup_table(speed_range))
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetSpecularPower(18)
        self._renderer.AddActor(actor)
        self._add_outline(source_poly_data)
        self._add_flow_labels()
        self._add_scalar_bar(mapper.GetLookupTable(), "|U| (m/s)")
        self._finish_scene(source_poly_data)

    def _reset_scene(self, status: str) -> None:
        self._status_label.setText(status)
        self._renderer.RemoveAllViewProps()
        self._renderer.SetBackground(1.0, 1.0, 1.0)

    def _finish_scene(self, poly_data) -> None:
        self._renderer.ResetCamera()
        camera = self._renderer.GetActiveCamera()
        camera.Azimuth(-35)
        camera.Elevation(18)
        camera.Zoom(1.15)
        self._renderer.ResetCameraClippingRange()
        self.show()
        self.raise_()
        self.activateWindow()
        QTimer.singleShot(0, self._render_window)

    def _render_window(self) -> None:
        if not self.isVisible() or not self._vtk_widget.isVisible():
            return
        self._initialize_interactor()
        self._vtk_widget.GetRenderWindow().Render()

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

    def _add_flow_labels(self) -> None:
        inlet = vtk.vtkTextActor()
        inlet.SetInput("INLET")
        inlet.GetTextProperty().SetColor(0.0, 0.18, 1.0)
        inlet.GetTextProperty().SetBold(True)
        inlet.GetTextProperty().SetFontSize(24)
        inlet.SetPosition(32, 96)
        outlet = vtk.vtkTextActor()
        outlet.SetInput("OUTLET")
        outlet.GetTextProperty().SetColor(1.0, 0.0, 0.0)
        outlet.GetTextProperty().SetBold(True)
        outlet.GetTextProperty().SetFontSize(24)
        outlet.SetPosition(930, 96)
        self._renderer.AddActor2D(inlet)
        self._renderer.AddActor2D(outlet)

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
