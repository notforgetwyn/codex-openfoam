from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import numpy as np
import vtk
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidgetItem,
    QVBoxLayout,
)
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOGeometry import vtkSTLReader

from foamdesk.domain.models import SimulationParameters
from foamdesk.services.geometry_import_service import SnappyHexMeshSettings, StlTransform
from foamdesk.services.project_service import BoundaryConditionSettings, ComputationDomainTemplate
from foamdesk.ui.visualization_widgets import NativeVtkPreviewWidget


class GeometryLogicMixin:
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
        domain_tmpl = self._current_domain_template() if self._current_project is not None else None
        transform = self._read_stl_transform_dialog(Path(file_path), template=domain_tmpl)
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
        self._workspace_tabs.setCurrentIndex(self.TAB_MESH_GENERATION)
        self._set_status("STL 几何导入完成。")

    def _read_stl_transform_dialog(
        self,
        source_path: Path,
        initial_transform: StlTransform | None = None,
        template: ComputationDomainTemplate | None = None,
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
        preview_hint = QLabel("预览说明：半透明区域为当前计算域，蓝色几何是当前缩放和平移后的 STL 位置。")
        preview_hint.setWordWrap(True)
        preview_canvas = NativeVtkPreviewWidget(dialog, background=(0.12, 0.12, 0.12))
        preview_canvas.setMinimumHeight(320)

        def current_transform() -> StlTransform:
            return StlTransform(
                translate=(x_input.value(), y_input.value(), z_input.value()),
                scale=scale_input.value(),
                rotate_degrees=(rx_input.value(), ry_input.value(), rz_input.value()),
            )

        def refresh_preview() -> None:
            self._draw_stl_transform_preview(preview_canvas, source_path, current_transform(), template)

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
        canvas: NativeVtkPreviewWidget,
        source_path: Path,
        transform: StlTransform,
        template: ComputationDomainTemplate | None = None,
    ) -> None:
        canvas.clear((0.12, 0.12, 0.12))
        points_for_limits: list[np.ndarray] = []
        if template is not None:
            points_for_limits.append(self._draw_domain_template_vtk(canvas, template))
        else:
            points_for_limits.append(self._draw_box_domain_vtk(canvas, np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)))

        if self._current_project is not None:
            for asset in self._context.geometry_import_service.list_assets(self._current_project):
                if asset.format.upper() != "STL" or not asset.stored_path.exists():
                    continue
                try:
                    pts, fcs = self._read_stl_preview_mesh(asset.stored_path)
                    if pts.size and fcs.size:
                        canvas.add_polydata(
                            self._polydata_from_points_faces(pts, fcs),
                            color=(0.55, 0.55, 0.55),
                            opacity=0.35,
                            edge_color=(0.25, 0.25, 0.25),
                        )
                        points_for_limits.append(pts)
                except (OSError, ValueError):
                    pass

        try:
            points, faces = self._read_stl_preview_mesh(source_path)
        except (OSError, ValueError) as error:
            canvas.add_message(f"STL preview failed: {error}")
            canvas.finish(np.vstack(points_for_limits) if points_for_limits else None)
            return

        if points.size == 0 or faces.size == 0:
            canvas.add_message("No STL triangles to preview.")
            canvas.finish(np.vstack(points_for_limits) if points_for_limits else None)
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
        canvas.add_polydata(
            self._polydata_from_points_faces(transformed_points, faces),
            color=(0.25, 0.74, 1.0),
            opacity=0.72,
            edge_color=(0.03, 0.18, 0.28),
        )
        points_for_limits.append(transformed_points)
        canvas.finish(np.vstack(points_for_limits))

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

    def _polydata_from_points_faces(self, points: np.ndarray, faces: np.ndarray) -> vtk.vtkPolyData:
        vtk_points = vtk.vtkPoints()
        for point in points:
            vtk_points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
        cells = vtk.vtkCellArray()
        for face in faces:
            ids = np.asarray(face, dtype=int)
            if len(ids) < 3:
                continue
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(len(ids))
            for index, point_id in enumerate(ids):
                polygon.GetPointIds().SetId(index, int(point_id))
            cells.InsertNextCell(polygon)
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)
        poly_data.SetPolys(cells)
        return poly_data

    def _grid_surface_polydata(self, x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray) -> vtk.vtkPolyData:
        rows, cols = x_grid.shape
        points = np.column_stack([x_grid.reshape(-1), y_grid.reshape(-1), z_grid.reshape(-1)])
        faces = []
        for row in range(rows - 1):
            for col in range(cols - 1):
                p0 = row * cols + col
                faces.append(np.array([p0, p0 + 1, p0 + cols + 1, p0 + cols], dtype=int))
        return self._polydata_from_points_faces(points, np.array(faces, dtype=object))

    def _draw_box_domain_vtk(self, canvas: NativeVtkPreviewWidget, corners_or_bounds: np.ndarray) -> np.ndarray:
        if corners_or_bounds.shape == (2, 3):
            mins = corners_or_bounds[0]
            maxs = corners_or_bounds[1]
            corners = np.array(
                [
                    [mins[0], mins[1], mins[2]],
                    [maxs[0], mins[1], mins[2]],
                    [maxs[0], maxs[1], mins[2]],
                    [mins[0], maxs[1], mins[2]],
                    [mins[0], mins[1], maxs[2]],
                    [maxs[0], mins[1], maxs[2]],
                    [maxs[0], maxs[1], maxs[2]],
                    [mins[0], maxs[1], maxs[2]],
                ],
                dtype=float,
            )
        else:
            corners = np.asarray(corners_or_bounds, dtype=float)
        faces = [[0, 3, 7, 4], [1, 5, 6, 2], [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [3, 2, 6, 7]]
        colors = [(0.54, 0.82, 0.52), (0.96, 0.53, 0.44)] + [(0.31, 0.76, 1.0)] * 4
        opacities = [0.30, 0.30, 0.10, 0.10, 0.10, 0.10]
        for face, color, opacity in zip(faces, colors, opacities):
            canvas.add_polygon(corners[np.asarray(face, dtype=int)], color=color, opacity=opacity, edge_color=color)
        canvas.add_text("inlet", tuple(corners[0]), color=(0.54, 0.82, 0.52), size=14)
        canvas.add_text("outlet", tuple(corners[1]), color=(0.96, 0.53, 0.44), size=14)
        return corners

    def _draw_domain_template_vtk(self, canvas: NativeVtkPreviewWidget, template: ComputationDomainTemplate) -> np.ndarray:
        if template.shape == "pipe":
            return self._draw_pipe_domain_vtk(canvas, template)
        if template.shape == "bend":
            return self._draw_bend_domain_vtk(canvas, template)
        return self._draw_box_domain_vtk(canvas, np.array(self._context.project_service.domain_vertices(template), dtype=float))

    def _draw_pipe_domain_vtk(self, canvas: NativeVtkPreviewWidget, template: ComputationDomainTemplate) -> np.ndarray:
        length_x, length_y, length_z = template.size
        center_y = length_y / 2.0
        center_z = length_z / 2.0
        radius = min(length_y, length_z) * 0.42
        theta = np.linspace(0.0, 2.0 * np.pi, 96)
        x_values = np.linspace(0.0, length_x, 18)
        theta_grid, x_grid = np.meshgrid(theta, x_values)
        y_grid = center_y + radius * np.cos(theta_grid)
        z_grid = center_z + radius * np.sin(theta_grid)
        canvas.add_polydata(self._grid_surface_polydata(x_grid, y_grid, z_grid), color=(0.31, 0.76, 1.0), opacity=0.18, edge_color=None)
        inlet = np.column_stack([np.zeros_like(theta), center_y + radius * np.cos(theta), center_z + radius * np.sin(theta)])
        outlet = np.column_stack([np.full_like(theta, length_x), center_y + radius * np.cos(theta), center_z + radius * np.sin(theta)])
        canvas.add_polyline(inlet, color=(0.54, 0.82, 0.52), width=3.0)
        canvas.add_polyline(outlet, color=(0.96, 0.53, 0.44), width=3.0)
        for angle in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
            y = center_y + radius * np.cos(angle)
            z = center_z + radius * np.sin(angle)
            canvas.add_polyline(np.array([[0.0, y, z], [length_x, y, z]], dtype=float), color=(0.31, 0.76, 1.0), width=1.8)
        canvas.add_text("inlet", (0.0, center_y, center_z), color=(0.54, 0.82, 0.52), size=14)
        canvas.add_text("outlet", (length_x, center_y, center_z), color=(0.96, 0.53, 0.44), size=14)
        return np.vstack([inlet, outlet])

    def _draw_bend_domain_vtk(self, canvas: NativeVtkPreviewWidget, template: ComputationDomainTemplate) -> np.ndarray:
        length_x, length_y, height = template.size
        inner_radius = min(length_x, length_y) * 0.28
        outer_radius = min(length_x, length_y) * 0.62
        theta = np.linspace(0.0, np.pi / 2.0, 96)
        z_values = np.linspace(0.0, height, 8)
        theta_grid, z_grid = np.meshgrid(theta, z_values)
        points: list[np.ndarray] = []
        for radius in (inner_radius, outer_radius):
            x_grid = radius * np.cos(theta_grid)
            y_grid = radius * np.sin(theta_grid)
            canvas.add_polydata(self._grid_surface_polydata(x_grid, y_grid, z_grid), color=(0.31, 0.76, 1.0), opacity=0.16, edge_color=None)
        for radius in (inner_radius, outer_radius):
            for z_value in (0.0, height):
                curve = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, z_value)])
                canvas.add_polyline(curve, color=(0.31, 0.76, 1.0), width=1.8)
                points.append(curve)
        for angle in (0.0, np.pi / 2.0):
            color = (0.54, 0.82, 0.52) if angle == 0.0 else (0.96, 0.53, 0.44)
            for z_value in (0.0, height):
                canvas.add_polyline(
                    np.array(
                        [
                            [inner_radius * np.cos(angle), inner_radius * np.sin(angle), z_value],
                            [outer_radius * np.cos(angle), outer_radius * np.sin(angle), z_value],
                        ],
                        dtype=float,
                    ),
                    color=color,
                    width=2.4,
                )
            for radius in (inner_radius, outer_radius):
                canvas.add_polyline(
                    np.array(
                        [
                            [radius * np.cos(angle), radius * np.sin(angle), 0.0],
                            [radius * np.cos(angle), radius * np.sin(angle), height],
                        ],
                        dtype=float,
                    ),
                    color=color,
                    width=2.4,
                )
        canvas.add_text("inlet", ((inner_radius + outer_radius) / 2.0, 0.0, height / 2.0), color=(0.54, 0.82, 0.52), size=14)
        canvas.add_text("outlet", (0.0, (inner_radius + outer_radius) / 2.0, height / 2.0), color=(0.96, 0.53, 0.44), size=14)
        return np.vstack(points)

    def _open_domain_preview_dialog(self):
        template = self._selected_domain_template() if self._current_project is not None else self._context.project_service.domain_templates()[0]
        dialog = QDialog(self)
        dialog.setWindowTitle("计算域 3D 预览")
        dialog.resize(900, 700)
        dialog.setMinimumSize(600, 450)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        canvas = NativeVtkPreviewWidget(dialog, background=(0.12, 0.12, 0.12))
        layout.addWidget(canvas)
        domain_points = self._draw_domain_template_vtk(canvas, template)
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
                canvas.add_polydata(
                    self._polydata_from_points_faces(points, faces),
                    color=(0.25, 0.74, 1.0),
                    opacity=0.42,
                    edge_color=(0.03, 0.18, 0.28),
                )
                points_for_limits.append(points)
        canvas.finish(np.vstack(points_for_limits))
        dialog.exec()

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
            asset = self._context.geometry_import_service.import_stl(self._current_project, stl_path)
        except (OSError, ValueError) as error:
            self._show_error(f"一键导入模板 STL 失败：{error}")
            return
        self._append_log(f"模板 STL 已导入：{asset.name}")
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
        selected_asset = self._pick_stl_asset_dialog(assets, "编辑 STL 位置")
        if selected_asset is None:
            return
        base_path = Path(selected_asset.source_path)
        if not base_path.exists():
            base_path = selected_asset.stored_path
        domain_template = self._current_domain_template() if self._current_project is not None else None
        transform = self._read_stl_transform_dialog(base_path, selected_asset.transform or StlTransform(), template=domain_template)
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

    def _pick_stl_asset_dialog(self, assets, title="选择 STL"):
        if not assets:
            return None
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(420, 150)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("选择已导入的 STL："))
        combo = QComboBox()
        for asset in assets:
            combo.addItem(asset.name, asset)
        layout.addWidget(combo)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return combo.currentData()

    def _refresh_domain_preview(self) -> None:
        if not hasattr(self, "_domain_preview_canvas"):
            return
        canvas = self._domain_preview_canvas
        if not isinstance(canvas, NativeVtkPreviewWidget):
            return
        canvas.clear((0.12, 0.12, 0.12))
        template = self._selected_domain_template()
        domain_points = self._draw_domain_template_vtk(canvas, template)
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
                canvas.add_polydata(
                    self._polydata_from_points_faces(points, faces),
                    color=(0.25, 0.74, 1.0),
                    opacity=0.42,
                    edge_color=(0.03, 0.18, 0.28),
                )
                points_for_limits.append(points)
        else:
            canvas.add_message("Select a project first.")

        canvas.finish(np.vstack(points_for_limits))

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
        if not hasattr(self, "_snappy_min_refinement_input"):
            if self._current_project is not None:
                settings = self._context.geometry_import_service.load_snappy_settings(self._current_project)
                if settings is not None:
                    return settings
            return SnappyHexMeshSettings()
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
        self._refresh_mesh_generation_panel()
        self._workspace_tabs.setCurrentIndex(self.TAB_MESH_GENERATION)
        self._set_status("snappyHexMeshDict 生成完成。")

    def _select_stl_asset_for_snappy(self) -> str | None:
        selected_name = self._auto_stl_asset_for_snappy()
        if selected_name:
            self._append_log(f"自动选择 STL：{selected_name}")
            return selected_name
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

    def _auto_stl_asset_for_snappy(self) -> str | None:
        if self._current_project is None:
            return None
        assets = self._context.geometry_import_service.list_assets(self._current_project)
        stl_assets = [asset for asset in assets if asset.format.upper() == "STL" and asset.stored_path.exists()]
        if not stl_assets:
            return None
        if len(stl_assets) == 1:
            return stl_assets[0].name

        draw_assets = [
            asset
            for asset in stl_assets
            if asset.name.startswith("body_") or asset.stored_path.name.startswith("body_")
        ]
        if draw_assets:
            newest = max(draw_assets, key=lambda asset: asset.stored_path.stat().st_mtime)
            return newest.name
        return None

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

        self._workspace_tabs.setCurrentIndex(self.TAB_SOLVER_RUN)
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

        self._workspace_tabs.setCurrentIndex(self.TAB_SOLVER_RUN)
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

        self._workspace_tabs.setCurrentIndex(self.TAB_SOLVER_RUN)
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
        try:
            parameters = self._context.case_parameter_service.load(self._current_project)
        except (OSError, ValueError) as error:
            self._show_error(f"读取求解器配置失败：{error}")
            return

        block_mesh_dict = self._current_project.case_dir / "system" / "blockMeshDict"
        if not block_mesh_dict.exists():
            self._show_error("当前 Case 缺少 system/blockMeshDict，一键仿真需要先有背景网格配置。")
            return

        selected_name = self._auto_stl_asset_for_snappy()
        has_stl_geometry = selected_name is not None
        try:
            if has_stl_geometry:
                dict_path = self._context.geometry_import_service.generate_snappy_hex_mesh_dict(
                    self._current_project,
                    selected_name,
                    self._read_snappy_settings(),
                )
                field_patches = ("movingWall", "fixedWalls", "importedGeometry")
            else:
                dict_path = None
                field_patches = None
            synced_fields = self._context.project_service.ensure_solver_support_files(
                self._current_project,
                parameters,
                field_patches,
            )
            custom_boundary_files = self._apply_boundary_table_to_case(parameters)
        except (OSError, ValueError) as error:
            self._show_error(f"一键求解器准备失败：{error}")
            return

        self._workspace_tabs.setCurrentIndex(self.TAB_SOLVER_RUN)
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
            self._append_log("已同步求解边界字段：")
            self._append_log("\n".join(f"- {path}" for path in relative_fields))
        if custom_boundary_files:
            relative_files = [str(path.relative_to(self._current_project.case_dir)) for path in custom_boundary_files]
            self._append_log("已应用求解器准备页边界表：")
            self._append_log("\n".join(f"- {path}" for path in relative_files))
        if dict_path is not None:
            self._append_log(f"已自动生成 snappyHexMeshDict：{dict_path}")
        else:
            self._append_log("当前 Case 没有 STL，流水线将跳过 snappyHexMesh。")

        command_steps = ["echo FOAMDESK_STEP:blockMesh && blockMesh"]
        if has_stl_geometry:
            command_steps.append("echo FOAMDESK_STEP:snappyHexMesh && snappyHexMesh -overwrite")
        command_steps.extend(
            [
                "echo FOAMDESK_STEP:checkMesh && checkMesh",
                f"echo FOAMDESK_STEP:{shlex.quote(parameters.solver_name)} && {shlex.quote(parameters.solver_name)}",
            ]
        )
        command = (
            f"source {shlex.quote(status.env_script_path)} >/dev/null 2>&1 && "
            f"cd {shlex.quote(str(self._current_project.case_dir))} && "
            + " && ".join(command_steps)
        )
        if hasattr(self, "_solver_command_label"):
            display_steps = ["blockMesh"]
            if has_stl_geometry:
                display_steps.append("snappyHexMesh -overwrite")
            display_steps.extend(["checkMesh", parameters.solver_name])
            self._solver_command_label.setText(
                "执行命令：" + " && ".join(display_steps)
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
            "执行流程：生成 snappyHexMeshDict -> blockMesh -> snappyHexMesh -overwrite -> "
            f"checkMesh -> {parameters.solver_name}"
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
        selected_asset = self._pick_stl_asset_dialog(stl_assets, "预览 STL")
        if selected_asset is None:
            return
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


    def _refresh_mesh_generation_panel(self) -> None:
        if not hasattr(self, "_mesh_generation_text"):
            return
        if self._current_project is None:
            self._mesh_generation_status.setText("网格状态：未选择 Case")
            self._mesh_generation_text.setPlainText("请先新建或打开项目，并选择一个 Case。")
            return

        project = self._current_project
        case_dir = project.case_dir
        system_dir = case_dir / "system"
        mesh_dir = case_dir / "constant" / "polyMesh"
        tri_surface_dir = case_dir / "constant" / "triSurface"
        required_files = [
            system_dir / "blockMeshDict",
            system_dir / "snappyHexMeshDict",
        ]
        lines = [
            "网格生成状态",
            "",
            f"- 项目：{project.name}",
            f"- Case：{project.case_name}",
            f"- Case 路径：{case_dir}",
            "",
            "1. 输入物检查：",
        ]
        for path in required_files:
            lines.append(f"- {'OK' if path.exists() else '缺失'}：{path.relative_to(case_dir)}")

        try:
            assets = self._context.geometry_import_service.list_assets(project)
        except (OSError, ValueError) as error:
            assets = []
            lines.append(f"- STL 清单读取失败：{error}")
        if assets:
            lines.append(f"- STL 数量：{len(assets)}")
            for asset in assets:
                lines.append(f"  - {asset.name} -> {asset.stored_path.relative_to(case_dir)}")
        else:
            stl_files = sorted(tri_surface_dir.glob("*.stl")) if tri_surface_dir.exists() else []
            lines.append(f"- STL 文件：{', '.join(path.name for path in stl_files) if stl_files else '暂无'}")

        lines.extend(["", "2. 网格输出："])
        if mesh_dir.exists():
            mesh_files = ["points", "faces", "owner", "neighbour", "boundary"]
            for name in mesh_files:
                path = mesh_dir / name
                lines.append(f"- {'OK' if path.exists() else '缺失'}：constant/polyMesh/{name}")
        else:
            lines.append("- constant/polyMesh：未生成")

        lines.extend(["", "3. 操作建议："])
        if not (system_dir / "blockMeshDict").exists():
            lines.append("- 先进入“绘制几何”生成 blockMeshDict + STL。")
        elif assets and not (system_dir / "snappyHexMeshDict").exists():
            lines.append("- 点击“生成 snappyHexMeshDict”。")
        elif not mesh_dir.exists():
            lines.append("- 点击“一键生成网格”执行 blockMesh -> snappyHexMesh -> checkMesh。")
        else:
            lines.append("- 网格目录已存在，建议点击“运行 checkMesh”查看网格质量。")
            lines.append("- checkMesh 通过后，再进入“求解器准备”配置 U/p/物性。")

        if self._last_diagnostic_summary and self._last_diagnostic_summary != "暂无诊断。":
            lines.extend(["", "4. 最近网格/任务诊断：", self._last_diagnostic_summary])

        has_mesh = mesh_dir.exists()
        has_block = (system_dir / "blockMeshDict").exists()
        has_snappy = (system_dir / "snappyHexMeshDict").exists()
        self._mesh_generation_status.setText(
            f"网格状态：blockMeshDict={'OK' if has_block else '缺失'}，"
            f"snappyHexMeshDict={'OK' if has_snappy else '缺失'}，"
            f"polyMesh={'已生成' if has_mesh else '未生成'}"
        )
        self._mesh_generation_text.setPlainText("\n".join(lines))
