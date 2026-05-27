from __future__ import annotations

import json
import shlex
from pathlib import Path

import numpy as np
import vtk
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidgetItem,
    QVBoxLayout,
)
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOGeometry import vtkSTLReader

from foamdesk.domain.models import SimulationParameters
from foamdesk.services.geometry_import_service import SnappyHexMeshSettings
from foamdesk.services.project_service import ComputationDomainTemplate
from foamdesk.ui.visualization_widgets import NativeVtkPreviewWidget


class GeometryLogicMixin:
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

    def _default_draw_geometry_objects(self) -> list[dict]:
        dv = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
        return [{"name":"计算域","verts":list(dv),"edges":[],"block_v":[0,1,2,3,4,5,6,7],"is_domain":True,"nx":10,"ny":10,"nz":10,"grading":"1 1 1"}]

    def _draw_geometry_state_path(self) -> Path | None:
        if self._current_project is None:
            return None
        return self._current_project.case_dir / ".foamdesk" / "draw_geometry_state.json"

    def _normalize_draw_geometry_object(self, raw: dict, index: int) -> dict:
        fallback = self._default_draw_geometry_objects()[0] if index == 0 else {"name":f"几何体{index}","verts":[],"edges":[],"is_domain":False}
        if not isinstance(raw, dict):
            raw = fallback
        obj = dict(raw)
        obj["name"] = str(obj.get("name") or fallback["name"])
        verts = []
        for vert in obj.get("verts", []):
            if not isinstance(vert, (list, tuple)) or len(vert) < 3:
                continue
            try:
                verts.append((float(vert[0]), float(vert[1]), float(vert[2])))
            except (TypeError, ValueError):
                continue
        obj["verts"] = verts or list(fallback.get("verts", []))
        edges = []
        for edge in obj.get("edges", []):
            if not isinstance(edge, (list, tuple)) or len(edge) < 3:
                continue
            try:
                edges.append((str(edge[0]), int(edge[1]), int(edge[2]), str(edge[3]) if len(edge) > 3 else ""))
            except (TypeError, ValueError):
                continue
        obj["edges"] = edges
        if index == 0:
            obj["is_domain"] = True
            block_v = obj.get("block_v", fallback["block_v"])
            obj["block_v"] = [int(v) for v in block_v[:8]] if isinstance(block_v, list) else list(fallback["block_v"])
            while len(obj["block_v"]) < 8:
                obj["block_v"].append(len(obj["block_v"]))
            obj["nx"] = max(1, int(obj.get("nx", 10)))
            obj["ny"] = max(1, int(obj.get("ny", 10)))
            obj["nz"] = max(1, int(obj.get("nz", 10)))
            obj["grading"] = str(obj.get("grading") or "1 1 1")
        else:
            obj["is_domain"] = False
        return obj

    def _load_draw_geometry_state(self) -> bool:
        path = self._draw_geometry_state_path()
        if path is None or not path.exists() or not hasattr(self, "_obj_combo"):
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            self._append_log(f"绘制几何草稿读取失败：{error}")
            return False
        raw_objects = data.get("objects", [])
        if not isinstance(raw_objects, list) or not raw_objects:
            return False
        self._suspend_draw_geometry_persist = True
        try:
            self._geo_objects = [self._normalize_draw_geometry_object(obj, idx) for idx, obj in enumerate(raw_objects)]
            self._active_obj_idx = min(max(int(data.get("active_object_index", 0)), 0), len(self._geo_objects) - 1)
            boundary = data.get("boundary", {}) if isinstance(data.get("boundary", {}), dict) else {}
            if hasattr(self, "_geo_inlet"):
                self._geo_inlet.setText(str(boundary.get("inlet", "inlet")))
                self._geo_outlet.setText(str(boundary.get("outlet", "outlet")))
                self._geo_walls.setText(str(boundary.get("walls", "fixedWalls")))
            self._rebuild_obj_combo()
            self._load_active_object_to_ui()
            self._refresh_draw_geo_preview()
        finally:
            self._suspend_draw_geometry_persist = False
        self._append_log(f"已恢复绘制几何草稿：{path}")
        return True

    def _persist_draw_geometry_state(self) -> None:
        if getattr(self, "_suspend_draw_geometry_persist", False):
            return
        path = self._draw_geometry_state_path()
        if path is None or not hasattr(self, "_vertex_table") or not hasattr(self, "_geo_inlet"):
            return
        try:
            self._save_current_object()
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "active_object_index": self._active_obj_idx,
                "boundary": {
                    "inlet": self._geo_inlet.text().strip() or "inlet",
                    "outlet": self._geo_outlet.text().strip() or "outlet",
                    "walls": self._geo_walls.text().strip() or "fixedWalls",
                },
                "objects": self._geo_objects,
            }
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, ValueError) as error:
            self._append_log(f"绘制几何草稿保存失败：{error}")

    # --- Object management ---

    def _rebuild_obj_combo(self):
        self._obj_combo.blockSignals(True)
        self._obj_combo.clear()
        for obj in self._geo_objects:
            self._obj_combo.addItem(obj["name"])
        self._obj_combo.setCurrentIndex(self._active_obj_idx)
        self._obj_combo.blockSignals(False)
        is_domain = (self._active_obj_idx == 0)
        if hasattr(self, "_block_widget"):
            self._block_widget.setVisible(is_domain)
        if hasattr(self, "_boundary_widget"):
            self._boundary_widget.setVisible(is_domain)

    def _save_current_object(self):
        if not self._geo_objects: return
        obj = self._geo_objects[self._active_obj_idx]
        obj["verts"] = self._read_draw_geo_vertices()
        obj["edges"] = list(self._edge_defs)
        if self._active_obj_idx == 0:
            obj["block_v"] = [sb.value() for sb in self._block_vert_inputs]
            obj["nx"] = self._geo_nx.value()
            obj["ny"] = self._geo_ny.value()
            obj["nz"] = self._geo_nz.value()
            obj["grading"] = self._geo_grading.text().strip() or "1 1 1"

    def _switch_active_object(self, idx):
        if idx < 0 or idx >= len(self._geo_objects): return
        self._save_current_object()
        self._active_obj_idx = idx
        self._load_active_object_to_ui()
        self._rebuild_obj_combo()
        self._refresh_draw_geo_preview()
        self._persist_draw_geometry_state()

    def _load_active_object_to_ui(self):
        obj = self._geo_objects[self._active_obj_idx]
        self._edge_defs = list(obj.get("edges", []))
        self._vertex_table.blockSignals(True)
        while self._vertex_table.rowCount() > 0:
            self._vertex_table.removeRow(0)
        for x,y,z in obj["verts"]:
            self._add_vertex_row(x,y,z)
        self._vertex_table.blockSignals(False)
        self._update_edge_list()
        if self._active_obj_idx == 0:
            bv = obj.get("block_v",[0,1,2,3,4,5,6,7])
            for i,sb in enumerate(self._block_vert_inputs):
                sb.setValue(bv[i] if i<len(bv) else i)
            self._geo_nx.setValue(obj.get("nx",10))
            self._geo_ny.setValue(obj.get("ny",10))
            self._geo_nz.setValue(obj.get("nz",10))
            self._geo_grading.setText(obj.get("grading","1 1 1"))

    def _new_geo_object(self):
        self._save_current_object()
        name = "几何体" + str(len(self._geo_objects))
        domain_verts = self._geo_objects[0]["verts"]
        xs = [v[0] for v in domain_verts]; ys = [v[1] for v in domain_verts]; zs = [v[2] for v in domain_verts]
        s = 0.25  # half-size = 50% of domain = 0.25 from center
        dx = (max(xs)-min(xs))*s; dy = (max(ys)-min(ys))*s; dz = (max(zs)-min(zs))*s
        cx = (min(xs)+max(xs))/2; cy = (min(ys)+max(ys))/2; cz = (min(zs)+max(zs))/2
        dv = [(cx-dx,cy-dy,cz-dz),(cx+dx,cy-dy,cz-dz),(cx+dx,cy+dy,cz-dz),(cx-dx,cy+dy,cz-dz),
              (cx-dx,cy-dy,cz+dz),(cx+dx,cy-dy,cz+dz),(cx+dx,cy+dy,cz+dz),(cx-dx,cy+dy,cz+dz)]
        self._geo_objects.append({"name":name,"verts":list(dv),"edges":[],"is_domain":False})
        self._active_obj_idx = len(self._geo_objects) - 1
        self._load_active_object_to_ui()
        self._rebuild_obj_combo()
        self._refresh_draw_geo_preview()
        self._persist_draw_geometry_state()

    def _delete_geo_object(self):
        if self._active_obj_idx == 0:
            self._show_error("不能删除计算域。")
            return
        self._geo_objects.pop(self._active_obj_idx)
        self._active_obj_idx = 0
        self._load_active_object_to_ui()
        self._rebuild_obj_combo()
        self._refresh_draw_geo_preview()
        self._persist_draw_geometry_state()

    # --- Vertex helpers ---

    def _import_vertices_csv(self):
        fp, _ = QFileDialog.getOpenFileName(self, "导入顶点 CSV", "", "CSV (*.csv)")
        if not fp: return
        import csv
        with open(fp, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                clean = [v.strip() for v in row if v.strip()]
                if len(clean) >= 3:
                    try:
                        x, y, z = float(clean[0]), float(clean[1]), float(clean[2])
                        self._add_vertex_row(x, y, z)
                    except ValueError:
                        continue
        self._save_vertex_table()
        self._refresh_draw_geo_preview()
        self._persist_draw_geometry_state()

    def _import_edges_csv(self):
        fp, _ = QFileDialog.getOpenFileName(self, "导入边 CSV", "", "CSV (*.csv)")
        if not fp: return
        import csv
        with open(fp, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                clean = [v.strip() for v in row if v.strip()]
                if len(clean) < 3: continue
                etype = clean[0]
                if etype not in ("arc","spline","polyLine","BSpline"): continue
                try:
                    s = int(clean[1]); e = int(clean[2])
                    interp = " ".join(clean[3:]) if len(clean) > 3 else ""
                    self._edge_defs.append((etype, s, e, interp))
                except ValueError:
                    continue
        if self._geo_objects:
            self._geo_objects[self._active_obj_idx]["edges"] = list(self._edge_defs)
        self._update_edge_list()
        self._refresh_draw_geo_preview()
        self._persist_draw_geometry_state()

    def _clear_all_vertices(self):
        self._vertex_table.blockSignals(True)
        while self._vertex_table.rowCount() > 0:
            self._vertex_table.removeRow(0)
        self._vertex_table.blockSignals(False)
        self._save_vertex_table()
        self._refresh_draw_geo_preview()
        self._persist_draw_geometry_state()

    def _clear_all_edges(self):
        self._edge_defs = []
        if self._geo_objects:
            self._geo_objects[self._active_obj_idx]["edges"] = []
        self._update_edge_list()
        self._refresh_draw_geo_preview()
        self._persist_draw_geometry_state()

    def _add_vertex_row(self, x=0.0, y=0.0, z=0.0):
        r = self._vertex_table.rowCount()
        self._vertex_table.insertRow(r)
        for j,v in enumerate([x,y,z]):
            self._vertex_table.setItem(r, j, QTableWidgetItem(str(float(v))))

    def _read_draw_geo_vertices(self):
        verts = []
        for i in range(self._vertex_table.rowCount()):
            row = []
            for j in range(3):
                item = self._vertex_table.item(i,j)
                try: row.append(float(item.text()) if item else 0.0)
                except ValueError: row.append(0.0)
            verts.append(tuple(row))
        return verts

    def _save_vertex_table(self):
        if self._geo_objects:
            self._geo_objects[self._active_obj_idx]["verts"] = self._read_draw_geo_vertices()

    def _on_vertex_table_changed(self):
        self._save_vertex_table()
        self._refresh_draw_geo_preview()
        self._persist_draw_geometry_state()

    def _delete_selected_vertex(self):
        rows = set(i.row() for i in self._vertex_table.selectedIndexes())
        for r in sorted(rows, reverse=True):
            self._vertex_table.removeRow(r)
        self._save_vertex_table()
        self._refresh_draw_geo_preview()
        self._persist_draw_geometry_state()

    # --- Edge helpers ---

    def _add_edge_to_current(self):
        et = self._edge_type_combo.currentText()
        s = self._edge_start.value(); e = self._edge_end.value()
        ip = self._edge_interp.text().strip()
        self._edge_defs.append((et,s,e,ip))
        if self._geo_objects:
            self._geo_objects[self._active_obj_idx]["edges"] = list(self._edge_defs)
        self._update_edge_list()
        self._refresh_draw_geo_preview()
        self._persist_draw_geometry_state()

    def _update_edge_list(self):
        lines = []
        for et,s,e,ip in self._edge_defs:
            lines.append(et + " " + str(s) + " " + str(e) + (" (" + ip + ")" if ip else ""))
        self._edge_list.setPlainText("\n".join(lines))

    def _delete_selected_edge(self):
        cur = self._edge_list.textCursor()
        if cur.hasSelection():
            st = cur.selectionStart(); ed = cur.selectionEnd()
            text = self._edge_list.toPlainText()
            lines = text.split("\n")
            pos = 0; rm = []
            for i,line in enumerate(lines):
                le = pos + len(line)
                if st < le and ed > pos: rm.append(i)
                pos = le + 1
            for i in sorted(rm, reverse=True):
                if i < len(lines): lines.pop(i)
                if i < len(self._edge_defs): self._edge_defs.pop(i)
            if self._geo_objects:
                self._geo_objects[self._active_obj_idx]["edges"] = list(self._edge_defs)
            self._edge_list.setPlainText("\n".join(lines))
            self._refresh_draw_geo_preview()
            self._persist_draw_geometry_state()

    # --- Block helpers ---

    def _on_block_vert_changed(self):
        if self._geo_objects and self._active_obj_idx == 0:
            self._geo_objects[0]["block_v"] = [sb.value() for sb in self._block_vert_inputs]
        self._refresh_draw_geo_preview()
        self._persist_draw_geometry_state()

    def _on_cell_changed(self):
        if self._geo_objects and self._active_obj_idx == 0:
            obj = self._geo_objects[0]
            obj["nx"] = self._geo_nx.value()
            obj["ny"] = self._geo_ny.value()
            obj["nz"] = self._geo_nz.value()
        self._persist_draw_geometry_state()

    def _on_grading_changed(self):
        if self._geo_objects and self._active_obj_idx == 0:
            self._geo_objects[0]["grading"] = self._geo_grading.text().strip() or "1 1 1"
        self._persist_draw_geometry_state()

    def _edge_preview_points(self, verts, etype, start, end, interp_str) -> np.ndarray:
        if start >= len(verts) or end >= len(verts):
            return np.empty((0, 3), dtype=float)
        p0 = np.array(verts[start]); p1 = np.array(verts[end])
        try: coords = [float(v) for v in interp_str.split()]
        except ValueError: coords = []
        if etype == "arc" and len(coords) >= 3:
            interp = np.array(coords[:3])
            mid = (p0+p1)/2.0; offset = interp - mid
            t = np.linspace(0,1,60)
            pts = (1-t)[:,None]*p0 + t[:,None]*p1 + np.sin(t*np.pi)[:,None]*offset
            return pts
        elif etype in ("spline","polyLine","BSpline") and len(coords) >= 3:
            ctrl = [p0]
            for k in range(len(coords)//3):
                ctrl.append(np.array(coords[k*3:(k+1)*3]))
            ctrl.append(p1); ctrl = np.array(ctrl)
            t = np.linspace(0,1,60); n = len(ctrl)-1; pts = np.zeros((len(t),3))
            import math
            for k in range(n+1):
                pts += math.comb(n,k) * (t**k)[:,None] * ((1-t)**(n-k))[:,None] * ctrl[k]
            return pts
        return np.array([p0, p1], dtype=float)

    def _refresh_draw_geo_preview(self):
        if not hasattr(self,"_geo_preview_canvas"): return
        self._render_draw_geometry_preview_vtk(self._geo_preview_canvas)

    def _open_draw_geometry_preview_dialog(self) -> None:
        self._save_current_object()
        dialog = QDialog(self)
        dialog.setWindowTitle("绘制几何 - 放大预览")
        dialog.resize(1100, 820)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        hint = QLabel("放大预览只用于查看当前绘制几何，不会修改 blockMeshDict、STL 或仿真参数。")
        hint.setWordWrap(True)
        canvas = NativeVtkPreviewWidget(dialog, background=(0.12, 0.12, 0.12))
        canvas.setMinimumHeight(680)
        self._render_draw_geometry_preview_vtk(canvas)
        layout.addWidget(hint)
        layout.addWidget(canvas, 1)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.close)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)
        dialog.exec()

    def _render_draw_geometry_preview_vtk(self, canvas: NativeVtkPreviewWidget) -> None:
        canvas.clear((0.12, 0.12, 0.12))
        if not self._geo_objects:
            canvas.add_message("No geometry to preview.")
            canvas.finish()
            return

        all_pts = []
        for oi, obj in enumerate(self._geo_objects):
            verts = obj.get("verts", [])
            if not verts: continue
            corners = np.array(verts, dtype=float)
            all_pts.append(corners)
            is_domain = (oi == 0)
            is_active = (oi == self._active_obj_idx)

            if is_domain:
                for i,v in enumerate(verts):
                    canvas.add_text(str(i), tuple(v), color=(1.0, 0.6, 0.27), size=12)
                for edef in obj.get("edges",[]):
                    canvas.add_polyline(
                        self._edge_preview_points(verts, edef[0],edef[1],edef[2],edef[3]),
                        color=(0.31, 0.76, 1.0),
                        width=2.0,
                        opacity=0.95,
                    )
                bv = obj.get("block_v",[0,1,2,3,4,5,6,7])
                if all(v < len(verts) for v in bv):
                    faces = [[0,3,7,4],[1,5,6,2],[0,1,2,3],[4,5,6,7],[0,1,5,4],[3,2,6,7]]
                    fc = [(0.537,0.820,0.522),(0.957,0.529,0.443)] + [(0.310,0.757,1.000)]*4
                    alpha = [0.25, 0.25] + [0.08] * 4
                    for face,fcol,opacity in zip(faces,fc,alpha):
                        vs = np.array([corners[bv[i]] for i in face], dtype=float)
                        canvas.add_polygon(vs, color=fcol, opacity=opacity, edge_color=fcol)
                    canvas.add_text("inlet", tuple(corners[bv[0]]), color=(0.54, 0.82, 0.52), size=14)
                    canvas.add_text("outlet", tuple(corners[bv[1]]), color=(0.96, 0.53, 0.44), size=14)
            else:
                ec = (1.0,0.6,0.2,0.9) if is_active else (0.5,0.5,0.5,0.6)
                fc = (1.0,0.6,0.2,0.30) if is_active else (0.5,0.5,0.5,0.15)
                if len(verts) >= 8:
                    esc = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
                    for s,e in esc:
                        if s<len(verts) and e<len(verts):
                            canvas.add_polyline(corners[[s,e]], color=ec[:3], width=2.0, opacity=ec[3])
                    faces = [[0,3,7,4],[1,5,6,2],[0,1,2,3],[4,5,6,7],[0,1,5,4],[3,2,6,7]]
                    for face in faces:
                        if all(v<len(verts) for v in face):
                            vs = np.array([corners[v] for v in face], dtype=float)
                            canvas.add_polygon(vs, color=fc[:3], opacity=fc[3], edge_color=ec[:3])

        if all_pts:
            canvas.finish(np.vstack(all_pts))
        else:
            canvas.finish()

    def _apply_draw_geometry(self):
        if self._current_project is None:
            self._show_error("请先新建或打开项目。"); return
        self._save_current_object()
        domain = self._geo_objects[0]
        verts = domain.get("verts", [])
        if len(verts) < 8:
            self._show_error("计算域至少需要 8 个顶点。"); return
        nx = domain.get("nx",10); ny = domain.get("ny",10); nz = domain.get("nz",10)
        grading = domain.get("grading","1 1 1")
        xs = [v[0] for v in verts]; ys = [v[1] for v in verts]; zs = [v[2] for v in verts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)
        mesh_verts = [
            (min_x, min_y, min_z),
            (max_x, min_y, min_z),
            (max_x, max_y, min_z),
            (min_x, max_y, min_z),
            (min_x, min_y, max_z),
            (max_x, min_y, max_z),
            (max_x, max_y, max_z),
            (min_x, max_y, max_z),
        ]
        bv = [0,1,2,3,4,5,6,7]
        vl = "\n".join("    ({} {} {})".format(x,y,z) for x,y,z in mesh_verts)
        el = ""
        bl_line = "    hex ({}) ({} {} {}) simpleGrading ({})\n".format(" ".join(str(v) for v in bv), nx, ny, nz, grading)
        il = self._geo_inlet.text().strip() or "inlet"
        ol = self._geo_outlet.text().strip() or "outlet"
        wl = self._geo_walls.text().strip() or "fixedWalls"
        bm = ("FoamFile\n{{\n    version     2.0;\n    format      ascii;\n"
              "    class       dictionary;\n    object      blockMeshDict;\n}}\n\n"
              "convertToMeters 1;\n\nvertices\n(\n{});\n\n"
              "blocks\n(\n{});\n\nedges\n(\n{});\n\n"
              "boundary\n(\n    {} {{ type patch; faces ((0 4 7 3)); }}\n"
              "    {} {{ type patch; faces ((1 2 6 5)); }}\n"
              "    {} {{ type wall; faces ((0 1 5 4) (0 3 2 1) (4 5 6 7) (3 7 6 2)); }}\n);\n\n"
              "mergePatchPairs\n(\n);\n").format(vl, bl_line, el, il, ol, wl)
        bp = self._current_project.case_dir / "system" / "blockMeshDict"
        bp.parent.mkdir(parents=True, exist_ok=True)
        bp.write_text(bm, encoding="utf-8")
        size = (max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
        import json as _json
        dc_path = self._current_project.case_dir / "system" / "domain_config.json"
        dc_path.write_text(_json.dumps({"key":"custom_domain","name":"手工绘制","size":[round(v,4) for v in size],"cells":[nx,ny,nz],"suggested_location_in_mesh":[round(size[0]*0.1,4),round(size[1]*0.5,4),round(size[2]*0.5,4)],"shape":"box"}, ensure_ascii=False, indent=2), encoding="utf-8")
        self._append_log("blockMeshDict 已生成 (domain): {} 顶点 {}x{}x{}".format(len(verts), nx, ny, nz))

        import struct as _struct
        stl_dir = self._current_project.case_dir / "constant" / "triSurface"
        stl_dir.mkdir(parents=True, exist_ok=True)
        for bi, body in enumerate(self._geo_objects[1:], 1):
            bverts = body.get("verts", [])
            if len(bverts) < 8: continue
            tris = []
            vv = [(float(x),float(y),float(z)) for x,y,z in bverts]
            faces_b = [(0,3,1),(1,3,2),(4,5,7),(5,6,7),(0,1,5),(0,5,4),(3,7,6),(3,6,2),(0,4,7),(0,7,3),(1,2,6),(1,6,5)]
            for f in faces_b:
                if all(idx < len(vv) for idx in f):
                    tris.append((vv[f[0]], vv[f[1]], vv[f[2]]))
            if not tris: continue
            import numpy as _np
            stl_name = "body_{}_{}.stl".format(bi, body["name"])
            stl_path = stl_dir / stl_name
            with open(stl_path, "wb") as sf:
                sf.write(b"\x00"*80)
                sf.write(_struct.pack("<I", len(tris)))
                for v0,v1,v2 in tris:
                    u = _np.array(v1)-_np.array(v0); v = _np.array(v2)-_np.array(v0)
                    n = _np.cross(u,v); n = n/(_np.linalg.norm(n)+1e-12)
                    sf.write(_struct.pack("<3f", *n))
                    sf.write(_struct.pack("<3f", *v0))
                    sf.write(_struct.pack("<3f", *v1))
                    sf.write(_struct.pack("<3f", *v2))
                    sf.write(_struct.pack("<H", 0))
            self._append_log("STL 已导出: " + stl_name + " (" + str(len(tris)) + " 三角形)")
        self._set_status("blockMeshDict + STL 已生成。")
        self._persist_draw_geometry_state()

    def _reset_draw_geometry(self):
        self._geo_objects = self._default_draw_geometry_objects()
        self._active_obj_idx = 0
        self._edge_defs = []
        self._geo_inlet.setText("inlet")
        self._geo_outlet.setText("outlet")
        self._geo_walls.setText("fixedWalls")
        self._rebuild_obj_combo()
        self._load_active_object_to_ui()
        self._refresh_draw_geo_preview()
        self._persist_draw_geometry_state()

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
