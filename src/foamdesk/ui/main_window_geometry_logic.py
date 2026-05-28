from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
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


@dataclass
class MeshImportAsset:
    name: str
    source_path: Path
    polydata: object  # vtkPolyData (cached for redraw)
    visible: bool = True
    translucent: bool = False
    color: tuple[float, float, float] = (0.58, 0.62, 0.66)


@dataclass
class BoundaryFaceGroup:
    name: str
    boundary_type: str
    cell_ids: set[int]


BOUNDARY_COLORS = {
    "inlet": (1.0, 0.18, 0.18),
    "outlet": (0.18, 0.35, 1.0),
    "wall": (0.52, 0.52, 0.52),
    "symmetry": (0.16, 0.78, 0.35),
    "patch": (1.0, 0.85, 0.15),
}

BOUNDARY_TYPE_LABELS = {
    "inlet": "速度入口 inlet",
    "outlet": "压力出口 outlet",
    "wall": "固壁 wall",
    "symmetry": "对称面 symmetry",
    "patch": "计算域外边界 patch",
}


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

    # ── mesh import / Group 1 logic ──────────────────────────

    def _init_mesh_import_state(self) -> None:
        self._mesh_imports: list[MeshImportAsset] = []
        self._mesh_import_selected_index: int = -1
        self._boundary_groups: list[BoundaryFaceGroup] = []
        self._boundary_pending_cells: set[int] = set()
        self._boundary_pick_active: bool = False
        self._domain_bounds_manual: bool = False
        self._last_checkmesh_output: str = ""

    def _import_geometry_file(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "导入 STL 几何",
            "/home/shihuayue/codex_project/assets/test_geometries",
            "STL 文件 (*.stl *.STL);;所有文件 (*)",
        )
        if not paths:
            return
        for fp in paths:
            source_path = Path(fp)
            name = source_path.stem
            reader = vtkSTLReader()
            reader.SetFileName(str(source_path))
            reader.Update()
            polydata = reader.GetOutput()
            if polydata is None or polydata.GetNumberOfPoints() == 0:
                continue
            asset = MeshImportAsset(name=name, source_path=source_path, polydata=polydata)
            self._mesh_imports.append(asset)
        self._rebuild_mesh_import_combo()
        self._auto_fill_domain_bounds()
        self._redraw_mesh_grid_vtk()
        self._save_mesh_workflow_state()

    def _rebuild_mesh_import_combo(self) -> None:
        combo = self._mesh_import_combo
        combo.blockSignals(True)
        combo.clear()
        for i, asset in enumerate(self._mesh_imports):
            combo.addItem(asset.name, i)
        if self._mesh_imports:
            self._mesh_import_selected_index = len(self._mesh_imports) - 1
            combo.setCurrentIndex(self._mesh_import_selected_index)
        else:
            self._mesh_import_selected_index = -1
        combo.blockSignals(False)
        # also sync local refinement part list
        if hasattr(self, "_snappy_local_part_combo"):
            parts = self._snappy_local_part_combo
            parts.blockSignals(True)
            parts.clear()
            for asset in self._mesh_imports:
                parts.addItem(asset.name)
            parts.blockSignals(False)

    def _on_mesh_import_selection_changed(self, _index: int) -> None:
        data = self._mesh_import_combo.currentData()
        if data is None:
            self._mesh_import_selected_index = -1
            return
        self._mesh_import_selected_index = int(data)
        asset = self._mesh_imports[self._mesh_import_selected_index]
        self._mesh_import_visible_check.blockSignals(True)
        self._mesh_import_visible_check.setChecked(asset.visible)
        self._mesh_import_visible_check.blockSignals(False)
        self._mesh_import_opacity_check.blockSignals(True)
        self._mesh_import_opacity_check.setChecked(asset.translucent)
        self._mesh_import_opacity_check.blockSignals(False)
        self._redraw_mesh_grid_vtk()

    def _on_import_visibility_toggled(self, checked: bool) -> None:
        if self._mesh_import_selected_index < 0:
            return
        self._mesh_imports[self._mesh_import_selected_index].visible = checked
        self._redraw_mesh_grid_vtk()

    def _on_import_opacity_toggled(self, checked: bool) -> None:
        if self._mesh_import_selected_index < 0:
            return
        self._mesh_imports[self._mesh_import_selected_index].translucent = checked
        self._redraw_mesh_grid_vtk()

    def _clear_mesh_imports(self) -> None:
        if self._mesh_import_selected_index < 0:
            return
        self._mesh_imports.pop(self._mesh_import_selected_index)
        self._mesh_import_selected_index = -1
        self._rebuild_mesh_import_combo()
        if self._mesh_imports:
            self._mesh_grid_vtk.clear()
            self._redraw_mesh_grid_vtk()
        else:
            self._mesh_grid_vtk.clear()

    def _redraw_mesh_grid_vtk(self) -> None:
        canvas = self._mesh_grid_vtk
        camera = canvas._renderer.GetActiveCamera()
        has_actors = bool(canvas._renderer.GetActors().GetNumberOfItems())
        saved = None
        if has_actors:
            saved = (
                camera.GetPosition(),
                camera.GetFocalPoint(),
                camera.GetViewUp(),
                camera.GetViewAngle(),
            )
        canvas.clear()
        for i, asset in enumerate(self._mesh_imports):
            if not asset.visible:
                continue
            opacity = 0.35 if asset.translucent else 0.92
            if i == self._mesh_import_selected_index:
                # ── selected geometry: show boundary regions ──
                has_boundary_data = bool(
                    self._boundary_groups or self._boundary_pending_cells)
                if not has_boundary_data:
                    # no boundaries yet → draw whole polydata
                    canvas.add_polydata(
                        asset.polydata, color=(0.25, 0.74, 1.0), opacity=opacity,
                        edge_color=(0.96, 0.53, 0.12), line_width=1.0,
                    )
                else:
                    all_boundary_cells: set[int] = set()
                    for group in self._boundary_groups:
                        boundary_pd = self._extract_boundary_polydata(
                            asset.polydata, group.cell_ids)
                        if boundary_pd is not None and boundary_pd.GetNumberOfCells() > 0:
                            color = BOUNDARY_COLORS.get(
                                group.boundary_type, (0.7, 0.7, 0.7))
                            canvas.add_polydata(
                                boundary_pd, color=color, opacity=opacity)
                        all_boundary_cells |= group.cell_ids
                    pending_only = self._boundary_pending_cells - all_boundary_cells
                    if pending_only:
                        pending_pd = self._extract_boundary_polydata(
                            asset.polydata, pending_only)
                        if pending_pd is not None and pending_pd.GetNumberOfCells() > 0:
                            canvas.add_polydata(
                                pending_pd, color=(1.0, 0.6, 0.0), opacity=opacity,
                                edge_color=(1.0, 0.4, 0.0), line_width=1.2,
                            )
                    all_used = all_boundary_cells | self._boundary_pending_cells
                    total_cells = asset.polydata.GetNumberOfCells()
                    if len(all_used) < total_cells:
                        remainder = set(range(total_cells)) - all_used
                        remainder_pd = self._extract_boundary_polydata(
                            asset.polydata, remainder)
                        if remainder_pd is not None and remainder_pd.GetNumberOfCells() > 0:
                            canvas.add_polydata(
                                remainder_pd, color=(0.25, 0.74, 1.0), opacity=opacity,
                                edge_color=(0.96, 0.53, 0.12), line_width=1.0,
                            )
            else:
                canvas.add_polydata(
                    asset.polydata, color=asset.color, opacity=opacity,
                )
        # ── draw domain bounding box ──
        if self._mesh_imports:
            bounds = self._get_domain_bounds()
            if bounds:
                x0, x1 = bounds["x_min"], bounds["x_max"]
                y0, y1 = bounds["y_min"], bounds["y_max"]
                z0, z1 = bounds["z_min"], bounds["z_max"]
                corners = np.array([
                    [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
                ])
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                ]
                for a, b in edges:
                    canvas.add_polyline(
                        np.array([corners[a], corners[b]]),
                        color=(0.2, 0.8, 0.4), width=2.0, opacity=0.7,
                    )

        if saved is not None:
            camera.SetPosition(*saved[0])
            camera.SetFocalPoint(*saved[1])
            camera.SetViewUp(*saved[2])
            camera.SetViewAngle(saved[3])
            canvas.render()
        else:
            canvas.finish()

    # ── boundary face / Group 2 logic ────────────────────────

    def _setup_boundary_picker(self) -> None:
        canvas = self._mesh_grid_vtk
        self._cell_picker = vtk.vtkCellPicker()
        self._cell_picker.SetTolerance(0.01)
        canvas._interactor.SetPicker(self._cell_picker)
        self._pick_observer_id = canvas._interactor.AddObserver(
            "LeftButtonPressEvent", self._on_boundary_pick_event)

    def _on_boundary_pick_event(self, obj, _event) -> None:
        if not self._boundary_pick_active:
            return
        if self._mesh_import_selected_index < 0:
            return
        x, y = obj.GetEventPosition()
        self._cell_picker.Pick(x, y, 0, self._mesh_grid_vtk._renderer)
        cell_id = self._cell_picker.GetCellId()
        if cell_id < 0:
            return
        asset = self._mesh_imports[self._mesh_import_selected_index]
        if cell_id >= asset.polydata.GetNumberOfCells():
            return
        if cell_id in self._boundary_pending_cells:
            self._boundary_pending_cells.discard(cell_id)
        else:
            self._boundary_pending_cells.add(cell_id)
        self._redraw_mesh_grid_vtk()

    def _toggle_boundary_pick(self) -> None:
        self._boundary_pick_active = not self._boundary_pick_active
        btn = self._boundary_pick_btn
        if self._boundary_pick_active:
            btn.setText("拾取中...(再按停止)")
            btn.setStyleSheet("background: #d9534f; color: #fff; font-weight: bold;")
            self._set_status("面拾取模式已激活，点击 3D 视图中的几何面片。")
        else:
            btn.setText("拾取面")
            btn.setStyleSheet("")
            self._set_status("面拾取模式已关闭。")

    def _on_boundary_type_changed(self, _index: int) -> None:
        btype = self._boundary_type_combo.currentData()
        if btype and not self._boundary_name_input.text().strip():
            self._boundary_name_input.setPlaceholderText(f"默认: {btype}")

    def _apply_boundary_to_selected(self) -> None:
        if not self._boundary_pending_cells:
            self._show_error("请先在 3D 视图中拾取面片。")
            return
        btype = self._boundary_type_combo.currentData()
        name = self._boundary_name_input.text().strip()
        if not name:
            name = btype
        idx = self._find_boundary_group(name)
        if idx >= 0:
            self._boundary_groups[idx].cell_ids |= self._boundary_pending_cells
        else:
            self._boundary_groups.append(BoundaryFaceGroup(
                name=name, boundary_type=btype,
                cell_ids=set(self._boundary_pending_cells)))
        self._boundary_pending_cells.clear()
        self._redraw_mesh_grid_vtk()
        self._set_status(f"已将 {len(self._boundary_groups[-1].cell_ids)} 个面片应用到 {name}({btype})。")

    def _find_boundary_group(self, name: str) -> int:
        for i, group in enumerate(self._boundary_groups):
            if group.name == name:
                return i
        return -1

    def _delete_boundary_group(self, row: int) -> None:
        if row < 0 or row >= len(self._boundary_groups):
            return
        self._boundary_groups.pop(row)
        self._redraw_mesh_grid_vtk()

    def _rebuild_boundary_table(self) -> None:
        if not hasattr(self, "_boundary_table"):
            return
        table = self._boundary_table
        table.setRowCount(0)
        for i, group in enumerate(self._boundary_groups):
            table.insertRow(i)
            table.setItem(i, 0, QTableWidgetItem(group.name))
            table.setItem(i, 1, QTableWidgetItem(
                BOUNDARY_TYPE_LABELS.get(group.boundary_type, group.boundary_type)))
            table.setItem(i, 2, QTableWidgetItem(str(len(group.cell_ids))))
            del_btn = QPushButton("删除")
            del_btn.clicked.connect(
                lambda _checked=False, r=i: self._delete_boundary_group(r))
            table.setCellWidget(i, 3, del_btn)

    def _clear_all_boundary_groups(self) -> None:
        self._boundary_groups.clear()
        self._boundary_pending_cells.clear()
        self._redraw_mesh_grid_vtk()
        self._set_status("所有边界定义已清空。")

    def _extract_boundary_polydata(self, source_polydata, cell_ids: set[int]):
        if not cell_ids:
            return None
        try:
            id_array = vtk.vtkIdTypeArray()
            id_array.SetNumberOfComponents(1)
            for cid in sorted(cell_ids):
                id_array.InsertNextValue(cid)
            sel_node = vtk.vtkSelectionNode()
            sel_node.SetFieldType(vtk.vtkSelectionNode.CELL)
            sel_node.SetContentType(vtk.vtkSelectionNode.INDICES)
            sel_node.SetSelectionList(id_array)
            sel = vtk.vtkSelection()
            sel.AddNode(sel_node)
            extract = vtk.vtkExtractSelection()
            extract.SetInputData(0, source_polydata)
            extract.SetInputData(1, sel)
            extract.Update()
            output = extract.GetOutput()
            if output is None:
                return None
            geom_filter = vtk.vtkGeometryFilter()
            geom_filter.SetInputData(output)
            geom_filter.Update()
            result = geom_filter.GetOutput()
            if result is None or result.GetNumberOfCells() == 0:
                return None
            return result
        except Exception:
            return None

    # ── domain mesh / Group 3 logic ─────────────────────────

    def _domain_geom_bbox(self) -> list[float] | None:
        if not self._mesh_imports:
            return None
        bbox = [float("inf"), float("-inf"),
                float("inf"), float("-inf"),
                float("inf"), float("-inf")]
        for asset in self._mesh_imports:
            bounds = asset.polydata.GetBounds()
            if not all(np.isfinite(bounds)):
                continue
            for i in range(3):
                bbox[i * 2] = min(bbox[i * 2], bounds[i * 2])
                bbox[i * 2 + 1] = max(bbox[i * 2 + 1], bounds[i * 2 + 1])
        if not all(np.isfinite(bbox)):
            return None
        return bbox

    def _auto_fill_domain_bounds(self) -> None:
        if self._domain_bounds_manual:
            return
        bbox = self._domain_geom_bbox()
        if bbox is None:
            return
        padding = max((bbox[1] - bbox[0]) * 0.1,
                     (bbox[3] - bbox[2]) * 0.1,
                     (bbox[5] - bbox[4]) * 0.1, 0.1)
        keys = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
        for i, key in enumerate(keys):
            spin = self._domain_bounds_inputs[key]
            spin.blockSignals(True)
            if i % 2 == 0:
                spin.setValue(bbox[i] - padding)
            else:
                spin.setValue(bbox[i] + padding)
            spin.blockSignals(False)
        self._domain_bounds_manual = False
        if hasattr(self, "_mesh_grid_vtk"):
            self._redraw_mesh_grid_vtk()
        if payload.get("mesh_generated") and hasattr(self, "_mesh_grid_vtk"):
            self._on_preview_mesh()
        self._set_status("计算域范围已从几何包围盒自动计算（含 10% 扩展边距）。")

    def _on_domain_manual_override(self) -> None:
        self._domain_bounds_manual = True
        self._redraw_mesh_grid_vtk()

    def _get_domain_bounds(self) -> dict | None:
        if not hasattr(self, "_domain_bounds_inputs"):
            return None
        try:
            return {key: inp.value() for key, inp in self._domain_bounds_inputs.items()}
        except Exception:
            return None

    def _get_domain_mesh_params(self) -> dict:
        return {
            "x_min": self._domain_bounds_inputs["x_min"].value(),
            "x_max": self._domain_bounds_inputs["x_max"].value(),
            "y_min": self._domain_bounds_inputs["y_min"].value(),
            "y_max": self._domain_bounds_inputs["y_max"].value(),
            "z_min": self._domain_bounds_inputs["z_min"].value(),
            "z_max": self._domain_bounds_inputs["z_max"].value(),
            "cells_x": self._domain_cells_inputs["X"].value(),
            "cells_y": self._domain_cells_inputs["Y"].value(),
            "cells_z": self._domain_cells_inputs["Z"].value(),
            "orthogonal": self._domain_orthogonal_check.isChecked(),
            "unit": self._domain_unit_combo.currentData(),
        }

    def _get_snappy_params(self) -> dict:
        return {
            "level": self._snappy_level_combo.currentData(),
            "n_layers": self._snappy_n_layers.value(),
            "first_layer_height": self._snappy_first_layer.value(),
            "expand_ratio": self._snappy_expand_ratio.value(),
            "local_enabled": self._snappy_local_enabled.isChecked(),
            "local_part": self._snappy_local_part_combo.currentText(),
            "local_level": self._snappy_local_level_combo.currentData(),
            "keep_outline": self._snappy_keep_outline.isChecked(),
        }

    def _get_quality_params(self) -> dict:
        return {
            "max_skew": self._quality_max_skew.value(),
            "min_volume": self._quality_min_volume.value(),
            "delete_negative": self._quality_del_negative.isChecked(),
            "smooth": self._quality_smooth.isChecked(),
        }

    # ── Group 6 button bar handlers ─────────────────────────

    def _on_generate_and_execute(self) -> None:
        if not self._mesh_imports:
            self._show_error("请先在组1中导入几何。")
            return
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        domain = self._get_domain_mesh_params()
        if domain["unit"] == "mm":
            convert = 0.001
        else:
            convert = 1.0
        case_dir = self._current_project.case_dir
        system_dir = case_dir / "system"
        system_dir.mkdir(parents=True, exist_ok=True)
        # ── write blockMeshDict ──
        x0, x1 = domain["x_min"], domain["x_max"]
        y0, y1 = domain["y_min"], domain["y_max"]
        z0, z1 = domain["z_min"], domain["z_max"]
        nx, ny, nz = domain["cells_x"], domain["cells_y"], domain["cells_z"]
        bm = (
            "FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }\n"
            f"convertToMeters {convert};\n\n"
            "vertices\n(\n"
            f"    ({x0} {y0} {z0})\n"
            f"    ({x1} {y0} {z0})\n"
            f"    ({x1} {y1} {z0})\n"
            f"    ({x0} {y1} {z0})\n"
            f"    ({x0} {y0} {z1})\n"
            f"    ({x1} {y0} {z1})\n"
            f"    ({x1} {y1} {z1})\n"
            f"    ({x0} {y1} {z1})\n"
            ");\n\n"
            f"blocks\n(\n    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz})"
            " simpleGrading (1 1 1)\n);\n\n"
            "edges\n(\n);\n\n"
            "boundary\n(\n"
            "    inlet  { type patch; faces ((0 4 7 3)); }\n"
            "    outlet { type patch; faces ((1 2 6 5)); }\n"
            "    walls  { type wall;  faces ((0 1 2 3) (4 5 6 7) (0 1 5 4) (3 2 6 7)); }\n"
            ");\n"
        )
        (system_dir / "blockMeshDict").write_text(bm, encoding="utf-8")
        # ── write snappyHexMeshDict ──
        snappy = self._get_snappy_params()
        tri_dir = case_dir / "constant" / "triSurface"
        tri_dir.mkdir(parents=True, exist_ok=True)
        for asset in self._mesh_imports:
            if not asset.source_path.exists():
                continue
            dest = tri_dir / asset.source_path.name
            if not dest.exists():
                import shutil
                shutil.copy2(asset.source_path, dest)
        stl_names = [a.source_path.name for a in self._mesh_imports]
        shm = (
            "FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }\n"
            "castellatedMesh true;\n"
            "snap            true;\n"
            "addLayers       true;\n"
            "mergeTolerance 1e-6;\n\n"
            "geometry\n{\n"
        )
        for name in stl_names:
            shm += f'    {Path(name).stem}\n'
            shm += "    {\n"
            shm += "        type triSurface;\n"
            shm += f'        file "{name}";\n'
            shm += "    }\n"
        shm += "};\n\n"
        shm += (
            "castellatedMeshControls\n{\n"
            "    maxLocalCells 100000;\n"
            "    maxGlobalCells 2000000;\n"
            "    minRefinementCells 0;\n"
            "    maxLoadUnbalance 0.10;\n"
            "    nCellsBetweenLevels 1;\n"
            "    features\n    (\n    );\n"
            "    refinementSurfaces\n    {\n"
        )
        for name in stl_names:
            shm += f'        {Path(name).stem}\n'
            shm += "        {\n"
            shm += f"            level ({snappy['level']} {snappy['level']});\n"
            shm += "        }\n"
        shm += "    }\n"
        shm += (
            "    resolveFeatureAngle 30;\n"
            "    locationInMesh (0.001 0.001 0.001);\n"
            "    allowFreeStandingZoneFaces true;\n"
            "}\n\n"
            "snapControls\n{\n"
            "    nSmoothPatch 3;\n"
            "    tolerance 2.0;\n"
            "    nSolveIter 30;\n"
            "    nRelaxIter 5;\n"
            "    nFeatureSnapIter 10;\n"
            "    implicitFeatureSnap false;\n"
            "    explicitFeatureSnap true;\n"
            "    multiRegionFeatureSnap false;\n"
            "}\n\n"
        )
        if snappy['n_layers'] > 0:
            shm += (
                "addLayersControls\n{\n"
                "    relativeSizes true;\n"
                "    layers\n    {\n"
            )
            for name in stl_names:
                shm += f'        {Path(name).stem}\n'
                shm += "        {\n"
                shm += f"            nSurfaceLayers {snappy['n_layers']};\n"
                shm += "        }\n"
            shm += "    }\n"
            shm += f"    expansionRatio {snappy['expand_ratio']};\n"
            shm += f"    finalLayerThickness {snappy['first_layer_height']};\n"
            shm += "    minThickness 0.001;\n"
            shm += "    nGrow 0;\n"
            shm += "    featureAngle 60;\n"
            shm += "    slipFeatureAngle 30;\n"
            shm += "    nRelaxIter 3;\n"
            shm += "    nSmoothSurfaceNormals 1;\n"
            shm += "    nSmoothNormals 3;\n"
            shm += "    nSmoothThickness 10;\n"
            shm += "    maxFaceThicknessRatio 0.5;\n"
            shm += "    maxThicknessToMedialRatio 0.3;\n"
            shm += "    minMedianAxisAngle 90;\n"
            shm += "    nBufferCellsNoExtrude 0;\n"
            shm += "    nLayerIter 50;\n"
            shm += "}\n\n"
        shm += (
            "meshQualityControls\n{\n"
            f"    maxNonOrtho {int(90 - self._quality_max_skew.value() * 40)};\n"
            "    maxBoundarySkewness 20;\n"
            "    maxInternalSkewness 4;\n"
            "    maxConcave 80;\n"
            "    minFlatness 0.5;\n"
            "    minVol 1e-13;\n"
            "    minTetQuality 1e-30;\n"
            "    minArea -1;\n"
            "    minTwist 0.02;\n"
            "    minDeterminant 0.001;\n"
            "    minFaceWeight 0.02;\n"
            "    minVolRatio 0.01;\n"
            "    minTriangleTwist -1;\n"
            "    nSmoothScale 4;\n"
            "    errorReduction 0.75;\n"
            "}\n\n"
            "writeFlags\n(\n"
            "    scalarLevels\n"
            "    layerSets\n"
            "    layerFields\n"
            ");\n"
        )
        (system_dir / "snappyHexMeshDict").write_text(shm, encoding="utf-8")
        self._append_log("已生成 blockMeshDict 和 snappyHexMeshDict。")
        # execute mesh pipeline via existing infrastructure
        self._run_mesh_pipeline_command()
        self._set_status("网格字典已生成，正在执行 blockMesh → snappyHexMesh → checkMesh...")

    def _run_mesh_pipeline_command(self) -> None:
        env_script = self._context.settings_service.load().openfoam_env_script or ""
        case_dir = self._current_project.case_dir
        if env_script:
            cmd = (
                f'source "{env_script}" && '
                f"cd {shlex.quote(str(case_dir))} && "
                "blockMesh && snappyHexMesh && checkMesh"
            )
        else:
            cmd = (
                f"cd {shlex.quote(str(case_dir))} && "
                "blockMesh && snappyHexMesh && checkMesh"
            )
        self._active_process_kind = "meshPipeline"
        self._foam_process = QProcess(self)
        self._foam_process.setProgram("bash")
        self._foam_process.setArguments(["-lc", cmd])
        self._foam_process.readyReadStandardOutput.connect(self._read_process_stdout)
        self._foam_process.readyReadStandardError.connect(self._read_process_stderr)
        self._foam_process.finished.connect(
            lambda ec, es: self._on_mesh_pipeline_finished(ec, es))
        self._foam_process.start()
        self._append_log(f"执行网格流水线：blockMesh -> snappyHexMesh -> checkMesh")
        self._append_log(f"Case: {case_dir}")

    def _on_mesh_pipeline_finished(self, exit_code: int, _es) -> None:
        try:
            self._last_checkmesh_output = self._current_process_output
        except Exception:
            return
        if exit_code == 0:
            self._save_mesh_workflow_state()
            self._set_status('网格流水线完成。可点击 [预览网格] 查看网格。')
            self._append_log("网格流水线完成。")
            self._on_preview_mesh()
        else:
            self._set_status(f"网格流水线失败，退出码 {exit_code}。请查看日志。")
            self._append_log(f"网格流水线失败，退出码 {exit_code}")

    def _on_preview_mesh(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        mesh_dir = self._current_project.case_dir / "constant" / "polyMesh"
        if not (mesh_dir / "points").exists():
            self._show_error('未找到网格文件。请先点击 [生成字典并执行] 生成网格。')
            return
        try:
            pd = self._read_openfoam_polymesh(mesh_dir)
        except Exception as e:
            self._show_error(f"读取网格失败：{e}")
            return
        canvas = self._mesh_grid_vtk
        canvas.clear()
        mesh_actor = canvas.add_polydata(
            pd, color=(0.75, 0.78, 0.82), opacity=1.0,
            edge_color=(0.03, 0.08, 0.15), line_width=0.5,
        )
        mesh_actor.GetProperty().SetRepresentationToWireframe()
        canvas.finish()
        self._set_status("网格预览已加载。")

    def _read_openfoam_polymesh(self, mesh_dir: Path):
        points_path = mesh_dir / "points"
        faces_path = mesh_dir / "faces"
        vtk_points = vtk.vtkPoints()

        def _skip_header(lines, idx):
            in_comment = False
            while idx < len(lines):
                line = lines[idx].strip()
                if in_comment:
                    if "*/" in line:
                        in_comment = False
                    idx += 1
                    continue
                if not line or line.startswith("//"):
                    idx += 1
                    continue
                if line.startswith("/*"):
                    if "*/" not in line:
                        in_comment = True
                    idx += 1
                    continue
                if line.startswith("FoamFile"):
                    brace = 0
                    while idx < len(lines):
                        l = lines[idx]
                        brace += l.count("{") - l.count("}")
                        idx += 1
                        if brace == 0:
                            break
                    continue
                return idx
            return idx

        with open(points_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        line_idx = _skip_header(lines, 0)
        n_points = int(lines[line_idx].strip())
        line_idx += 1
        if lines[line_idx].strip() == "(":
            line_idx += 1
        for _ in range(n_points):
            parts = lines[line_idx].strip().strip("()").split()
            line_idx += 1
            if not parts:
                continue
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            vtk_points.InsertNextPoint(x, y, z)
        with open(faces_path, "r", encoding="utf-8") as f:
            flines = f.readlines()
        f_idx = _skip_header(flines, 0)
        n_faces = int(flines[f_idx].strip())
        f_idx += 1
        if flines[f_idx].strip() == "(":
            f_idx += 1
        cells = vtk.vtkCellArray()
        for _ in range(n_faces):
            line = flines[f_idx].strip()
            f_idx += 1
            if not line:
                continue
            # handle OpenFOAM formats: "4(0 1 5 4)" or "4 (0 1 5 4)"
            line = line.strip("()")
            parts = line.split()
            if not parts:
                continue
            count = 0
            for part in parts:
                # first token that is a standalone digit is the count
                if part.isdigit():
                    count = int(part)
                    break
                # or the first token has count before '(' like "4(0"
                cleaned = part.lstrip("(")
                if cleaned.isdigit():
                    count = int(cleaned)
                    break
            if count == 0:
                continue
            # collect remaining point IDs (skip the count token)
            values_part = line.split("(", 1)
            if len(values_part) > 1:
                id_str = values_part[1].rstrip(")")
            else:
                id_str = " ".join(parts[1:])
            ids = [int(x) for x in id_str.split() if x]
            if len(ids) != count:
                count = len(ids)  # use actual count
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(count)
            for j in range(count):
                polygon.GetPointIds().SetId(j, ids[j])
            cells.InsertNextCell(polygon)
        pd = vtk.vtkPolyData()
        pd.SetPoints(vtk_points)
        pd.SetPolys(cells)
        return pd

    def _on_check_quality(self) -> None:
        if not hasattr(self, "_last_checkmesh_output") or not self._last_checkmesh_output:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "检查网格质量", '尚未运行网格流水线，或流水线输出为空。\n请先点击 [生成字典并执行]。')
            return
        self._show_error(f"checkMesh 输出：\n\n{self._last_checkmesh_output[-2000:]}")

    def _on_export_boundary_stl(self) -> None:
        if not self._boundary_groups:
            self._show_error("请先在组2中定义边界面并应用到选中面。")
            return
        if self._mesh_import_selected_index < 0:
            self._show_error("请先在组1中选中一个几何。")
            return
        import shutil
        case_dir = self._current_project.case_dir if self._current_project else None
        if case_dir is None:
            self._show_error("请先新建或打开项目。")
            return
        tri_dir = case_dir / "constant" / "triSurface"
        tri_dir.mkdir(parents=True, exist_ok=True)
        asset = self._mesh_imports[self._mesh_import_selected_index]
        for g in self._boundary_groups:
            pd = self._extract_boundary_polydata(asset.polydata, g.cell_ids)
            if pd is None or pd.GetNumberOfCells() == 0:
                continue
            stl_path = tri_dir / f"{g.name}.stl"
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(str(stl_path))
            writer.SetInputData(pd)
            writer.Write()
            self._append_log(f"已导出：{stl_path} ({g.boundary_type})")
        self._set_status(f"已按边界拆分导出 {len(self._boundary_groups)} 个 STL 到 constant/triSurface/。")

    def _on_reset_all_params(self) -> None:
        self._boundary_groups.clear()
        self._boundary_pending_cells.clear()
        self._domain_bounds_manual = False
        self._last_checkmesh_output = ""
        self._rebuild_mesh_import_combo()
        self._rebuild_boundary_table()
        self._redraw_mesh_grid_vtk()
        self._set_status("所有网格参数已重置。")

    # ── state persistence ──────────────────────────────────

    def _mesh_workflow_state_path(self) -> Path | None:
        if self._current_project is None:
            return None
        return self._current_project.case_dir / "system" / "mesh_workflow.json"

    def _save_mesh_workflow_state(self) -> None:
        sp = self._mesh_workflow_state_path()
        if sp is None:
            return
        payload = {
            "imports": [
                {"name": a.name, "source_path": str(a.source_path),
                 "visible": a.visible, "translucent": a.translucent}
                for a in self._mesh_imports
            ],
            "boundary_groups": [
                {"name": g.name, "boundary_type": g.boundary_type,
                 "cell_ids": sorted(g.cell_ids)}
                for g in self._boundary_groups
            ],
            "domain": self._get_domain_mesh_params() if self._mesh_imports else {},
            "snappy": self._get_snappy_params() if hasattr(self, "_snappy_level_combo") else {},
            "quality": self._get_quality_params() if hasattr(self, "_quality_max_skew") else {},
            "mesh_generated": (
                self._current_project is not None and
                (self._current_project.case_dir / "constant" / "polyMesh" / "points").exists()
            ),
        }
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_mesh_workflow_state(self) -> None:
        sp = self._mesh_workflow_state_path()
        if sp is None or not sp.exists():
            return
        try:
            payload = json.loads(sp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        # restore imports
        for item in payload.get("imports", []):
            source_path = Path(item["source_path"])
            if not source_path.exists():
                continue
            reader = vtkSTLReader()
            reader.SetFileName(str(source_path))
            reader.Update()
            polydata = reader.GetOutput()
            if polydata is None or polydata.GetNumberOfPoints() == 0:
                continue
            asset = MeshImportAsset(
                name=item.get("name", source_path.stem),
                source_path=source_path, polydata=polydata,
                visible=item.get("visible", True),
                translucent=item.get("translucent", False),
            )
            self._mesh_imports.append(asset)
        if hasattr(self, "_mesh_import_combo"):
            self._rebuild_mesh_import_combo()
        # restore boundaries
        self._boundary_groups.clear()
        for g in payload.get("boundary_groups", []):
            self._boundary_groups.append(BoundaryFaceGroup(
                name=g["name"], boundary_type=g["boundary_type"],
                cell_ids=set(g.get("cell_ids", [])),
            ))
        if hasattr(self, "_boundary_table"):
            self._rebuild_boundary_table()
        # restore domain
        domain = payload.get("domain", {})
        if domain and hasattr(self, "_domain_bounds_inputs"):
            keys = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
            for key in keys:
                if key in domain:
                    self._domain_bounds_inputs[key].blockSignals(True)
                    self._domain_bounds_inputs[key].setValue(domain[key])
                    self._domain_bounds_inputs[key].blockSignals(False)
            for axis, key in [("X", "cells_x"), ("Y", "cells_y"), ("Z", "cells_z")]:
                if key in domain:
                    self._domain_cells_inputs[axis].setValue(domain[key])
            if "orthogonal" in domain:
                self._domain_orthogonal_check.setChecked(domain["orthogonal"])
            if "unit" in domain:
                idx = self._domain_unit_combo.findData(domain["unit"])
                if idx >= 0:
                    self._domain_unit_combo.setCurrentIndex(idx)
        # restore snappy
        snappy = payload.get("snappy", {})
        if snappy and hasattr(self, "_snappy_level_combo"):
            for key, combo in [("level", self._snappy_level_combo)]:
                if key in snappy:
                    idx = combo.findData(snappy[key])
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
            for key, spin in [("n_layers", self._snappy_n_layers),
                            ("first_layer_height", self._snappy_first_layer),
                            ("expand_ratio", self._snappy_expand_ratio)]:
                if key in snappy:
                    spin.setValue(snappy[key])
            if "local_enabled" in snappy:
                self._snappy_local_enabled.setChecked(snappy["local_enabled"])
            if "local_level" in snappy:
                idx = self._snappy_local_level_combo.findData(snappy["local_level"])
                if idx >= 0:
                    self._snappy_local_level_combo.setCurrentIndex(idx)
            if "keep_outline" in snappy:
                self._snappy_keep_outline.setChecked(snappy["keep_outline"])
        # restore quality
        quality = payload.get("quality", {})
        if quality and hasattr(self, "_quality_max_skew"):
            for key, spin in [("max_skew", self._quality_max_skew),
                            ("min_volume", self._quality_min_volume)]:
                if key in quality:
                    spin.setValue(quality[key])
            if "delete_negative" in quality:
                self._quality_del_negative.setChecked(quality["delete_negative"])
            if "smooth" in quality:
                self._quality_smooth.setChecked(quality["smooth"])
        self._domain_bounds_manual = False
        self._redraw_mesh_grid_vtk()
