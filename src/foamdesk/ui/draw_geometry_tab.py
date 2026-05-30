from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import vtk
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QInputDialog,
    QMenu,
    QTreeWidgetItem,
)


@dataclass
class GeometryObject:
    name: str
    kind: str  # cube, sphere, cylinder, cone, airfoil, bend_pipe
    actor: object  # vtkActor
    source: object  # vtk source
    visible: bool = True
    section: str = "stl"  # "domain" or "stl"
    opacity: float = 1.0
    source_path: str = ""
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    color: tuple[float, float, float] = (0.25, 0.74, 1.0)


def _create_cube_source() -> vtk.vtkCubeSource:
    src = vtk.vtkCubeSource()
    src.SetXLength(1.0)
    src.SetYLength(1.0)
    src.SetZLength(1.0)
    src.SetCenter(0.0, 0.0, 0.0)
    src.Update()
    return src


def _create_sphere_source() -> vtk.vtkSphereSource:
    src = vtk.vtkSphereSource()
    src.SetRadius(0.5)
    src.SetThetaResolution(24)
    src.SetPhiResolution(24)
    src.SetCenter(0.0, 0.0, 0.0)
    src.Update()
    return src


def _create_cylinder_source() -> vtk.vtkCylinderSource:
    src = vtk.vtkCylinderSource()
    src.SetRadius(0.3)
    src.SetHeight(1.0)
    src.SetResolution(24)
    src.SetCenter(0.0, 0.0, 0.0)
    src.Update()
    return src


def _create_cone_source() -> vtk.vtkConeSource:
    src = vtk.vtkConeSource()
    src.SetRadius(0.3)
    src.SetHeight(1.0)
    src.SetResolution(24)
    src.SetCenter(0.0, 0.0, 0.0)
    src.Update()
    return src


def _polydata_from_points_faces(points: list[tuple[float, float, float]], faces: list[list[int]]) -> vtk.vtkPolyData:
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
    cells = vtk.vtkCellArray()
    for face in faces:
        if len(face) < 3:
            continue
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(face))
        for index, point_id in enumerate(face):
            polygon.GetPointIds().SetId(index, int(point_id))
        cells.InsertNextCell(polygon)
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetPolys(cells)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly_data)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


def _create_airfoil_source() -> vtk.vtkPolyData:
    chord = 1.4
    span = 2.4
    thickness = 0.12
    x_values = np.linspace(0.0, 1.0, 42)
    yt = 5.0 * thickness * (
        0.2969 * np.sqrt(np.maximum(x_values, 1e-6))
        - 0.1260 * x_values
        - 0.3516 * x_values**2
        + 0.2843 * x_values**3
        - 0.1015 * x_values**4
    )
    upper = np.column_stack([(x_values - 0.5) * chord, yt * chord])
    lower = np.column_stack([(x_values[::-1] - 0.5) * chord, -yt[::-1] * chord])
    profile = np.vstack([upper, lower])
    points: list[tuple[float, float, float]] = []
    for y_value in (-span / 2.0, span / 2.0):
        for x_value, z_value in profile:
            points.append((float(x_value), float(y_value), float(z_value)))
    count = len(profile)
    faces: list[list[int]] = []
    faces.append(list(range(count - 1, -1, -1)))
    faces.append(list(range(count, count * 2)))
    for index in range(count):
        next_index = (index + 1) % count
        faces.append([index, next_index, count + next_index, count + index])
    return _polydata_from_points_faces(points, faces)


def _create_bend_pipe_source() -> vtk.vtkPolyData:
    major_radius = 0.65
    tube_radius = 0.18
    bend_angle = np.pi / 2.0
    straight_length = 0.55
    bend_segments = 40
    ring_segments = 28
    centerline: list[np.ndarray] = []
    pre_count = 10
    for index in range(pre_count):
        x_value = -straight_length + straight_length * index / max(pre_count - 1, 1)
        centerline.append(np.array([x_value, 0.0, 0.0], dtype=float))
    for angle in np.linspace(0.0, bend_angle, bend_segments):
        centerline.append(np.array([major_radius * np.sin(angle), major_radius * (1.0 - np.cos(angle)), 0.0], dtype=float))
    post_start = centerline[-1].copy()
    post_count = 10
    for index in range(1, post_count + 1):
        y_value = post_start[1] + straight_length * index / post_count
        centerline.append(np.array([post_start[0], y_value, 0.0], dtype=float))

    points: list[tuple[float, float, float]] = []
    for center_index, center in enumerate(centerline):
        if center_index == 0:
            tangent = centerline[1] - center
        elif center_index == len(centerline) - 1:
            tangent = center - centerline[center_index - 1]
        else:
            tangent = centerline[center_index + 1] - centerline[center_index - 1]
        tangent_norm = max(float(np.linalg.norm(tangent)), 1e-9)
        tangent = tangent / tangent_norm
        normal = np.array([-tangent[1], tangent[0], 0.0], dtype=float)
        if float(np.linalg.norm(normal)) <= 1e-9:
            normal = np.array([1.0, 0.0, 0.0], dtype=float)
        normal = normal / max(float(np.linalg.norm(normal)), 1e-9)
        binormal = np.array([0.0, 0.0, 1.0], dtype=float)
        for theta in np.linspace(0.0, 2.0 * np.pi, ring_segments, endpoint=False):
            point = center + tube_radius * np.cos(theta) * normal + tube_radius * np.sin(theta) * binormal
            points.append((float(point[0]), float(point[1]), float(point[2])))

    faces: list[list[int]] = []
    ring_count = len(centerline)
    for ring_index in range(ring_count - 1):
        base = ring_index * ring_segments
        next_base = (ring_index + 1) * ring_segments
        for segment_index in range(ring_segments):
            next_segment = (segment_index + 1) % ring_segments
            faces.append([base + segment_index, base + next_segment, next_base + next_segment, next_base + segment_index])
    faces.append(list(range(ring_segments - 1, -1, -1)))
    end_base = (ring_count - 1) * ring_segments
    faces.append([end_base + index for index in range(ring_segments)])
    return _polydata_from_points_faces(points, faces)


_PRIMITIVE_FACTORIES = {
    "cube": (_create_cube_source, "立方体"),
    "sphere": (_create_sphere_source, "球体"),
    "cylinder": (_create_cylinder_source, "圆柱"),
    "cone": (_create_cone_source, "圆锥"),
    "airfoil": (_create_airfoil_source, "机翼"),
    "bend_pipe": (_create_bend_pipe_source, "弯管"),
}


class DrawGeometryLogicMixin:
    """Mixin providing all modeling logic methods for MainWindow."""

    def _init_modeling_state(self) -> None:
        if hasattr(self, "_modeling_viewport") and self._modeling_viewport is not None:
            for obj in self._modeling_objects:
                self._modeling_viewport._renderer.RemoveActor(obj.actor)
        self._modeling_objects: list[GeometryObject] = []
        self._modeling_selected_index: int = -1
        self._modeling_counter: dict[str, int] = {}
        self._modeling_active_section: str = "stl"
        self._load_modeling_state()
        if hasattr(self, "_modeling_viewport") and self._modeling_viewport is not None:
            for obj in self._modeling_objects:
                self._modeling_viewport._renderer.AddActor(obj.actor)
                self._apply_transform(obj)
            if hasattr(self, "_modeling_tree"):
                self._rebuild_tree()
            self._modeling_viewport.render()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _modeling_state_path(self) -> str:
        if hasattr(self, "_current_project") and self._current_project is not None:
            return str(self._current_project.case_dir / "modeling_state.json")
        from pathlib import Path
        return str(Path(__file__).parent.parent.parent.parent / "config" / "modeling_state.json")

    def _save_modeling_state(self) -> None:
        import json
        data = []
        for obj_index, obj in enumerate(self._modeling_objects):
            kind = obj.kind
            source_path = obj.source_path
            if isinstance(obj.source, vtk.vtkPolyData) or (kind == "stl" and not source_path):
                stored_path = self._write_modeling_asset(obj, obj_index)
                if stored_path:
                    kind = "stl"
                    source_path = str(stored_path)
                    obj.kind = kind
                    obj.source_path = source_path
            data.append({
                "name": obj.name, "kind": kind, "section": obj.section,
                "position": obj.position, "rotation": obj.rotation,
                "scale": obj.scale, "color": list(obj.color),
                "opacity": obj.opacity, "visible": obj.visible,
                "source_path": source_path,
            })
        with open(self._modeling_state_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _modeling_assets_dir(self):
        if hasattr(self, "_current_project") and self._current_project is not None:
            return self._current_project.case_dir / ".foamdesk_modeling" / "assets"
        from pathlib import Path
        return Path(__file__).parent.parent.parent.parent / "config" / "modeling_assets"

    def _write_modeling_asset(self, obj: GeometryObject, obj_index: int = 0):
        directory = self._modeling_assets_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{obj_index:03d}_{self._safe_modeling_filename(obj.name)}.stl"
        poly_data = self._source_polydata(obj)
        if poly_data is None or poly_data.GetNumberOfPoints() == 0:
            return None
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(str(path))
        writer.SetInputData(poly_data)
        writer.Write()
        return path

    def _safe_modeling_filename(self, name: str) -> str:
        safe = "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in name)
        return safe or "geometry"

    def _save_modeling_state_if_ready(self) -> None:
        if getattr(self, "_suspend_draw_geometry_persist", False):
            return
        try:
            self._save_modeling_state()
        except (OSError, AttributeError):
            pass

    def _load_modeling_state(self) -> None:
        import json
        from pathlib import Path
        path = self._modeling_state_path()
        if not Path(path).exists():
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for d in data:
            kind = d.get("kind", "cube")
            source_path = d.get("source_path", "")
            if kind == "stl" and source_path:
                from pathlib import Path as _Path
                if not _Path(source_path).exists():
                    continue
                reader = vtk.vtkSTLReader()
                reader.SetFileName(source_path)
                reader.Update()
                src = reader.GetOutput()
                if src.GetNumberOfPoints() == 0:
                    continue
            elif kind in _PRIMITIVE_FACTORIES:
                factory, _ = _PRIMITIVE_FACTORIES[kind]
                src = factory()
            else:
                continue
            mapper = vtk.vtkPolyDataMapper()
            if isinstance(src, vtk.vtkPolyData):
                mapper.SetInputData(src)
            else:
                mapper.SetInputConnection(src.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*d.get("color", [0.25, 0.74, 1.0]))
            actor.GetProperty().SetOpacity(d.get("opacity", 1.0))
            actor.GetProperty().SetInterpolationToPhong()
            actor.SetVisibility(d.get("visible", True))
            obj = GeometryObject(
                name=d["name"], kind=kind, actor=actor, source=src,
                section=d.get("section", "stl"),
                position=d.get("position", [0, 0, 0]),
                rotation=d.get("rotation", [0, 0, 0]),
                scale=d.get("scale", [1, 1, 1]),
                color=tuple(d.get("color", [0.25, 0.74, 1.0])),
                opacity=d.get("opacity", 1.0),
                visible=d.get("visible", True),
                source_path=source_path,
            )
            self._modeling_objects.append(obj)

    # ------------------------------------------------------------------
    # primitive management
    # ------------------------------------------------------------------

    def _add_primitive(self, kind: str) -> None:
        factory, label = _PRIMITIVE_FACTORIES[kind]
        self._modeling_counter[kind] = self._modeling_counter.get(kind, 0) + 1
        name = f"{label}{self._modeling_counter[kind]}"

        source = factory()
        mapper = vtk.vtkPolyDataMapper()
        if isinstance(source, vtk.vtkPolyData):
            mapper.SetInputData(source)
        else:
            mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.25, 0.74, 1.0)
        actor.GetProperty().SetInterpolationToPhong()

        obj = GeometryObject(name=name, kind=kind, actor=actor, source=source, section=self._modeling_active_section)
        self._modeling_objects.append(obj)
        self._modeling_viewport._renderer.AddActor(actor)

        self._rebuild_tree()
        self._select_object(len(self._modeling_objects) - 1)
        self._modeling_viewport.render()
        self._save_modeling_state_if_ready()
        self._set_modeling_status(f"已创建 {name}")

    def _delete_selected(self) -> None:
        if self._modeling_selected_index < 0 or self._modeling_selected_index >= len(self._modeling_objects):
            return
        obj = self._modeling_objects.pop(self._modeling_selected_index)
        self._modeling_viewport._renderer.RemoveActor(obj.actor)
        self._modeling_selected_index = -1
        self._rebuild_tree()
        self._load_properties()
        self._set_property_enabled(False)
        self._modeling_viewport.render()
        self._save_modeling_state_if_ready()
        self._set_modeling_status(f"已删除 {obj.name}")

    # ------------------------------------------------------------------
    # tree
    # ------------------------------------------------------------------

    def _rebuild_tree(self) -> None:
        tree = self._modeling_tree
        tree.blockSignals(True)
        tree.clear()
        active_marker = " ←" if self._modeling_active_section == "domain" else ""
        domain_header = QTreeWidgetItem([f"计算域{active_marker}", ""])
        domain_header.setData(0, Qt.ItemDataRole.UserRole, -10)
        domain_header.setFlags(domain_header.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
        stl_marker = " ←" if self._modeling_active_section == "stl" else ""
        stl_header = QTreeWidgetItem([f"STL{stl_marker}", ""])
        stl_header.setData(0, Qt.ItemDataRole.UserRole, -20)
        stl_header.setFlags(stl_header.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
        tree.addTopLevelItem(domain_header)
        tree.addTopLevelItem(stl_header)
        for i, obj in enumerate(self._modeling_objects):
            item = QTreeWidgetItem([obj.name, obj.kind])
            item.setData(0, Qt.ItemDataRole.UserRole, i)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked if obj.visible else Qt.CheckState.Unchecked)
            if i == self._modeling_selected_index:
                tree.setCurrentItem(item)
            if obj.section == "domain":
                domain_header.addChild(item)
            else:
                stl_header.addChild(item)
        tree.expandAll()
        tree.blockSignals(False)

    def _on_tree_item_clicked(self, item: QTreeWidgetItem) -> None:
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        if idx == -10:
            self._modeling_active_section = "domain"
            self._rebuild_tree()
            self._set_modeling_status("当前区间：计算域")
            return
        if idx == -20:
            self._modeling_active_section = "stl"
            self._rebuild_tree()
            self._set_modeling_status("当前区间：STL")
            return
        if idx >= 0:
            self._select_object(int(idx))
            obj = self._modeling_objects[int(idx)]
            self._modeling_active_section = obj.section
            self._rebuild_tree()

    def _on_tree_item_changed(self, item: QTreeWidgetItem) -> None:
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        if idx is None or idx < 0:
            return
        i = int(idx)
        if i < 0 or i >= len(self._modeling_objects):
            return
        visible = item.checkState(0) == Qt.CheckState.Checked
        self._modeling_objects[i].visible = visible
        self._modeling_objects[i].actor.SetVisibility(visible)
        self._modeling_viewport.render()
        self._save_modeling_state_if_ready()

    def _tree_context_menu(self, pos) -> None:
        item = self._modeling_tree.itemAt(pos)
        if item is None:
            return
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        menu = QMenu(self)
        rename_act = menu.addAction("重命名")
        delete_act = menu.addAction("删除")
        action = menu.exec(self._modeling_tree.viewport().mapToGlobal(pos))
        if action == rename_act:
            self._modeling_tree.editItem(item, 0)
        elif action == delete_act:
            self._select_object(int(idx))
            self._delete_selected()

    # ------------------------------------------------------------------
    # selection
    # ------------------------------------------------------------------

    def _select_object(self, index: int) -> None:
        if index < 0 or index >= len(self._modeling_objects):
            return
        self._modeling_selected_index = index
        self._load_properties()
        self._set_property_enabled(True)
        self._set_modeling_status(f"选中 {self._modeling_objects[index].name}")

    def _activate_select_mode(self) -> None:
        self._set_modeling_status("选择模式：请在模型树中点击选择模型")

    def _import_stl_file(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        from pathlib import Path
        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入 STL 文件", "", "STL 文件 (*.stl *.STL)"
        )
        if not file_path:
            return
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_path)
        reader.Update()
        poly_data = reader.GetOutput()
        if poly_data.GetNumberOfPoints() == 0:
            self._set_modeling_status("STL 文件为空或读取失败")
            return
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.25, 0.74, 1.0)
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetProperty().SetEdgeVisibility(True)
        actor.GetProperty().SetEdgeColor(0.1, 0.1, 0.1)
        actor.GetProperty().SetLineWidth(0.5)
        name = Path(file_path).stem
        obj = GeometryObject(
            name=name, kind="stl", actor=actor, source=poly_data,
            section=self._modeling_active_section, source_path=file_path,
        )
        self._modeling_objects.append(obj)
        self._modeling_viewport._renderer.AddActor(actor)
        self._rebuild_tree()
        self._select_object(len(self._modeling_objects) - 1)
        self._modeling_viewport.render()
        self._save_modeling_state_if_ready()
        self._set_modeling_status(f"已导入 {name}")

    def _export_stl_file(self) -> None:
        if self._modeling_selected_index < 0:
            self._set_modeling_status("请先选择一个几何体")
            return
        from PySide6.QtWidgets import QFileDialog
        obj = self._modeling_objects[self._modeling_selected_index]
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出 STL 文件", f"{obj.name}.stl", "STL 文件 (*.stl)"
        )
        if not file_path:
            return
        pd = obj.source if isinstance(obj.source, vtk.vtkPolyData) else obj.source.GetOutput()
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(pd)
        t = vtk.vtkTransform()
        t.Translate(*obj.position)
        t.RotateX(obj.rotation[0]); t.RotateY(obj.rotation[1]); t.RotateZ(obj.rotation[2])
        t.Scale(*obj.scale)
        tf.SetTransform(t)
        tf.Update()
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(file_path)
        writer.SetInputData(tf.GetOutput())
        writer.Write()
        self._set_modeling_status(f"已导出 {obj.name} → {file_path}")

    def _finish_draw_geometry(self) -> None:
        if self._current_project is None:
            self._set_modeling_status("请先新建或打开项目")
            return
        checked_objects = [
            obj
            for obj in self._modeling_objects
            if obj.visible and obj.section == "stl"
        ]
        if not checked_objects:
            self._set_modeling_status("请在 STL 模型树中勾选要导入网格页的几何体")
            return

        from pathlib import Path
        from foamdesk.ui.main_window_geometry_logic import MeshImportAsset

        cache_dir = self._draw_geometry_cache_dir()
        self._clear_draw_geometry_cache()
        cache_dir.mkdir(parents=True, exist_ok=True)
        exported_assets: list[MeshImportAsset] = []
        for obj in checked_objects:
            output_path = self._unique_draw_export_path(cache_dir, f"{obj.name}.stl")
            poly_data = self._transformed_object_polydata(obj)
            if poly_data is None or poly_data.GetNumberOfPoints() == 0:
                continue
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(str(output_path))
            writer.SetInputData(poly_data)
            writer.Write()
            exported_assets.append(
                MeshImportAsset(
                    name=Path(output_path).stem,
                    source_path=output_path,
                    polydata=poly_data,
                )
            )
        if not exported_assets:
            self._set_modeling_status("勾选几何导出失败，请检查模型是否为空")
            return

        self._workspace_tabs.setCurrentIndex(self.TAB_MESH_GENERATION)
        self._mesh_imports = exported_assets
        self._mesh_import_selected_index = len(exported_assets) - 1
        self._domain_bounds_manual = False
        self._rebuild_mesh_import_combo()
        self._auto_fill_domain_bounds()
        self._redraw_mesh_grid_vtk()
        self._save_mesh_workflow_state()
        self._refresh_geometry_panel()
        self._set_modeling_status(f"已完成绘制：导入 {len(exported_assets)} 个几何到网格生成页")
        self._set_status("绘制几何已导入网格生成页。")

    def _draw_geometry_cache_dir(self):
        return self._current_project.case_dir / ".foamdesk_cache" / "draw_geometry"

    def _clear_draw_geometry_cache(self) -> None:
        if self._current_project is None:
            return
        cache_dir = self._draw_geometry_cache_dir()
        if not cache_dir.exists():
            return
        import shutil
        try:
            shutil.rmtree(cache_dir)
        except OSError:
            pass

    def _transformed_object_polydata(self, obj: GeometryObject):
        source_poly_data = obj.source if isinstance(obj.source, vtk.vtkPolyData) else obj.source.GetOutput()
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputData(source_poly_data)
        transform = vtk.vtkTransform()
        transform.Translate(*obj.position)
        transform.RotateX(obj.rotation[0])
        transform.RotateY(obj.rotation[1])
        transform.RotateZ(obj.rotation[2])
        transform.Scale(*obj.scale)
        transform_filter.SetTransform(transform)
        transform_filter.Update()
        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(transform_filter.GetOutput())
        clean.Update()
        return clean.GetOutput()

    def _unique_draw_export_path(self, directory, filename: str):
        from pathlib import Path
        safe_name = "".join(character if character.isalnum() or character in ("-", "_", ".") else "_" for character in filename)
        path = Path(directory) / (safe_name or "geometry.stl")
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix or ".stl"
        index = 1
        while True:
            candidate = Path(directory) / f"{stem}_{index}{suffix}"
            if not candidate.exists():
                return candidate
            index += 1

    # ------------------------------------------------------------------
    # direct geometry editing / boolean operations
    # ------------------------------------------------------------------

    def _selected_modeling_object(self) -> GeometryObject | None:
        if self._modeling_selected_index < 0 or self._modeling_selected_index >= len(self._modeling_objects):
            self._set_modeling_status("请先选择一个几何体")
            return None
        return self._modeling_objects[self._modeling_selected_index]

    def _source_polydata(self, obj: GeometryObject):
        if isinstance(obj.source, vtk.vtkPolyData):
            return obj.source
        obj.source.Update()
        return obj.source.GetOutput()

    def _replace_object_polydata(self, obj: GeometryObject, poly_data) -> None:
        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(poly_data)
        clean.Update()
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(clean.GetOutput())
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.SplittingOff()
        normals.Update()
        output = vtk.vtkPolyData()
        output.DeepCopy(normals.GetOutput())
        obj.source = output
        obj.kind = "stl"
        obj.source_path = ""
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(output)
        obj.actor.SetMapper(mapper)
        self._apply_transform(obj)
        self._save_modeling_state_if_ready()

    def _ask_edit_delta(self) -> tuple[float, float, float] | None:
        text, ok = QInputDialog.getText(self, "输入移动量", "输入 dx dy dz，例如：0.2 0 0")
        if not ok:
            return None
        parts = text.replace(",", " ").split()
        if len(parts) != 3:
            self._set_modeling_status("移动量格式应为 dx dy dz")
            return None
        try:
            return float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            self._set_modeling_status("移动量必须是数字")
            return None

    def _edit_selected_vertices(self, region_type: str) -> None:
        obj = self._selected_modeling_object()
        if obj is None:
            return
        source_pd = self._source_polydata(obj)
        if source_pd is None or source_pd.GetNumberOfPoints() == 0:
            self._set_modeling_status("当前几何体没有可编辑点")
            return
        points = source_pd.GetPoints()
        bounds = source_pd.GetBounds()
        ranges = [max(bounds[i * 2 + 1] - bounds[i * 2], 1e-9) for i in range(3)]
        tolerance = max(ranges) * 0.04

        if region_type == "point":
            options = [
                f"{sx}X {sy}Y {sz}Z"
                for sx in ("-", "+") for sy in ("-", "+") for sz in ("-", "+")
            ]
            title = "按点编辑"
        elif region_type == "edge":
            options = [
                "-Y -Z 边", "-Y +Z 边", "+Y -Z 边", "+Y +Z 边",
                "-X -Z 边", "-X +Z 边", "+X -Z 边", "+X +Z 边",
                "-X -Y 边", "-X +Y 边", "+X -Y 边", "+X +Y 边",
            ]
            title = "按线编辑"
        else:
            options = ["-X 面", "+X 面", "-Y 面", "+Y 面", "-Z 面", "+Z 面"]
            title = "按面拉伸"
        option, ok = QInputDialog.getItem(self, title, "选择要移动的区域", options, 0, False)
        if not ok:
            return
        delta = self._ask_edit_delta()
        if delta is None:
            return

        def near_axis(point, axis: int, side: str) -> bool:
            target = bounds[axis * 2] if side == "-" else bounds[axis * 2 + 1]
            return abs(point[axis] - target) <= tolerance

        selected_ids: list[int] = []
        for point_id in range(points.GetNumberOfPoints()):
            point = points.GetPoint(point_id)
            if region_type == "point":
                sx, sy, sz = option.split()
                selected = near_axis(point, 0, sx[0]) and near_axis(point, 1, sy[0]) and near_axis(point, 2, sz[0])
            elif region_type == "edge":
                selected = True
                for token in option.split()[:2]:
                    axis = "XYZ".index(token[1])
                    selected = selected and near_axis(point, axis, token[0])
            else:
                token = option.split()[0]
                selected = near_axis(point, "XYZ".index(token[1]), token[0])
            if selected:
                selected_ids.append(point_id)
        if not selected_ids:
            self._set_modeling_status("没有找到可编辑的点，换一个区域试试")
            return

        edited = vtk.vtkPolyData()
        edited.DeepCopy(source_pd)
        edited_points = vtk.vtkPoints()
        edited_points.DeepCopy(points)
        for point_id in selected_ids:
            x, y, z = edited_points.GetPoint(point_id)
            edited_points.SetPoint(point_id, x + delta[0], y + delta[1], z + delta[2])
        edited.SetPoints(edited_points)
        self._replace_object_polydata(obj, edited)
        self._modeling_viewport.render()
        self._set_modeling_status(f"{title}完成：移动 {len(selected_ids)} 个点")

    def _triangulated_polydata(self, poly_data):
        triangle = vtk.vtkTriangleFilter()
        triangle.SetInputData(poly_data)
        triangle.Update()
        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(triangle.GetOutput())
        clean.Update()
        return clean.GetOutput()

    def _boolean_selected(self, operation: str) -> None:
        target = self._selected_modeling_object()
        if target is None:
            return
        candidates = [
            obj.name
            for index, obj in enumerate(self._modeling_objects)
            if index != self._modeling_selected_index and obj.visible
        ]
        if not candidates:
            self._set_modeling_status("请至少再勾选/显示一个几何体作为布尔运算对象")
            return
        tool_name, ok = QInputDialog.getItem(self, "布尔运算", "选择参与运算的第二个几何体", candidates, 0, False)
        if not ok:
            return
        tool = next(obj for obj in self._modeling_objects if obj.name == tool_name)
        operation_map = {
            "union": vtk.vtkBooleanOperationPolyDataFilter.VTK_UNION,
            "difference": vtk.vtkBooleanOperationPolyDataFilter.VTK_DIFFERENCE,
            "intersection": vtk.vtkBooleanOperationPolyDataFilter.VTK_INTERSECTION,
        }
        boolean_filter = vtk.vtkBooleanOperationPolyDataFilter()
        boolean_filter.SetOperation(operation_map[operation])
        boolean_filter.SetInputData(0, self._triangulated_polydata(self._transformed_object_polydata(target)))
        boolean_filter.SetInputData(1, self._triangulated_polydata(self._transformed_object_polydata(tool)))
        boolean_filter.Update()
        output = boolean_filter.GetOutput()
        if output is None or output.GetNumberOfPoints() == 0:
            self._set_modeling_status("布尔运算没有生成有效结果，请确认两个模型封闭且存在相交关系")
            return
        result_name = self._unique_modeling_name(f"{target.name}_{operation}_{tool.name}")
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(output)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*target.color)
        actor.GetProperty().SetOpacity(target.opacity)
        actor.GetProperty().SetInterpolationToPhong()
        result = GeometryObject(
            name=result_name,
            kind="stl",
            actor=actor,
            source=output,
            section=target.section,
            color=target.color,
            opacity=target.opacity,
        )
        self._modeling_objects.append(result)
        self._modeling_viewport._renderer.AddActor(actor)
        self._rebuild_tree()
        self._select_object(len(self._modeling_objects) - 1)
        self._modeling_viewport.render()
        self._save_modeling_state_if_ready()
        self._set_modeling_status(f"已生成布尔结果：{result_name}")

    def _unique_modeling_name(self, base_name: str) -> str:
        existing = {obj.name for obj in self._modeling_objects}
        if base_name not in existing:
            return base_name
        index = 1
        while f"{base_name}_{index}" in existing:
            index += 1
        return f"{base_name}_{index}"

    # ------------------------------------------------------------------
    # property panel
    # ------------------------------------------------------------------

    def _load_properties(self) -> None:
        if self._modeling_selected_index < 0:
            return
        obj = self._modeling_objects[self._modeling_selected_index]
        self._block_prop_signals(True)
        self._modeling_prop_name.setText(obj.name)
        self._modeling_prop_kind.setText(obj.kind)
        self._modeling_prop_pos_x.setValue(obj.position[0])
        self._modeling_prop_pos_y.setValue(obj.position[1])
        self._modeling_prop_pos_z.setValue(obj.position[2])
        self._modeling_prop_rot_x.setValue(obj.rotation[0])
        self._modeling_prop_rot_y.setValue(obj.rotation[1])
        self._modeling_prop_rot_z.setValue(obj.rotation[2])
        self._modeling_prop_scl_x.setValue(obj.scale[0])
        self._modeling_prop_scl_y.setValue(obj.scale[1])
        self._modeling_prop_scl_z.setValue(obj.scale[2])
        r, g, b = obj.color
        self._modeling_color_btn.setStyleSheet(
            f"background-color: rgb({int(r*255)},{int(g*255)},{int(b*255)}); border: 1px solid #555;"
        )
        self._modeling_prop_opacity.setValue(obj.opacity)
        self._block_prop_signals(False)

    def _on_prop_name_changed(self) -> None:
        if self._modeling_selected_index < 0:
            return
        new_name = self._modeling_prop_name.text().strip()
        if new_name:
            self._modeling_objects[self._modeling_selected_index].name = new_name
            self._rebuild_tree()
            self._save_modeling_state_if_ready()

    def _on_prop_transform_changed(self) -> None:
        if self._modeling_selected_index < 0:
            return
        obj = self._modeling_objects[self._modeling_selected_index]
        obj.position = [
            self._modeling_prop_pos_x.value(),
            self._modeling_prop_pos_y.value(),
            self._modeling_prop_pos_z.value(),
        ]
        obj.rotation = [
            self._modeling_prop_rot_x.value(),
            self._modeling_prop_rot_y.value(),
            self._modeling_prop_rot_z.value(),
        ]
        obj.scale = [
            self._modeling_prop_scl_x.value(),
            self._modeling_prop_scl_y.value(),
            self._modeling_prop_scl_z.value(),
        ]
        self._apply_transform(obj)
        self._save_modeling_state_if_ready()

    def _apply_transform(self, obj: GeometryObject) -> None:
        transform = vtk.vtkTransform()
        transform.Translate(obj.position[0], obj.position[1], obj.position[2])
        transform.RotateX(obj.rotation[0])
        transform.RotateY(obj.rotation[1])
        transform.RotateZ(obj.rotation[2])
        transform.Scale(obj.scale[0], obj.scale[1], obj.scale[2])
        obj.actor.SetUserTransform(transform)
        self._modeling_viewport.render()

    def _on_color_pick(self) -> None:
        if self._modeling_selected_index < 0:
            return
        obj = self._modeling_objects[self._modeling_selected_index]
        r, g, b = obj.color
        color = QColorDialog.getColor(
            QColor(int(r * 255), int(g * 255), int(b * 255)), self, "选择颜色"
        )
        if not color.isValid():
            return
        r, g, b = color.redF(), color.greenF(), color.blueF()
        obj.color = (r, g, b)
        obj.actor.GetProperty().SetColor(r, g, b)
        self._modeling_color_btn.setStyleSheet(
            f"background-color: rgb({int(r*255)},{int(g*255)},{int(b*255)}); border: 1px solid #555;"
        )
        self._modeling_viewport.render()
        self._save_modeling_state_if_ready()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _set_property_enabled(self, enabled: bool) -> None:
        for w in (
            self._modeling_prop_name,
            self._modeling_prop_pos_x, self._modeling_prop_pos_y, self._modeling_prop_pos_z,
            self._modeling_prop_rot_x, self._modeling_prop_rot_y, self._modeling_prop_rot_z,
            self._modeling_prop_scl_x, self._modeling_prop_scl_y, self._modeling_prop_scl_z,
            self._modeling_color_btn,
            self._modeling_prop_opacity,
        ):
            w.setEnabled(enabled)

    def _on_prop_opacity_changed(self) -> None:
        if self._modeling_selected_index < 0:
            return
        obj = self._modeling_objects[self._modeling_selected_index]
        obj.opacity = self._modeling_prop_opacity.value()
        obj.actor.GetProperty().SetOpacity(obj.opacity)
        self._modeling_viewport.render()
        self._save_modeling_state_if_ready()

    def _block_prop_signals(self, block: bool) -> None:
        for w in (
            self._modeling_prop_pos_x, self._modeling_prop_pos_y, self._modeling_prop_pos_z,
            self._modeling_prop_rot_x, self._modeling_prop_rot_y, self._modeling_prop_rot_z,
            self._modeling_prop_scl_x, self._modeling_prop_scl_y, self._modeling_prop_scl_z,
        ):
            w.blockSignals(block)

    def _set_modeling_status(self, msg: str) -> None:
        self._modeling_status_label.setText(msg)

    # ------------------------------------------------------------------
    # camera
    # ------------------------------------------------------------------

    def _reset_camera(self) -> None:
        self._modeling_viewport._renderer.ResetCamera()
        cam = self._modeling_viewport._renderer.GetActiveCamera()
        cam.SetPosition(5, 4, 6)
        cam.SetFocalPoint(0, 0, 0)
        cam.SetViewUp(0, 0, 1)
        self._modeling_viewport.render()
        self._set_modeling_status("视角已重置")
