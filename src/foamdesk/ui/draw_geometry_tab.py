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
    kind: str
    actor: object  # vtkActor
    source: object  # vtk source
    visible: bool = True
    section: str = "stl"  # "domain" or "stl"
    opacity: float = 1.0
    source_path: str = ""
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    color: tuple[float, float, float] = (1.0, 0.82, 0.08)
    wire_actor: object | None = None
    point_actor: object | None = None


class DrawGeometryLogicMixin:
    """Mixin providing all modeling logic methods for MainWindow."""

    def _init_modeling_state(self) -> None:
        if hasattr(self, "_modeling_viewport") and self._modeling_viewport is not None:
            for obj in self._modeling_objects:
                self._remove_modeling_object_actors(obj)
        self._modeling_objects: list[GeometryObject] = []
        self._modeling_selected_index: int = -1
        self._modeling_active_section: str = "stl"
        self._interactive_edit_mode: str | None = None
        self._interactive_edit_drag: dict | None = None
        self._interactive_edit_gizmo_actors: list[object] = []
        self._interactive_edit_gizmo_axes: dict[str, np.ndarray] = {}
        self._interactive_edit_selection: dict | None = None
        self._interactive_edit_observer_tags: list[int] = []
        self._interactive_edit_old_style = None
        self._interactive_edit_handlers_installed = False
        self._load_modeling_state()
        if hasattr(self, "_modeling_viewport") and self._modeling_viewport is not None:
            for obj in self._modeling_objects:
                self._add_modeling_object_actors(obj)
                self._apply_transform(obj)
            self._apply_modeling_selection_styles()
            if hasattr(self, "_modeling_tree"):
                self._rebuild_tree()
            self._modeling_viewport.render()

    def _modeling_display_polydata(self, source):
        if isinstance(source, vtk.vtkPolyData):
            source_poly_data = source
        else:
            source.Update()
            source_poly_data = source.GetOutput()
        triangle = vtk.vtkTriangleFilter()
        triangle.SetInputData(source_poly_data)
        triangle.Update()
        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(triangle.GetOutput())
        clean.Update()
        output = vtk.vtkPolyData()
        output.DeepCopy(clean.GetOutput())
        return output

    def _create_modeling_actor_bundle(
        self,
        source,
        color: tuple[float, float, float] = (1.0, 0.82, 0.08),
        opacity: float = 1.0,
        visible: bool = True,
    ):
        display_poly_data = self._modeling_display_polydata(source)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(display_poly_data)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(0.95, 0.72, 0.02)
        actor.GetProperty().SetLineWidth(0.8)
        actor.SetVisibility(visible)

        edge_filter = vtk.vtkExtractEdges()
        edge_filter.SetInputData(display_poly_data)
        edge_mapper = vtk.vtkPolyDataMapper()
        edge_mapper.SetInputConnection(edge_filter.GetOutputPort())
        wire_actor = vtk.vtkActor()
        wire_actor.SetMapper(edge_mapper)
        wire_actor.GetProperty().SetColor(0.95, 0.72, 0.02)
        wire_actor.GetProperty().SetLineWidth(1.15)
        wire_actor.GetProperty().SetOpacity(0.95)
        wire_actor.SetVisibility(visible)

        vertex_filter = vtk.vtkVertexGlyphFilter()
        vertex_filter.SetInputData(display_poly_data)
        point_mapper = vtk.vtkPolyDataMapper()
        point_mapper.SetInputConnection(vertex_filter.GetOutputPort())
        point_actor = vtk.vtkActor()
        point_actor.SetMapper(point_mapper)
        point_actor.GetProperty().SetColor(1.0, 0.96, 0.30)
        point_actor.GetProperty().SetPointSize(4.2)
        point_actor.GetProperty().SetOpacity(0.92)
        point_actor.SetVisibility(visible)

        return actor, wire_actor, point_actor

    def _add_modeling_object_actors(self, obj: GeometryObject) -> None:
        if not hasattr(self, "_modeling_viewport") or self._modeling_viewport is None:
            return
        for actor in (obj.actor, obj.wire_actor, obj.point_actor):
            if actor is not None:
                self._modeling_viewport._renderer.AddActor(actor)

    def _remove_modeling_object_actors(self, obj: GeometryObject) -> None:
        if not hasattr(self, "_modeling_viewport") or self._modeling_viewport is None:
            return
        for actor in (obj.actor, obj.wire_actor, obj.point_actor):
            if actor is not None:
                self._modeling_viewport._renderer.RemoveActor(actor)

    def _set_modeling_object_visible(self, obj: GeometryObject, visible: bool) -> None:
        obj.visible = visible
        for actor in (obj.actor, obj.wire_actor, obj.point_actor):
            if actor is not None:
                actor.SetVisibility(visible)

    def _apply_modeling_selection_styles(self) -> None:
        for index, obj in enumerate(self._modeling_objects):
            selected = index == self._modeling_selected_index
            surface_color = (1.0, 0.08, 0.04) if selected else (1.0, 0.82, 0.08)
            edge_color = (0.95, 0.0, 0.0) if selected else (0.95, 0.68, 0.0)
            if obj.actor is not None:
                obj.actor.GetProperty().SetColor(*surface_color)
                obj.actor.GetProperty().SetEdgeColor(*edge_color)
                obj.actor.GetProperty().SetLineWidth(1.4 if selected else 0.8)
                obj.actor.GetProperty().EdgeVisibilityOn()
            if obj.wire_actor is not None:
                obj.wire_actor.GetProperty().SetColor(*edge_color)
                obj.wire_actor.GetProperty().SetLineWidth(2.2 if selected else 1.15)
                obj.wire_actor.GetProperty().SetOpacity(1.0 if selected else 0.95)
            if obj.point_actor is not None:
                obj.point_actor.GetProperty().SetColor(*surface_color)
                obj.point_actor.GetProperty().SetPointSize(6.0 if selected else 4.2)

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
            else:
                continue
            actor, wire_actor, point_actor = self._create_modeling_actor_bundle(
                src,
                tuple(d.get("color", [0.25, 0.74, 1.0])),
                d.get("opacity", 1.0),
                d.get("visible", True),
            )
            obj = GeometryObject(
                name=d["name"], kind=kind, actor=actor, source=src,
                wire_actor=wire_actor, point_actor=point_actor,
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

    def _delete_selected(self) -> None:
        if self._modeling_selected_index < 0 or self._modeling_selected_index >= len(self._modeling_objects):
            return
        obj = self._modeling_objects.pop(self._modeling_selected_index)
        self._remove_modeling_object_actors(obj)
        self._modeling_selected_index = -1
        self._rebuild_tree()
        self._apply_modeling_selection_styles()
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
        sketch_header = QTreeWidgetItem(["草图", ""])
        sketch_header.setData(0, Qt.ItemDataRole.UserRole, -30)
        sketch_header.setFlags(sketch_header.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
        tree.addTopLevelItem(domain_header)
        tree.addTopLevelItem(stl_header)
        tree.addTopLevelItem(sketch_header)
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
        active_sketch = getattr(self, "_active_sketch_index", -1)
        for si, sketch in enumerate(getattr(self, "_sketches", [])):
            marker = " ✎" if si == active_sketch else ""
            sketch_item = QTreeWidgetItem([sketch.name + marker, "草图"])
            sketch_item.setData(0, Qt.ItemDataRole.UserRole, 1000 + si)
            sketch_item.setFlags(sketch_item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            sketch_header.addChild(sketch_item)
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
        if idx == -30:
            return
        if idx >= 1000:
            self._enter_sketch_mode(int(idx) - 1000)
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
        self._set_modeling_object_visible(self._modeling_objects[i], visible)
        self._apply_modeling_selection_styles()
        self._modeling_viewport.render()
        self._save_modeling_state_if_ready()

    def _tree_context_menu(self, pos) -> None:
        item = self._modeling_tree.itemAt(pos)
        if item is None:
            return
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        if int(idx) < 0:
            self._set_modeling_status("模型树分组不能删除，请右键删除具体几何体或草图")
            return
        menu = QMenu(self)
        rename_act = menu.addAction("重命名")
        delete_act = menu.addAction("删除")
        action = menu.exec(self._modeling_tree.viewport().mapToGlobal(pos))
        if action == rename_act:
            self._modeling_tree.editItem(item, 0)
        elif action == delete_act:
            if int(idx) >= 1000:
                self._delete_sketch(int(idx) - 1000)
            elif int(idx) >= 0:
                self._select_object(int(idx))
                self._delete_selected()

    # ------------------------------------------------------------------
    # selection
    # ------------------------------------------------------------------

    def _select_object(self, index: int) -> None:
        if index < 0 or index >= len(self._modeling_objects):
            return
        self._modeling_selected_index = index
        self._apply_modeling_selection_styles()
        self._load_properties()
        self._set_property_enabled(True)
        self._set_modeling_status(f"选中 {self._modeling_objects[index].name}")

    def _activate_select_mode(self) -> None:
        self._set_modeling_status("选择模式：请在模型树中点击选择模型")

    def _finish_draw_geometry(self) -> None:
        if self._current_project is None:
            self._set_modeling_status("请先新建或打开项目")
            return
        checked_objects = [obj for obj in self._modeling_objects if obj.visible]
        if not checked_objects:
            self._set_modeling_status("请在模型树中勾选要导入网格页的几何体")
            return

        from pathlib import Path
        from foamdesk.ui.main_window_geometry_logic import MeshImportAsset

        cache_dir = self._draw_geometry_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        exported_stl_assets: list[MeshImportAsset] = []
        exported_domain_assets: list[MeshImportAsset] = []
        for obj in checked_objects:
            export_prefix = "domain_patch" if obj.section == "domain" else "obstacle"
            output_path = self._unique_draw_export_path(cache_dir, f"{export_prefix}_{obj.name}.stl")
            poly_data = self._transformed_object_polydata(obj)
            if poly_data is None or poly_data.GetNumberOfPoints() == 0:
                continue
            if obj.section == "domain":
                exported_domain_assets.extend(
                    self._domain_face_assets_from_polydata(obj.name, poly_data, cache_dir)
                )
                continue
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(str(output_path))
            writer.SetInputData(poly_data)
            writer.Write()
            asset = MeshImportAsset(
                name=Path(output_path).stem,
                source_path=output_path,
                polydata=poly_data,
            )
            exported_stl_assets.append(asset)
        if not exported_stl_assets and not exported_domain_assets:
            self._set_modeling_status("勾选几何导出失败，请检查模型是否为空")
            return

        self._workspace_tabs.setCurrentIndex(self.TAB_MESH_GENERATION)
        if not hasattr(self, "_mesh_imports"):
            self._init_mesh_import_state()
        if exported_stl_assets:
            self._mesh_imports.extend(exported_stl_assets)
            self._mesh_import_selected_index = len(self._mesh_imports) - 1
        if exported_domain_assets:
            first_new_domain_row = len(self._cad_domain_imports)
            self._cad_domain_imports.extend(exported_domain_assets)
            if hasattr(self, "_mesh_domain_type_combo"):
                cad_index = self._mesh_domain_type_combo.findData("cad")
                if cad_index >= 0:
                    self._mesh_domain_type_combo.setCurrentIndex(cad_index)
        else:
            first_new_domain_row = -1
        self._domain_bounds_manual = False
        self._rebuild_mesh_import_combo()
        if hasattr(self, "_rebuild_cad_domain_file_list"):
            self._rebuild_cad_domain_file_list()
        if hasattr(self, "_init_domain_face_table"):
            self._init_domain_face_table()
        if first_new_domain_row >= 0 and hasattr(self, "_domain_face_table"):
            row = min(first_new_domain_row, self._domain_face_table.rowCount() - 1)
            if row >= 0:
                self._domain_face_table.selectRow(row)
                self._highlighted_domain_face = row
        self._auto_fill_domain_bounds()
        if exported_domain_assets and hasattr(self, "_auto_recommend_location_in_mesh"):
            self._auto_recommend_location_in_mesh(update_view=False)
        if hasattr(self, "_update_cad_domain_status"):
            self._update_cad_domain_status()
        self._redraw_mesh_grid_vtk()
        self._save_mesh_workflow_state()
        self._refresh_geometry_panel()
        self._set_modeling_status(
            f"已完成绘制：追加障碍物 {len(exported_stl_assets)} 个，计算域 {len(exported_domain_assets)} 个到网格生成页"
        )
        self._set_status("绘制几何已追加导入网格生成页。")

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

    def _domain_face_assets_from_polydata(self, object_name: str, poly_data, cache_dir) -> list:
        """Split a drawn domain into planar face assets for boundary editing.

        A closed cube-like domain should appear in Group 2 as one row per face,
        so users can later turn any face into inlet/outlet/wall. Curved or
        highly tessellated surfaces fall back to the whole surface if no stable
        planar grouping is found.
        """
        from pathlib import Path
        from foamdesk.ui.main_window_geometry_logic import MeshImportAsset

        face_groups = self._split_polydata_by_planar_faces(poly_data)
        if not face_groups:
            face_groups = [("face1", poly_data)]

        assets: list[MeshImportAsset] = []
        for index, (face_label, face_poly_data) in enumerate(face_groups, start=1):
            if face_poly_data is None or face_poly_data.GetNumberOfPoints() == 0:
                continue
            output_path = self._unique_draw_export_path(
                cache_dir,
                f"domain_patch_{object_name}_{face_label or f'face{index}'}.stl",
            )
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(str(output_path))
            writer.SetInputData(face_poly_data)
            writer.Write()
            assets.append(
                MeshImportAsset(
                    name=Path(output_path).stem,
                    source_path=output_path,
                    polydata=face_poly_data,
                )
            )
        return assets

    def _split_polydata_by_planar_faces(self, poly_data) -> list[tuple[str, object]]:
        triangle = vtk.vtkTriangleFilter()
        triangle.SetInputData(poly_data)
        triangle.Update()
        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(triangle.GetOutputPort())
        clean.Update()
        source = clean.GetOutput()
        if source is None or source.GetNumberOfCells() == 0:
            return []

        bounds = source.GetBounds()
        diag = 1.0
        if bounds and all(np.isfinite(bounds)):
            diag = max(
                float(np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])),
                1e-9,
            )
        distance_scale = max(diag * 0.001, 1e-8)
        groups: dict[tuple[float, float, float, int], list[int]] = {}
        id_list = vtk.vtkIdList()
        points = source.GetPoints()
        for cell_id in range(source.GetNumberOfCells()):
            source.GetCellPoints(cell_id, id_list)
            if id_list.GetNumberOfIds() < 3:
                continue
            pts = [np.array(points.GetPoint(id_list.GetId(i)), dtype=float) for i in range(id_list.GetNumberOfIds())]
            normal = None
            for i in range(1, len(pts) - 1):
                candidate = np.cross(pts[i] - pts[0], pts[i + 1] - pts[0])
                norm = float(np.linalg.norm(candidate))
                if norm > 1e-12:
                    normal = candidate / norm
                    break
            if normal is None:
                continue
            if tuple(normal) < tuple(-normal):
                normal = -normal
            centroid = np.mean(pts, axis=0)
            offset = float(np.dot(normal, centroid))
            key = (
                round(float(normal[0]), 3),
                round(float(normal[1]), 3),
                round(float(normal[2]), 3),
                int(round(offset / distance_scale)),
            )
            groups.setdefault(key, []).append(cell_id)

        if len(groups) <= 1 or len(groups) > 64:
            return []

        result: list[tuple[str, object]] = []
        for face_index, (_key, cell_ids) in enumerate(sorted(groups.items(), key=lambda item: item[0]), start=1):
            if not cell_ids:
                continue
            new_points = vtk.vtkPoints()
            new_polys = vtk.vtkCellArray()
            point_map: dict[int, int] = {}
            for cell_id in cell_ids:
                source.GetCellPoints(cell_id, id_list)
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(id_list.GetNumberOfIds())
                for i in range(id_list.GetNumberOfIds()):
                    old_id = int(id_list.GetId(i))
                    if old_id not in point_map:
                        point_map[old_id] = new_points.InsertNextPoint(points.GetPoint(old_id))
                    polygon.GetPointIds().SetId(i, point_map[old_id])
                new_polys.InsertNextCell(polygon)
            face_poly_data = vtk.vtkPolyData()
            face_poly_data.SetPoints(new_points)
            face_poly_data.SetPolys(new_polys)
            face_clean = vtk.vtkCleanPolyData()
            face_clean.SetInputData(face_poly_data)
            face_clean.Update()
            out = vtk.vtkPolyData()
            out.DeepCopy(face_clean.GetOutput())
            if out.GetNumberOfPoints() > 0:
                result.append((f"face{face_index}", out))
        return result

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
        old_wire_actor = obj.wire_actor
        old_point_actor = obj.point_actor
        for actor in (old_wire_actor, old_point_actor):
            if actor is not None and hasattr(self, "_modeling_viewport"):
                self._modeling_viewport._renderer.RemoveActor(actor)
        actor, wire_actor, point_actor = self._create_modeling_actor_bundle(output, obj.color, obj.opacity, obj.visible)
        obj.actor.SetMapper(actor.GetMapper())
        obj.actor.GetProperty().SetColor(*obj.color)
        obj.actor.GetProperty().SetOpacity(obj.opacity)
        obj.actor.GetProperty().EdgeVisibilityOn()
        obj.actor.GetProperty().SetEdgeColor(0.02, 0.06, 0.08)
        obj.actor.GetProperty().SetLineWidth(0.6)
        obj.wire_actor = wire_actor
        obj.point_actor = point_actor
        self._add_modeling_object_actors(GeometryObject(name=obj.name, kind=obj.kind, actor=None, source=obj.source, wire_actor=wire_actor, point_actor=point_actor))
        self._apply_transform(obj)
        self._save_modeling_state_if_ready()

    def _install_interactive_edit_handlers(self) -> None:
        if getattr(self, "_interactive_edit_handlers_installed", False):
            return
        if not hasattr(self, "_modeling_viewport") or self._modeling_viewport is None:
            return
        interactor = self._modeling_viewport._interactor
        self._interactive_edit_observer_tags = [
            interactor.AddObserver("LeftButtonPressEvent", self._on_interactive_edit_press, 1.0),
            interactor.AddObserver("MouseMoveEvent", self._on_interactive_edit_move, 1.0),
            interactor.AddObserver("LeftButtonReleaseEvent", self._on_interactive_edit_release, 1.0),
        ]
        self._interactive_edit_handlers_installed = True

    def _edit_selected_vertices(self, region_type: str) -> None:
        obj = self._selected_modeling_object()
        if obj is None:
            return
        source_pd = self._source_polydata(obj)
        if source_pd is None or source_pd.GetNumberOfPoints() == 0:
            self._set_modeling_status("当前几何体没有可编辑点")
            return
        self._install_interactive_edit_handlers()
        self._interactive_edit_mode = region_type
        self._interactive_edit_drag = None
        if self._interactive_edit_old_style is None:
            self._interactive_edit_old_style = self._modeling_viewport._interactor.GetInteractorStyle()
        self._modeling_viewport._interactor.SetInteractorStyle(vtk.vtkInteractorStyleUser())
        labels = {"point": "点", "edge": "线", "face": "面"}
        self._set_modeling_status(
            f"交互式{labels.get(region_type, '面')}编辑：在 3D 视图里按住目标{labels.get(region_type, '面')}拖动，松开鼠标保存"
        )

    def _finish_interactive_edit_mode(self, message: str = "") -> None:
        self._interactive_edit_drag = None
        self._interactive_edit_mode = None
        self._interactive_edit_selection = None
        self._clear_interactive_edit_gizmo()
        if self._interactive_edit_old_style is not None and hasattr(self, "_modeling_viewport"):
            self._modeling_viewport._interactor.SetInteractorStyle(self._interactive_edit_old_style)
        self._interactive_edit_old_style = None
        if message:
            self._set_modeling_status(message)

    def _clear_interactive_edit_gizmo(self) -> None:
        if hasattr(self, "_modeling_viewport") and self._modeling_viewport is not None:
            for actor in getattr(self, "_interactive_edit_gizmo_actors", []):
                self._modeling_viewport._renderer.RemoveActor(actor)
        self._interactive_edit_gizmo_actors = []
        self._interactive_edit_gizmo_axes = {}

    def _vtk_actor_key(self, actor) -> str:
        if actor is None:
            return ""
        try:
            return actor.GetAddressAsString("")
        except Exception:
            return str(id(actor))

    def _set_edit_delta_fields(self, delta: np.ndarray) -> None:
        widgets = (
            getattr(self, "_modeling_edit_dx", None),
            getattr(self, "_modeling_edit_dy", None),
            getattr(self, "_modeling_edit_dz", None),
        )
        if any(widget is None for widget in widgets):
            return
        for widget, value in zip(widgets, delta):
            widget.blockSignals(True)
            widget.setValue(float(value))
            widget.blockSignals(False)

    def _edit_delta_from_fields(self) -> np.ndarray:
        return np.array([
            float(getattr(self, "_modeling_edit_dx").value()),
            float(getattr(self, "_modeling_edit_dy").value()),
            float(getattr(self, "_modeling_edit_dz").value()),
        ], dtype=float)

    def _show_interactive_edit_gizmo(self, obj: GeometryObject, poly_data, selected_ids: list[int]) -> None:
        self._clear_interactive_edit_gizmo()
        if not selected_ids:
            return
        selected_world = [self._local_to_world_point(obj, poly_data.GetPoint(point_id)) for point_id in selected_ids]
        center = np.mean(np.array(selected_world, dtype=float), axis=0)
        bounds = obj.actor.GetBounds() if obj.actor is not None else poly_data.GetBounds()
        if bounds is None or not all(np.isfinite(bounds)):
            length = 0.5
        else:
            diagonal = float(np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]))
            length = max(diagonal * 0.28, 0.15)
        radius = max(length * 0.035, 0.006)
        axes = [
            (np.array([1.0, 0.0, 0.0], dtype=float), (1.0, 0.12, 0.10)),
            (np.array([0.0, 1.0, 0.0], dtype=float), (0.12, 0.85, 0.18)),
            (np.array([0.0, 0.0, 1.0], dtype=float), (0.16, 0.35, 1.0)),
        ]
        for axis, color in axes:
            arrow = vtk.vtkArrowSource()
            arrow.SetTipLength(0.28)
            arrow.SetTipRadius(radius * 2.6)
            arrow.SetShaftRadius(radius)
            arrow.Update()
            transform = vtk.vtkTransform()
            transform.PostMultiply()
            transform.Scale(length, length, length)
            base = np.array([1.0, 0.0, 0.0], dtype=float)
            rotation_axis = np.cross(base, axis)
            rotation_norm = float(np.linalg.norm(rotation_axis))
            if rotation_norm > 1e-12:
                rotation_axis = rotation_axis / rotation_norm
                angle = float(np.degrees(np.arccos(np.clip(float(np.dot(base, axis)), -1.0, 1.0))))
                transform.RotateWXYZ(angle, float(rotation_axis[0]), float(rotation_axis[1]), float(rotation_axis[2]))
            elif float(np.dot(base, axis)) < 0.0:
                transform.RotateWXYZ(180.0, 0.0, 0.0, 1.0)
            transform.Translate(float(center[0]), float(center[1]), float(center[2]))
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetInputConnection(arrow.GetOutputPort())
            tf.SetTransform(transform)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(tf.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetOpacity(0.95)
            actor.GetProperty().SetSpecular(0.25)
            actor.GetProperty().SetSpecularPower(16)
            self._modeling_viewport._renderer.AddActor(actor)
            self._interactive_edit_gizmo_actors.append(actor)
            self._interactive_edit_gizmo_axes[self._vtk_actor_key(actor)] = axis
        self._modeling_viewport.render()

    def _interactive_edit_polydata(self, obj: GeometryObject):
        mapper = obj.actor.GetMapper() if obj.actor is not None else None
        if mapper is not None and mapper.GetInput() is not None:
            return mapper.GetInput()
        return self._source_polydata(obj)

    def _object_transform(self, obj: GeometryObject):
        transform = vtk.vtkTransform()
        transform.Translate(obj.position[0], obj.position[1], obj.position[2])
        transform.RotateX(obj.rotation[0])
        transform.RotateY(obj.rotation[1])
        transform.RotateZ(obj.rotation[2])
        transform.Scale(obj.scale[0], obj.scale[1], obj.scale[2])
        return transform

    def _local_to_world_point(self, obj: GeometryObject, point) -> np.ndarray:
        return np.array(self._object_transform(obj).TransformPoint(point), dtype=float)

    def _world_delta_to_local(self, obj: GeometryObject, delta_world: np.ndarray) -> np.ndarray:
        transform = self._object_transform(obj)
        transform.Inverse()
        return np.array(transform.TransformVector(delta_world), dtype=float)

    def _pick_selected_edit_ids(self, obj: GeometryObject, poly_data, cell_id: int, pick_world: np.ndarray) -> list[int]:
        cell = poly_data.GetCell(cell_id)
        if cell is None:
            return []
        point_ids = [int(cell.GetPointId(i)) for i in range(cell.GetNumberOfPoints())]
        if not point_ids:
            return []
        if self._interactive_edit_mode == "face":
            return point_ids
        world_points = [self._local_to_world_point(obj, poly_data.GetPoint(point_id)) for point_id in point_ids]
        if self._interactive_edit_mode == "point":
            closest_index = int(np.argmin([np.linalg.norm(point - pick_world) for point in world_points]))
            return [point_ids[closest_index]]
        edge_candidates: list[tuple[float, int, int]] = []
        for index, start_id in enumerate(point_ids):
            end_id = point_ids[(index + 1) % len(point_ids)]
            midpoint = (world_points[index] + world_points[(index + 1) % len(world_points)]) * 0.5
            edge_candidates.append((float(np.linalg.norm(midpoint - pick_world)), start_id, end_id))
        _, start_id, end_id = min(edge_candidates, key=lambda item: item[0])
        return [start_id, end_id]

    def _screen_drag_to_world_delta(self, obj: GeometryObject, start_pos: tuple[int, int], current_pos: tuple[int, int]) -> np.ndarray:
        dx = float(current_pos[0] - start_pos[0])
        dy = float(current_pos[1] - start_pos[1])
        renderer = self._modeling_viewport._renderer
        camera = renderer.GetActiveCamera()
        view_up = np.array(camera.GetViewUp(), dtype=float)
        view_up = view_up / max(float(np.linalg.norm(view_up)), 1e-12)
        direction = np.array(camera.GetDirectionOfProjection(), dtype=float)
        direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
        view_right = np.cross(direction, view_up)
        view_right = view_right / max(float(np.linalg.norm(view_right)), 1e-12)
        bounds = obj.actor.GetBounds() if obj.actor is not None else None
        if bounds is None or not all(np.isfinite(bounds)):
            scale = 0.002
        else:
            diagonal = float(np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]))
            width, height = self._modeling_viewport._vtk_widget.GetRenderWindow().GetSize()
            scale = diagonal / max(float(max(width, height)), 1.0) * 1.6
        return (view_right * dx + view_up * dy) * scale

    def _on_interactive_edit_press(self, _obj, _event) -> None:
        if not self._interactive_edit_mode:
            return
        obj = self._selected_modeling_object()
        if obj is None:
            return
        x, y = self._modeling_viewport._interactor.GetEventPosition()
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.01)
        if not picker.Pick(x, y, 0, self._modeling_viewport._renderer):
            self._set_modeling_status("没有点中当前几何体，请在模型表面、边线或顶点附近按住拖动")
            return
        picked_actor = picker.GetActor()
        picked_key = self._vtk_actor_key(picked_actor)
        if picked_key in getattr(self, "_interactive_edit_gizmo_axes", {}):
            selection = self._interactive_edit_selection
            if not selection:
                return
            selected_obj = selection["object"]
            poly_data = selection["poly_data"]
            self._interactive_edit_drag = {
                "object": selected_obj,
                "poly_data": poly_data,
                "selected_ids": selection["selected_ids"],
                "axis_world": self._interactive_edit_gizmo_axes[picked_key],
                "start_pos": (int(x), int(y)),
                "base_points": np.array(selection.get("base_points"), dtype=float),
            }
            return
        if picked_actor not in (obj.actor, obj.wire_actor, obj.point_actor):
            self._set_modeling_status("请先点选当前几何体上的点/线/面，或拖动已出现的红绿蓝轴")
            return
        poly_data = self._interactive_edit_polydata(obj)
        cell_id = int(picker.GetCellId())
        if poly_data is None or cell_id < 0:
            return
        pick_world = np.array(picker.GetPickPosition(), dtype=float)
        selected_ids = self._pick_selected_edit_ids(obj, poly_data, cell_id, pick_world)
        if not selected_ids:
            self._set_modeling_status("没有找到可拖动的点")
            return
        self._interactive_edit_selection = {
            "object": obj,
            "poly_data": poly_data,
            "selected_ids": selected_ids,
            "base_points": np.array([poly_data.GetPoint(point_id) for point_id in range(poly_data.GetNumberOfPoints())], dtype=float),
        }
        self._set_edit_delta_fields(np.zeros(3, dtype=float))
        self._show_interactive_edit_gizmo(obj, poly_data, selected_ids)
        labels = {"point": "点", "edge": "线", "face": "面"}
        self._set_modeling_status(
            f"已选中{labels.get(self._interactive_edit_mode or '', '面')}，拖动红/绿/蓝轴可沿 X/Y/Z 方向编辑"
        )

    def _on_interactive_edit_move(self, _obj, _event) -> None:
        drag = self._interactive_edit_drag
        if not drag:
            return
        obj = drag["object"]
        poly_data = drag["poly_data"]
        x, y = self._modeling_viewport._interactor.GetEventPosition()
        delta_world = self._screen_drag_to_world_delta(obj, drag["start_pos"], (int(x), int(y)))
        axis_world = drag.get("axis_world")
        if axis_world is not None:
            axis_world = np.array(axis_world, dtype=float)
            axis_world = axis_world / max(float(np.linalg.norm(axis_world)), 1e-12)
            delta_world = axis_world * float(np.dot(delta_world, axis_world))
        delta_local = self._world_delta_to_local(obj, delta_world)
        self._set_edit_delta_fields(delta_local)
        self._apply_edit_delta_to_polydata(poly_data, drag["base_points"], drag["selected_ids"], delta_local)
        self._refresh_edit_actor_mappers(obj)
        if drag.get("axis_world") is not None:
            self._show_interactive_edit_gizmo(obj, poly_data, drag["selected_ids"])
        self._modeling_viewport.render()

    def _apply_edit_delta_to_polydata(self, poly_data, base_points, selected_ids: list[int], delta_local: np.ndarray) -> None:
        points = vtk.vtkPoints()
        selected = set(selected_ids)
        for point_id, point in enumerate(base_points):
            moved = point + delta_local if point_id in selected else point
            points.InsertNextPoint(float(moved[0]), float(moved[1]), float(moved[2]))
        poly_data.SetPoints(points)
        poly_data.Modified()

    def _refresh_edit_actor_mappers(self, obj: GeometryObject) -> None:
        for actor in (obj.actor, obj.wire_actor, obj.point_actor):
            if actor is not None and actor.GetMapper() is not None:
                actor.GetMapper().Modified()

    def _apply_interactive_edit_delta_from_fields(self) -> None:
        selection = self._interactive_edit_selection
        if not selection:
            self._set_modeling_status("请先进入点/线/面编辑，并在 3D 视图中选中要编辑的点、线或面")
            return
        obj = selection["object"]
        poly_data = selection["poly_data"]
        delta_local = self._edit_delta_from_fields()
        self._apply_edit_delta_to_polydata(poly_data, selection["base_points"], selection["selected_ids"], delta_local)
        self._refresh_edit_actor_mappers(obj)
        self._show_interactive_edit_gizmo(obj, poly_data, selection["selected_ids"])
        edited = vtk.vtkPolyData()
        edited.DeepCopy(poly_data)
        moved_count = len(selection["selected_ids"])
        self._replace_object_polydata(obj, edited)
        self._modeling_viewport.render()
        labels = {"point": "点", "edge": "线", "face": "面"}
        mode_label = labels.get(self._interactive_edit_mode or "", "面")
        self._finish_interactive_edit_mode(f"数值位移{mode_label}编辑完成：移动 {moved_count} 个点")

    def _on_interactive_edit_release(self, _obj, _event) -> None:
        drag = self._interactive_edit_drag
        if not drag:
            return
        obj = drag["object"]
        edited = vtk.vtkPolyData()
        edited.DeepCopy(drag["poly_data"])
        moved_count = len(drag["selected_ids"])
        self._replace_object_polydata(obj, edited)
        self._modeling_viewport.render()
        labels = {"point": "点", "edge": "线", "face": "面"}
        mode_label = labels.get(self._interactive_edit_mode or "", "面")
        self._finish_interactive_edit_mode(f"三轴操纵器{mode_label}编辑完成：移动 {moved_count} 个点")

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
        actor, wire_actor, point_actor = self._create_modeling_actor_bundle(output, target.color, target.opacity)
        result = GeometryObject(
            name=result_name,
            kind="stl",
            actor=actor,
            source=output,
            wire_actor=wire_actor, point_actor=point_actor,
            section=target.section,
            color=target.color,
            opacity=target.opacity,
        )
        self._modeling_objects.append(result)
        self._add_modeling_object_actors(result)
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
        for actor in (obj.actor, obj.wire_actor, obj.point_actor):
            if actor is not None:
                actor.SetUserTransform(transform)
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
