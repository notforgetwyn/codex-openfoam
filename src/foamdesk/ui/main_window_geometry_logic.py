from __future__ import annotations

import json
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vtk
from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import QComboBox, QFileDialog, QLineEdit, QMessageBox, QSpinBox, QTableWidgetItem
from vtkmodules.vtkIOGeometry import vtkSTLReader

from foamdesk.ui import domain_templates
from foamdesk.ui.visualization_widgets import NativeVtkPreviewWidget


@dataclass
class MeshImportAsset:
    name: str
    source_path: Path
    polydata: object  # vtkPolyData (cached for redraw)
    visible: bool = True
    translucent: bool = False
    color: tuple[float, float, float] = (0.58, 0.62, 0.66)


BOUNDARY_TYPE_OPTIONS = ["patch", "wall", "symmetry", "empty", "cyclic"]

DOMAIN_FACE_DEFAULTS = [
    {"label": "X-min", "face": "0 4 7 3", "name": "inlet",   "type": "patch"},
    {"label": "X-max", "face": "1 2 6 5", "name": "outlet",  "type": "patch"},
    {"label": "Y-min", "face": "0 1 2 3", "name": "walls",   "type": "wall"},
    {"label": "Y-max", "face": "4 5 6 7", "name": "walls",   "type": "wall"},
    {"label": "Z-min", "face": "0 1 5 4", "name": "walls",   "type": "wall"},
    {"label": "Z-max", "face": "3 2 6 7", "name": "walls",   "type": "wall"},
]


class GeometryLogicMixin:







    def _refresh_geometry_panel(self) -> None:
        if self._current_project is None:
            return
        if hasattr(self, "_mesh_grid_vtk") and hasattr(self, "_mesh_imports"):
            self._redraw_mesh_grid_vtk()
        if hasattr(self, "_domain_face_table"):
            self._init_domain_face_table()














    # ── mesh import / Group 1 logic ──────────────────────────

    def _init_mesh_import_state(self) -> None:
        self._mesh_imports: list[MeshImportAsset] = []
        self._mesh_import_selected_index: int = -1
        self._cad_domain_imports: list[MeshImportAsset] = []
        self._domain_bounds_manual: bool = False
        self._last_checkmesh_output: str = ""
        self._highlighted_domain_face: int = -1

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
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        if not self._mesh_imports:
            self._mesh_grid_vtk.clear()
            return
        confirm = QMessageBox.question(
            self,
            "清空几何",
            "确定要清空当前 Case 的导入几何吗？\n\n"
            "这会删除 constant/triSurface 下的 STL 文件，并清除几何导入记录。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        case_dir = self._current_project.case_dir
        tri_surface_dir = case_dir / "constant" / "triSurface"
        removed_count = 0
        if tri_surface_dir.exists():
            for path in tri_surface_dir.glob("*.stl"):
                try:
                    path.unlink()
                    removed_count += 1
                except OSError as error:
                    self._show_error(f"删除几何失败：{path.name}\n{error}")
                    return
            for metadata_name in ("geometry_manifest.json", "snappy_config.json"):
                metadata_path = tri_surface_dir / metadata_name
                if metadata_path.exists():
                    try:
                        metadata_path.unlink()
                    except OSError as error:
                        self._show_error(f"删除几何记录失败：{metadata_name}\n{error}")
                        return

        snappy_dict = case_dir / "system" / "snappyHexMeshDict"
        if snappy_dict.exists():
            try:
                snappy_dict.unlink()
            except OSError:
                pass

        self._mesh_imports.clear()
        self._mesh_import_selected_index = -1
        self._rebuild_mesh_import_combo()
        self._mesh_grid_vtk.clear()
        self._append_log(f"已清空导入几何：删除 {removed_count} 个 STL 文件。")
        self._set_status("导入几何已清空。")


    def _load_stl_asset(self, source_path: Path, color: tuple[float, float, float] | None = None) -> MeshImportAsset | None:
        reader = vtkSTLReader()
        reader.SetFileName(str(source_path))
        reader.Update()
        polydata = reader.GetOutput()
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            return None
        out = vtk.vtkPolyData()
        out.DeepCopy(polydata)
        return MeshImportAsset(
            name=source_path.stem,
            source_path=source_path,
            polydata=out,
            color=color or (0.58, 0.62, 0.66),
        )

    def _split_ascii_stl_asset_by_solids(self, asset: MeshImportAsset) -> list[MeshImportAsset]:
        """Split an ASCII STL by `solid name` blocks before geometric guessing.

        This supports one file such as:
            solid inlet
            ...
            endsolid inlet
            solid outlet
            ...
            endsolid outlet
            solid wall
            ...
            endsolid wall

        Binary STL and single-solid ASCII STL fall back to connected-region split.
        """
        try:
            text = asset.source_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []
        if "\0" in text[:512] or "solid" not in text[:256].lower():
            return []

        solids: list[tuple[str, list[tuple[float, float, float]]]] = []
        current_name: str | None = None
        current_vertices: list[tuple[float, float, float]] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            keyword = parts[0].lower()
            if keyword == "solid":
                if current_name is not None and current_vertices:
                    solids.append((current_name, current_vertices))
                current_name = " ".join(parts[1:]).strip() or asset.name
                current_vertices = []
            elif keyword == "vertex" and current_name is not None and len(parts) >= 4:
                try:
                    current_vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                except ValueError:
                    continue
            elif keyword == "endsolid":
                if current_name is not None and current_vertices:
                    solids.append((current_name, current_vertices))
                current_name = None
                current_vertices = []

        if current_name is not None and current_vertices:
            solids.append((current_name, current_vertices))

        named_solids = [(name, verts) for name, verts in solids if len(verts) >= 3]
        if len(named_solids) <= 1:
            return []

        assets: list[MeshImportAsset] = []
        used_names: set[str] = set()
        for solid_index, (name, vertices) in enumerate(named_solids, start=1):
            points = vtk.vtkPoints()
            triangles = vtk.vtkCellArray()
            tri_count = len(vertices) // 3
            for tri_i in range(tri_count):
                ids = []
                for v in vertices[tri_i * 3: tri_i * 3 + 3]:
                    ids.append(points.InsertNextPoint(float(v[0]), float(v[1]), float(v[2])))
                tri = vtk.vtkTriangle()
                for j, pid in enumerate(ids):
                    tri.GetPointIds().SetId(j, pid)
                triangles.InsertNextCell(tri)
            if tri_count <= 0:
                continue
            pd = vtk.vtkPolyData()
            pd.SetPoints(points)
            pd.SetPolys(triangles)
            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputData(pd)
            cleaner.Update()
            out = vtk.vtkPolyData()
            out.DeepCopy(cleaner.GetOutput())
            clean_name = self._safe_openfoam_patch_name(name) or f"{asset.name}_solid{solid_index}"
            base_name = clean_name
            suffix = 2
            while clean_name in used_names:
                clean_name = f"{base_name}_{suffix}"
                suffix += 1
            used_names.add(clean_name)
            assets.append(MeshImportAsset(
                name=clean_name,
                source_path=asset.source_path,
                polydata=out,
                color=asset.color,
            ))
        return assets

    def _safe_openfoam_patch_name(self, name: str) -> str:
        import re
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", (name or "").strip())
        cleaned = cleaned.strip("_")
        if not cleaned:
            return ""
        if cleaned[0].isdigit():
            cleaned = f"patch_{cleaned}"
        return cleaned

    def _split_stl_asset_by_connected_regions(self, asset: MeshImportAsset) -> list[MeshImportAsset]:
        """Split one STL into CAD-domain boundary rows.

        Priority:
        1) ASCII STL `solid name` blocks, preserving names such as inlet/outlet/wall.
        2) VTK connected regions, for files containing disconnected unnamed patches.
        """
        solid_assets = self._split_ascii_stl_asset_by_solids(asset)
        if len(solid_assets) > 1:
            return solid_assets

        conn = vtk.vtkPolyDataConnectivityFilter()
        conn.SetInputData(asset.polydata)
        conn.SetExtractionModeToAllRegions()
        conn.ColorRegionsOn()
        conn.Update()
        region_count = int(conn.GetNumberOfExtractedRegions())
        if region_count <= 1:
            return [asset]

        parts: list[MeshImportAsset] = []
        for region_id in range(region_count):
            region = vtk.vtkPolyDataConnectivityFilter()
            region.SetInputData(asset.polydata)
            region.SetExtractionModeToSpecifiedRegions()
            region.AddSpecifiedRegion(region_id)
            region.Update()
            pd = vtk.vtkPolyData()
            pd.DeepCopy(region.GetOutput())
            if pd.GetNumberOfPoints() == 0:
                continue
            parts.append(MeshImportAsset(
                name=f"{asset.name}_part{region_id + 1}",
                source_path=asset.source_path,
                polydata=pd,
                color=asset.color,
            ))
        if len(parts) <= 1:
            return [asset]
        self._infer_cad_domain_part_names(parts, asset.name)
        return parts

    def _infer_cad_domain_part_names(self, parts: list[MeshImportAsset], base_name: str) -> None:
        bounds = [part.polydata.GetBounds() for part in parts]
        valid = [b for b in bounds if b and all(np.isfinite(b))]
        if len(valid) < 3:
            return
        overall = np.array([
            min(b[0] for b in valid), max(b[1] for b in valid),
            min(b[2] for b in valid), max(b[3] for b in valid),
            min(b[4] for b in valid), max(b[5] for b in valid),
        ], dtype=float)
        spans = np.array([overall[1] - overall[0], overall[3] - overall[2], overall[5] - overall[4]], dtype=float)
        axis = int(np.argmax(spans))
        centers = []
        for i, b in enumerate(bounds):
            if not b or not all(np.isfinite(b)):
                centers.append((i, 0.0))
                continue
            centers.append((i, (float(b[axis * 2]) + float(b[axis * 2 + 1])) * 0.5))
        ordered = sorted(centers, key=lambda item: item[1])
        inlet_i = ordered[0][0]
        outlet_i = ordered[-1][0]
        wall_count = 0
        for i, part in enumerate(parts):
            if i == inlet_i:
                part.name = "inlet"
            elif i == outlet_i:
                part.name = "outlet"
            else:
                wall_count += 1
                part.name = "wall" if wall_count == 1 else f"wall_{wall_count}"

    def _import_cad_domain_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "导入计算域 STL 几何体",
            "/home/shihuayue/codex_project1",
            "STL 文件 (*.stl *.STL);;所有文件 (*)",
        )
        if not paths:
            return
        new_assets: list[MeshImportAsset] = []
        existing_paths = {str(a.source_path.resolve()) for a in self._cad_domain_imports if a.source_path.exists()}
        existing_names = {a.name for a in self._cad_domain_imports}
        skipped = 0
        for fp in paths:
            source_path = Path(fp)
            resolved = str(source_path.resolve()) if source_path.exists() else str(source_path)
            if resolved in existing_paths:
                skipped += 1
                continue
            asset = self._load_stl_asset(source_path, color=(0.28, 0.62, 1.0))
            if asset is None:
                continue
            split_assets = self._split_stl_asset_by_connected_regions(asset)
            for split_asset in split_assets:
                base_name = split_asset.name
                unique_name = base_name
                suffix = 2
                while unique_name in existing_names:
                    unique_name = f"{base_name}_{suffix}"
                    suffix += 1
                split_asset.name = unique_name
                existing_names.add(unique_name)
                new_assets.append(split_asset)
            existing_paths.add(resolved)
        if not new_assets:
            if skipped:
                self._set_status("选择的计算域 STL 已经导入过，未重复添加。")
                return
            self._show_error("没有读取到有效的计算域 STL。")
            return
        self._cad_domain_imports.extend(new_assets)
        if hasattr(self, "_mesh_domain_type_combo"):
            idx = self._mesh_domain_type_combo.findData("cad")
            if idx >= 0:
                self._mesh_domain_type_combo.setCurrentIndex(idx)
        self._rebuild_cad_domain_file_list()
        self._init_domain_face_table()
        self._auto_recommend_location_in_mesh(update_view=False)
        self._update_cad_domain_status()
        self._redraw_mesh_grid_vtk()
        self._save_mesh_workflow_state()
        total = len(self._cad_domain_imports)
        suffix = f"，跳过 {skipped} 个重复文件" if skipped else ""
        self._set_status(f"本次追加导入 {len(new_assets)} 个计算域 STL，当前共 {total} 个边界面{suffix}。")

    def _rebuild_cad_domain_file_list(self) -> None:
        table = getattr(self, "_cad_domain_file_table", None)
        if table is None:
            return
        selected_row = table.currentRow()
        table.blockSignals(True)
        table.setRowCount(0)
        for i, asset in enumerate(self._cad_domain_imports):
            table.insertRow(i)
            table.setRowHeight(i, 30)
            values = [str(i + 1), asset.name, asset.source_path.name, str(asset.source_path)]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(i, col, item)
        table.blockSignals(False)
        if table.rowCount() > 0:
            row = min(max(selected_row, 0), table.rowCount() - 1)
            table.selectRow(row)
        else:
            self._highlighted_domain_face = -1

    def _on_cad_domain_file_selection_changed(self) -> None:
        table = getattr(self, "_cad_domain_file_table", None)
        if table is None:
            return
        row = table.currentRow()
        if row < 0 or row >= len(self._cad_domain_imports):
            return
        self._highlighted_domain_face = row
        if hasattr(self, "_domain_face_table") and row < self._domain_face_table.rowCount():
            self._domain_face_table.blockSignals(True)
            self._domain_face_table.selectRow(row)
            self._domain_face_table.blockSignals(False)
        self._redraw_mesh_grid_vtk()

    def _remove_selected_cad_domain_file(self) -> None:
        table = getattr(self, "_cad_domain_file_table", None)
        if table is None:
            return
        row = table.currentRow()
        if row < 0 or row >= len(self._cad_domain_imports):
            self._set_status("请先在计算域 STL 表格中选中要删除的项。")
            return
        removed = self._cad_domain_imports.pop(row)
        self._rebuild_cad_domain_file_list()
        self._init_domain_face_table()
        if self._cad_domain_imports:
            self._auto_recommend_location_in_mesh(update_view=False)
        self._update_cad_domain_status()
        self._redraw_mesh_grid_vtk()
        self._save_mesh_workflow_state()
        self._set_status(f"已删除计算域 STL：{removed.source_path.name}")

    def _clear_cad_domain_files(self) -> None:
        if not self._cad_domain_imports:
            self._set_status("当前没有已导入的计算域 STL。")
            return
        reply = QMessageBox.question(
            self,
            "清空计算域 STL",
            "确定要清空所有已导入的计算域 STL 吗？边界配置表会同步清空。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        count = len(self._cad_domain_imports)
        self._cad_domain_imports.clear()
        self._rebuild_cad_domain_file_list()
        self._init_domain_face_table()
        self._update_cad_domain_status()
        self._redraw_mesh_grid_vtk()
        self._save_mesh_workflow_state()
        self._set_status(f"已清空 {count} 个计算域 STL。")

    def _cad_domain_patch_type(self, name: str) -> str:
        lowered = name.lower()
        if "patch" in lowered or "open" in lowered or "开放" in lowered:
            return "patch"
        if "inlet" in lowered or "入口" in lowered or "outlet" in lowered or "出口" in lowered:
            return "patch"
        if "sym" in lowered:
            return "symmetry"
        if "empty" in lowered:
            return "empty"
        return "wall"

    def _cad_domain_faces_from_assets(self) -> list[dict]:
        return [
            {"label": a.name, "name": a.name, "type": self._cad_domain_patch_type(a.name)}
            for a in self._cad_domain_imports
        ]

    def _cad_domain_transform_values(self) -> dict[str, float]:
        defaults = {
            "translate_x": 0.0, "translate_y": 0.0, "translate_z": 0.0,
            "rotate_x": 0.0, "rotate_y": 0.0, "rotate_z": 0.0,
            "scale_x": 1.0, "scale_y": 1.0, "scale_z": 1.0,
        }
        inputs = getattr(self, "_cad_domain_transform_inputs", {})
        for key, spin in inputs.items():
            defaults[key] = float(spin.value())
        return defaults

    def _cad_domain_transform(self) -> vtk.vtkTransform:
        vals = self._cad_domain_transform_values()
        tf = vtk.vtkTransform()
        tf.PostMultiply()
        tf.Scale(vals["scale_x"], vals["scale_y"], vals["scale_z"])
        tf.RotateX(vals["rotate_x"])
        tf.RotateY(vals["rotate_y"])
        tf.RotateZ(vals["rotate_z"])
        tf.Translate(vals["translate_x"], vals["translate_y"], vals["translate_z"])
        return tf

    def _transformed_cad_domain_polydata(self, asset: MeshImportAsset):
        flt = vtk.vtkTransformPolyDataFilter()
        flt.SetInputData(asset.polydata)
        flt.SetTransform(self._cad_domain_transform())
        flt.Update()
        out = vtk.vtkPolyData()
        out.DeepCopy(flt.GetOutput())
        return out

    def _cad_domain_has_duplicate_surfaces(self) -> bool:
        names: set[str] = set()
        for asset in self._cad_domain_imports:
            name = self._safe_openfoam_patch_name(asset.name)
            if not name:
                continue
            if name in names:
                self._show_error(
                    f"计算域 STL 边界名称重复：{name}\n"
                    "OpenFOAM patch 名称必须唯一。请在计算域表格中删除重复项，或重新导入。"
                )
                return True
            names.add(name)
        return False

    def _cad_domain_bounds(self) -> tuple[float, float, float, float, float, float] | None:
        if not self._cad_domain_imports:
            return None
        bbox = [float("inf"), float("-inf"), float("inf"), float("-inf"), float("inf"), float("-inf")]
        for asset in self._cad_domain_imports:
            pd = self._transformed_cad_domain_polydata(asset)
            bounds = pd.GetBounds()
            if not bounds or not all(np.isfinite(bounds)):
                continue
            for i in range(3):
                bbox[i * 2] = min(bbox[i * 2], bounds[i * 2])
                bbox[i * 2 + 1] = max(bbox[i * 2 + 1], bounds[i * 2 + 1])
        if not all(np.isfinite(bbox)):
            return None
        return tuple(float(v) for v in bbox)

    def _bbox_contains(self, outer, inner, eps: float = 1e-7) -> bool:
        return (
            inner[0] >= outer[0] - eps and inner[1] <= outer[1] + eps and
            inner[2] >= outer[2] - eps and inner[3] <= outer[3] + eps and
            inner[4] >= outer[4] - eps and inner[5] <= outer[5] + eps
        )

    def _update_cad_domain_status(self) -> None:
        if not hasattr(self, "_cad_domain_status_label"):
            return
        if not self._cad_domain_imports:
            self._cad_domain_status_label.setText("包围状态：未导入计算域")
            self._cad_domain_status_label.setStyleSheet("color: #d7ba7d;")
            return
        domain_bbox = self._cad_domain_bounds()
        obstacle_bbox = self._domain_geom_bbox()
        if domain_bbox is None:
            self._cad_domain_status_label.setText("包围状态：计算域几何无有效包围盒")
            self._cad_domain_status_label.setStyleSheet("color: #f48771;")
            return
        if obstacle_bbox is None:
            self._cad_domain_status_label.setText("包围状态：已导入计算域，尚未导入障碍物")
            self._cad_domain_status_label.setStyleSheet("color: #d7ba7d;")
            return
        if self._bbox_contains(domain_bbox, obstacle_bbox):
            self._cad_domain_status_label.setText("包围状态：已包围全部障碍物")
            self._cad_domain_status_label.setStyleSheet("color: #6a9955;")
        else:
            self._cad_domain_status_label.setText("包围状态：未完全包围障碍物，请调整位置或缩放")
            self._cad_domain_status_label.setStyleSheet("color: #f48771;")

    def _cad_location_values(self) -> tuple[float, float, float] | None:
        inputs = getattr(self, "_cad_location_inputs", {})
        if not inputs:
            return None
        return (float(inputs["x"].value()), float(inputs["y"].value()), float(inputs["z"].value()))

    def _set_cad_location_values(self, point: tuple[float, float, float], update_view: bool = True) -> None:
        inputs = getattr(self, "_cad_location_inputs", {})
        for axis, value in zip(("x", "y", "z"), point):
            spin = inputs.get(axis)
            if spin is not None:
                spin.blockSignals(True)
                spin.setValue(float(value))
                spin.blockSignals(False)
        if update_view:
            self._save_mesh_workflow_state()
            self._redraw_mesh_grid_vtk()

    def _point_inside_bbox(self, point, bbox, eps: float = 1e-9) -> bool:
        return (
            bbox[0] - eps <= point[0] <= bbox[1] + eps and
            bbox[2] - eps <= point[1] <= bbox[3] + eps and
            bbox[4] - eps <= point[2] <= bbox[5] + eps
        )

    def _auto_location_candidate(self) -> tuple[float, float, float] | None:
        domain_bbox = self._cad_domain_bounds()
        if domain_bbox is None:
            return None
        center = np.array([
            (domain_bbox[0] + domain_bbox[1]) * 0.5,
            (domain_bbox[2] + domain_bbox[3]) * 0.5,
            (domain_bbox[4] + domain_bbox[5]) * 0.5,
        ], dtype=float)
        obstacle_bbox = self._domain_geom_bbox()
        if obstacle_bbox is None or not self._point_inside_bbox(center, obstacle_bbox):
            return tuple(float(v) for v in center)
        size = np.array([domain_bbox[1] - domain_bbox[0], domain_bbox[3] - domain_bbox[2], domain_bbox[5] - domain_bbox[4]], dtype=float)
        axis = int(np.argmax(size))
        candidates = []
        for ratio in (0.25, 0.75, 0.15, 0.85):
            pt = center.copy()
            pt[axis] = domain_bbox[axis * 2] + size[axis] * ratio
            candidates.append(pt)
        for pt in candidates:
            if self._point_inside_bbox(pt, domain_bbox) and not self._point_inside_bbox(pt, obstacle_bbox):
                return tuple(float(v) for v in pt)
        return tuple(float(v) for v in center)

    def _auto_recommend_location_in_mesh(self, _checked: bool = False, update_view: bool = True) -> None:
        point = self._auto_location_candidate()
        if point is None:
            return
        self._set_cad_location_values(point, update_view=update_view)
        if update_view:
            self._set_status("locationInMesh 已自动推荐，红点已在 3D 预览中标出。")

    def _on_cad_location_changed(self, *_args) -> None:
        self._save_mesh_workflow_state()
        if hasattr(self, "_mesh_grid_vtk"):
            self._redraw_mesh_grid_vtk()

    def _on_cad_domain_transform_changed(self, *_args) -> None:
        self._update_cad_domain_status()
        self._save_mesh_workflow_state()
        if hasattr(self, "_mesh_grid_vtk"):
            self._redraw_mesh_grid_vtk()

    def _auto_wrap_cad_domain_around_obstacles(self) -> None:
        domain_bbox = self._cad_domain_bounds()
        obstacle_bbox = self._domain_geom_bbox()
        if domain_bbox is None or obstacle_bbox is None:
            self._show_error("请先导入计算域 STL 和障碍物 STL。")
            return
        d_size = np.array([domain_bbox[1] - domain_bbox[0], domain_bbox[3] - domain_bbox[2], domain_bbox[5] - domain_bbox[4]], dtype=float)
        o_size = np.array([obstacle_bbox[1] - obstacle_bbox[0], obstacle_bbox[3] - obstacle_bbox[2], obstacle_bbox[5] - obstacle_bbox[4]], dtype=float)
        ratios = np.divide(o_size * 1.25, np.maximum(d_size, 1e-9))
        factor = max(1.0, float(np.max(ratios)))
        d_center = np.array([(domain_bbox[0] + domain_bbox[1]) * 0.5, (domain_bbox[2] + domain_bbox[3]) * 0.5, (domain_bbox[4] + domain_bbox[5]) * 0.5])
        o_center = np.array([(obstacle_bbox[0] + obstacle_bbox[1]) * 0.5, (obstacle_bbox[2] + obstacle_bbox[3]) * 0.5, (obstacle_bbox[4] + obstacle_bbox[5]) * 0.5])
        inputs = getattr(self, "_cad_domain_transform_inputs", {})
        for axis_i, axis in enumerate(("x", "y", "z")):
            s_key = f"scale_{axis}"
            t_key = f"translate_{axis}"
            if s_key in inputs:
                inputs[s_key].setValue(inputs[s_key].value() * factor)
            if t_key in inputs:
                inputs[t_key].setValue(inputs[t_key].value() + float(o_center[axis_i] - d_center[axis_i]))
        self._update_cad_domain_status()
        self._redraw_mesh_grid_vtk()
        self._save_mesh_workflow_state()

    def _check_cad_domain_closure(self) -> None:
        if not self._cad_domain_imports:
            self._show_error("请先导入计算域 STL。")
            return
        msg = []
        for asset in self._cad_domain_imports:
            feature = vtk.vtkFeatureEdges()
            feature.SetInputData(asset.polydata)
            feature.BoundaryEdgesOn()
            feature.FeatureEdgesOff()
            feature.NonManifoldEdgesOn()
            feature.ManifoldEdgesOff()
            feature.Update()
            n_edges = feature.GetOutput().GetNumberOfCells()
            msg.append(f"{asset.source_path.name}: {'可能封闭' if n_edges == 0 else f'发现 {n_edges} 条开放/非流形边'}")
        QMessageBox.information(self, "计算域封闭性检查", "\n".join(msg))

    def _infer_domain_faces_from_filenames(self) -> None:
        if self._current_domain_template_key() == "cad":
            self._init_domain_face_table()
            self._set_status("已根据 CAD 计算域 STL 文件名重新识别边界类型。")
        else:
            self._init_domain_face_table()
            self._set_status("已恢复当前计算域模板的默认边界。")
        self._save_mesh_workflow_state()
        self._redraw_mesh_grid_vtk()

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
                canvas.add_polydata(
                    asset.polydata, color=(0.25, 0.74, 1.0), opacity=opacity,
                    edge_color=(0.96, 0.53, 0.12), line_width=1.0,
                )
            else:
                canvas.add_polydata(
                    asset.polydata, color=asset.color, opacity=opacity,
                )
        # ── draw computational domain (per selected template) ──
        self._draw_domain_preview(canvas)

        if saved is not None:
            camera.SetPosition(*saved[0])
            camera.SetFocalPoint(*saved[1])
            camera.SetViewUp(*saved[2])
            camera.SetViewAngle(saved[3])
            canvas.render()
        else:
            canvas.finish()

    # ── domain face / Group 2 logic ──────────────────────────

    def _init_domain_face_table(self) -> None:
        """Fill the dynamic face table for the current domain template."""
        if not hasattr(self, "_domain_face_table"):
            return
        table = self._domain_face_table
        previous_faces = []
        if table.rowCount() > 0:
            try:
                previous_faces = self._get_domain_face_definitions()
            except Exception:
                previous_faces = []
        previous_by_label = {f.get("label", ""): f for f in previous_faces}
        previous_by_name = {f.get("name", ""): f for f in previous_faces}
        table.blockSignals(True)
        table.setRowCount(0)
        for i, face in enumerate(self._current_template_faces()):
            saved = previous_by_label.get(face["label"]) or previous_by_name.get(face["name"]) or {}
            table.insertRow(i)
            table.setRowHeight(i, 34)
            show_check = QComboBox()
            show_check.addItem("✔", True)
            show_check.addItem("—", False)
            show_check.setMinimumHeight(28)
            show_check.setCurrentIndex(0 if saved.get("visible", True) else 1)
            show_check.currentIndexChanged.connect(self._on_domain_face_field_changed)
            table.setCellWidget(i, 0, show_check)

            label_item = QTableWidgetItem(face["label"])
            label_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(i, 1, label_item)

            combo = QComboBox()
            combo.addItems(BOUNDARY_TYPE_OPTIONS)
            combo.setMinimumHeight(28)
            btype = saved.get("type", face["type"])
            idx = combo.findText(btype)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.currentIndexChanged.connect(lambda _idx, row=i: self._on_domain_face_type_changed(row))
            table.setCellWidget(i, 2, combo)

            name_edit = QLineEdit()
            name_edit.setText(saved.get("name", face["name"]))
            name_edit.setPlaceholderText("patch 名称")
            name_edit.setMinimumHeight(28)
            name_edit.editingFinished.connect(lambda row=i: self._on_domain_face_name_changed(row))
            table.setCellWidget(i, 3, name_edit)

            u_default, p_default = self._default_domain_face_field_values(name_edit.text().strip(), btype)
            u_edit = QLineEdit(saved.get("u_value", u_default))
            u_edit.setPlaceholderText("如 (10 0 0)、noSlip、zeroGradient")
            u_edit.setMinimumHeight(28)
            u_edit.editingFinished.connect(self._on_domain_face_field_changed)
            table.setCellWidget(i, 4, u_edit)

            p_edit = QLineEdit(saved.get("p_value", p_default))
            p_edit.setPlaceholderText("如 0、zeroGradient")
            p_edit.setMinimumHeight(28)
            p_edit.editingFinished.connect(self._on_domain_face_field_changed)
            table.setCellWidget(i, 5, p_edit)

            level_spin = QSpinBox()
            level_spin.setRange(0, 9)
            level_spin.setValue(int(saved.get("refinement_level", 3 if btype == "wall" else 2)))
            level_spin.setMinimumHeight(28)
            level_spin.valueChanged.connect(self._on_domain_face_field_changed)
            table.setCellWidget(i, 6, level_spin)
        table.blockSignals(False)
        if table.rowCount() > 0:
            table.selectRow(0)
            self._highlighted_domain_face = 0
        else:
            self._highlighted_domain_face = -1
        try:
            table.selectionModel().selectionChanged.disconnect(
                self._on_domain_face_selection_changed)
        except (TypeError, RuntimeError):
            pass
        table.selectionModel().selectionChanged.connect(
            self._on_domain_face_selection_changed)

    def _on_domain_face_selection_changed(self) -> None:
        rows = set()
        for idx in self._domain_face_table.selectedIndexes():
            rows.add(idx.row())
        self._highlighted_domain_face = rows.pop() if rows else -1
        self._redraw_mesh_grid_vtk()

    def _get_domain_face_definitions(self) -> list[dict]:
        """Return {label, name, type, u_value, p_value, visible, refinement_level}."""
        template_faces = self._current_template_faces()
        if not hasattr(self, "_domain_face_table"):
            return [dict(f) for f in template_faces]
        result = []
        table = self._domain_face_table
        for i in range(table.rowCount()):
            label_item = table.item(i, 1)
            if label_item is None:
                continue
            label = label_item.text().strip()
            fallback = template_faces[i] if i < len(template_faces) else {"name": "patch", "type": "patch"}
            show_combo = table.cellWidget(i, 0)
            visible = bool(show_combo.currentData()) if show_combo else True
            combo = table.cellWidget(i, 2)
            btype = combo.currentText() if combo else fallback["type"]
            name_edit = table.cellWidget(i, 3)
            name = name_edit.text().strip() if name_edit else fallback["name"]
            u_edit = table.cellWidget(i, 4)
            p_edit = table.cellWidget(i, 5)
            level_spin = table.cellWidget(i, 6)
            default_u, default_p = self._default_domain_face_field_values(name, btype)
            u_value = u_edit.text().strip() if u_edit and u_edit.text().strip() else default_u
            p_value = p_edit.text().strip() if p_edit and p_edit.text().strip() else default_p
            result.append({
                "label": label,
                "name": name or btype,
                "type": btype,
                "u_value": u_value,
                "p_value": p_value,
                "visible": visible,
                "refinement_level": int(level_spin.value()) if level_spin else (3 if btype == "wall" else 2),
            })
        return result

    def _on_domain_face_type_changed(self, row: int) -> None:
        if not hasattr(self, "_domain_face_table"):
            return
        table = self._domain_face_table
        if row < 0 or row >= table.rowCount():
            return
        combo = table.cellWidget(row, 2)
        name_edit = table.cellWidget(row, 3)
        u_edit = table.cellWidget(row, 4)
        p_edit = table.cellWidget(row, 5)
        level_spin = table.cellWidget(row, 6)
        tfaces = self._current_template_faces()
        fb = tfaces[row] if row < len(tfaces) else {"type": "patch", "name": "patch"}
        btype = combo.currentText() if combo else fb["type"]
        name = name_edit.text().strip() if name_edit else fb["name"]
        u_value, p_value = self._default_domain_face_field_values(name, btype)
        if u_edit:
            u_edit.setText(u_value)
        if p_edit:
            p_edit.setText(p_value)
        if level_spin:
            level_spin.setValue(3 if btype == "wall" else 2)
        self._save_mesh_workflow_state()
        self._redraw_mesh_grid_vtk()

    def _on_domain_face_name_changed(self, row: int) -> None:
        if not hasattr(self, "_domain_face_table"):
            return
        table = self._domain_face_table
        if row < 0 or row >= table.rowCount():
            return
        combo = table.cellWidget(row, 2)
        name_edit = table.cellWidget(row, 3)
        u_edit = table.cellWidget(row, 4)
        p_edit = table.cellWidget(row, 5)
        tfaces = self._current_template_faces()
        fb = tfaces[row] if row < len(tfaces) else {"type": "patch", "name": "patch"}
        btype = combo.currentText() if combo else fb["type"]
        name = name_edit.text().strip() if name_edit else fb["name"]
        default_u, default_p = self._default_domain_face_field_values(name, btype)
        if u_edit and not u_edit.text().strip():
            u_edit.setText(default_u)
        if p_edit and not p_edit.text().strip():
            p_edit.setText(default_p)
        self._save_mesh_workflow_state()
        self._redraw_mesh_grid_vtk()

    def _on_domain_face_field_changed(self) -> None:
        self._save_mesh_workflow_state()
        self._redraw_mesh_grid_vtk()

    # ── domain template helpers ──────────────────────────────
    def _current_domain_template_key(self) -> str:
        return getattr(self, "_domain_template_key", "box")

    def _current_template_faces(self) -> list[dict]:
        if self._current_domain_template_key() == "cad":
            return self._cad_domain_faces_from_assets()
        return domain_templates.template_default_faces(self._current_domain_template_key())

    def _get_domain_unit_convert(self) -> float:
        if hasattr(self, "_domain_unit_combo") and self._domain_unit_combo.currentData() == "mm":
            return 0.001
        return 1.0

    def _get_cylinder_params(self) -> dict:
        return {
            "radius": self._cyl_radius.value(),
            "length": self._cyl_length.value(),
            "axis": self._cyl_axis_combo.currentData(),
            "center_x": self._cyl_center_x.value(),
            "center_y": self._cyl_center_y.value(),
            "center_z": self._cyl_center_z.value(),
            "n_circ": self._cyl_ncirc.value(),
            "n_radial": self._cyl_nradial.value(),
            "n_axial": self._cyl_naxial.value(),
            "unit": self._domain_unit_combo.currentData(),
        }

    def _get_nozzle_params(self) -> dict:
        return {
            "h_in": self._nozzle_hin.value(),
            "h_out": self._nozzle_hout.value(),
            "length": self._nozzle_length.value(),
            "width": self._nozzle_width.value(),
            "center_x": self._nozzle_center_x.value(),
            "center_y": self._nozzle_center_y.value(),
            "center_z": self._nozzle_center_z.value(),
            "cells_x": self._nozzle_cx.value(),
            "cells_y": self._nozzle_cy.value(),
            "cells_z": self._nozzle_cz.value(),
            "unit": self._domain_unit_combo.currentData(),
        }

    def _get_cad_domain_params(self) -> dict:
        bbox = self._cad_domain_bounds()
        if bbox is None:
            bbox = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        pad = max(bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4], 1.0) * 0.08
        cells = getattr(self, "_cad_domain_cells_inputs", {})
        return {
            "x_min": bbox[0] - pad,
            "x_max": bbox[1] + pad,
            "y_min": bbox[2] - pad,
            "y_max": bbox[3] + pad,
            "z_min": bbox[4] - pad,
            "z_max": bbox[5] + pad,
            "cells_x": cells.get("X").value() if cells.get("X") else 40,
            "cells_y": cells.get("Y").value() if cells.get("Y") else 24,
            "cells_z": cells.get("Z").value() if cells.get("Z") else 24,
            "unit": self._domain_unit_combo.currentData() if hasattr(self, "_domain_unit_combo") else "m",
            "transform": self._cad_domain_transform_values(),
        }

    def _get_domain_params(self) -> dict:
        key = self._current_domain_template_key()
        if key == "cylinder":
            return self._get_cylinder_params()
        if key == "nozzle":
            return self._get_nozzle_params()
        if key == "cad":
            return self._get_cad_domain_params()
        return self._get_domain_mesh_params()

    def _domain_param_page_index(self, key: str) -> int:
        return {"box": 0, "cylinder": 1, "nozzle": 2, "cad": 3}.get(key, 0)

    def _domain_bbox(self, key: str, params: dict) -> tuple[float, float, float, float, float, float]:
        if key == "cad":
            return (params["x_min"], params["x_max"], params["y_min"],
                    params["y_max"], params["z_min"], params["z_max"])
        if key == "cylinder":
            lines = domain_templates.cylinder_preview_polylines(params)
            pts = np.vstack(lines) if lines else np.array(domain_templates.cylinder_vertices(params), dtype=float)
            return (float(pts[:, 0].min()), float(pts[:, 0].max()),
                    float(pts[:, 1].min()), float(pts[:, 1].max()),
                    float(pts[:, 2].min()), float(pts[:, 2].max()))
        if key == "nozzle":
            pts = domain_templates.nozzle_corners(params)
            return (float(pts[:, 0].min()), float(pts[:, 0].max()),
                    float(pts[:, 1].min()), float(pts[:, 1].max()),
                    float(pts[:, 2].min()), float(pts[:, 2].max()))
        return (params["x_min"], params["x_max"], params["y_min"],
                params["y_max"], params["z_min"], params["z_max"])

    def _domain_location_in_mesh(self, key: str, params: dict) -> tuple[float, float, float]:
        if key == "cad":
            loc = self._cad_location_values()
            if loc is not None:
                return loc
            x0, x1, y0, y1, z0, z1 = self._domain_bbox(key, params)
            return ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0)
        if key in {"cylinder", "nozzle"}:
            x0, x1, y0, y1, z0, z1 = self._domain_bbox(key, params)
            return ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0)
        x0, x1, y0, y1, z0, z1 = self._domain_bbox(key, params)
        return (x1 - (x1 - x0) * 0.05, (y0 + y1) / 2.0, (z0 + z1) / 2.0)

    def _on_domain_param_changed(self, *_args) -> None:
        self._update_cad_domain_status()
        self._save_mesh_workflow_state()
        if hasattr(self, "_mesh_grid_vtk"):
            self._redraw_mesh_grid_vtk()

    def _on_domain_template_changed(self) -> None:
        if not hasattr(self, "_mesh_domain_type_combo"):
            return
        self._domain_template_key = self._mesh_domain_type_combo.currentData() or "box"
        if hasattr(self, "_domain_param_stack"):
            self._domain_param_stack.setCurrentIndex(
                self._domain_param_page_index(self._domain_template_key))
        self._init_domain_face_table()
        self._rebuild_cad_domain_file_list()
        self._update_cad_domain_status()
        self._save_mesh_workflow_state()
        self._redraw_mesh_grid_vtk()

    def _default_domain_face_field_values(self, name: str, boundary_type: str) -> tuple[str, str]:
        lowered = name.lower()
        if boundary_type == "empty":
            return "empty", "empty"
        if boundary_type == "symmetry":
            return "symmetry", "symmetry"
        if "inlet" in lowered or "入口" in lowered:
            return "(10 0 0)", "zeroGradient"
        if "outlet" in lowered or "出口" in lowered:
            return "zeroGradient", "0"
        if boundary_type == "wall" or "wall" in lowered or "壁" in lowered:
            return "noSlip", "zeroGradient"
        return "zeroGradient", "zeroGradient"

    def _parse_velocity_vector_value(self, value: str) -> np.ndarray | None:
        import re
        text = (value or "").strip()
        if not text or text in {"noSlip", "zeroGradient", "symmetry", "empty"}:
            return None
        numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
        if len(numbers) < 3:
            return None
        vector = np.array([float(numbers[0]), float(numbers[1]), float(numbers[2])], dtype=float)
        if float(np.linalg.norm(vector)) <= 1e-12:
            return None
        return vector

    def _draw_domain_face_velocity_arrows(self, canvas: NativeVtkPreviewWidget, face_corners: np.ndarray, face_def: dict) -> None:
        velocity = self._parse_velocity_vector_value(str(face_def.get("u_value", "")))
        if velocity is None:
            return
        origin = face_corners[0]
        u_axis = face_corners[1] - face_corners[0]
        v_axis = face_corners[3] - face_corners[0]
        face_size = max(float(np.linalg.norm(u_axis)), float(np.linalg.norm(v_axis)), 1e-9)
        normal = np.cross(u_axis, v_axis)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm > 1e-12:
            normal = normal / normal_norm
        else:
            normal = np.zeros(3, dtype=float)
        arrow_length = face_size * 0.16
        offset = normal * face_size * 0.015
        for u_ratio in (0.22, 0.5, 0.78):
            for v_ratio in (0.22, 0.5, 0.78):
                start = origin + u_axis * u_ratio + v_axis * v_ratio + offset
                canvas.add_arrow(start, velocity, arrow_length, color=(0.10, 0.45, 1.0), opacity=0.95)

    def _draw_cylinder_end_arrows(self, canvas: NativeVtkPreviewWidget, params: dict, at_outlet: bool, face_def: dict) -> None:
        velocity = self._parse_velocity_vector_value(str(face_def.get("u_value", "")))
        if velocity is None:
            return
        centers = domain_templates.cylinder_end_centers(params)
        center = np.array(centers[1 if at_outlet else 0], dtype=float)
        ring = domain_templates.cylinder_end_circle_points(params, at_outlet, n=12)
        radius = float(params["radius"])
        arrow_length = radius * 0.5
        axis_dir = domain_templates.cylinder_axis_direction(params)
        offset = axis_dir * radius * 0.02 * (-1.0 if at_outlet else 1.0)
        starts = [center]
        for cp in ring[:-1:2]:
            starts.append(center + 0.5 * (cp - center))
            starts.append(center + 0.85 * (cp - center))
        for start in starts:
            canvas.add_arrow(start + offset, velocity, arrow_length, color=(0.10, 0.45, 1.0), opacity=0.95)


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
        bbox = self._domain_geom_bbox()
        if bbox is None:
            return
        center = ((bbox[0] + bbox[1]) * 0.5, (bbox[2] + bbox[3]) * 0.5, (bbox[4] + bbox[5]) * 0.5)
        if self._current_domain_template_key() == "cylinder" and hasattr(self, "_cyl_radius"):
            dims = {"X": bbox[1] - bbox[0], "Y": bbox[3] - bbox[2], "Z": bbox[5] - bbox[4]}
            axis = max(dims, key=dims.get)
            others = [d for k, d in dims.items() if k != axis]
            self._cyl_length.setValue(max(dims[axis] * 1.25, 1e-4))
            self._cyl_radius.setValue(max((max(others) if others else 1.0) * 0.75, 1e-4))
            self._cyl_center_x.setValue(center[0])
            self._cyl_center_y.setValue(center[1])
            self._cyl_center_z.setValue(center[2])
            ai = self._cyl_axis_combo.findData(axis)
            if ai >= 0:
                self._cyl_axis_combo.setCurrentIndex(ai)
            if hasattr(self, "_mesh_grid_vtk"):
                self._redraw_mesh_grid_vtk()
            self._set_status("圆柱计算域已按障碍物中心自动包围（半径/长度/轴向/中心）。")
            return
        if self._current_domain_template_key() == "nozzle" and hasattr(self, "_nozzle_hin"):
            self._nozzle_length.setValue(max((bbox[1] - bbox[0]) * 1.35, 1e-4))
            self._nozzle_hin.setValue(max((bbox[3] - bbox[2]) * 1.6, 1e-4))
            self._nozzle_hout.setValue(max((bbox[3] - bbox[2]) * 1.6, 1e-4))
            self._nozzle_width.setValue(max((bbox[5] - bbox[4]) * 1.35, 1e-4))
            self._nozzle_center_x.setValue(center[0])
            self._nozzle_center_y.setValue(center[1])
            self._nozzle_center_z.setValue(center[2])
            if hasattr(self, "_mesh_grid_vtk"):
                self._redraw_mesh_grid_vtk()
            self._set_status("渐变通道计算域已按障碍物中心自动包围（尺寸/中心）。")
            return
        if self._domain_bounds_manual:
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
        self._set_status("计算域范围已从几何包围盒自动计算（含 10% 扩展边距）。")

    def _on_domain_manual_override(self) -> None:
        self._domain_bounds_manual = True
        self._redraw_mesh_grid_vtk()

    def _draw_domain_preview(self, canvas: NativeVtkPreviewWidget) -> None:
        key = self._current_domain_template_key()
        params = self._get_domain_params()
        if key == "cad":
            faces = self._get_domain_face_definitions()
            hi = self._highlighted_domain_face
            for i, asset in enumerate(self._cad_domain_imports):
                if i < len(faces) and not faces[i].get("visible", True):
                    continue
                pd = self._transformed_cad_domain_polydata(asset)
                btype = faces[i].get("type", self._cad_domain_patch_type(asset.name)) if i < len(faces) else self._cad_domain_patch_type(asset.name)
                lname = asset.name.lower()
                if "inlet" in lname or "入口" in lname:
                    color = (0.12, 0.42, 1.0)
                elif "outlet" in lname or "出口" in lname:
                    color = (1.0, 0.18, 0.12)
                elif btype == "wall":
                    color = (0.62, 0.66, 0.70)
                else:
                    color = (0.72, 0.48, 0.95)
                edge = (1.0, 0.85, 0.0) if i == hi else tuple(max(c * 0.72, 0.08) for c in color)
                opacity = 0.68 if i == hi else 0.42
                line_width = 1.7 if i == hi else 0.45
                canvas.add_polydata(pd, color=color, opacity=opacity, edge_color=edge, line_width=line_width)
                bounds = pd.GetBounds()
                if bounds and all(np.isfinite(bounds)):
                    label_pos = ((bounds[0]+bounds[1])*0.5, (bounds[2]+bounds[3])*0.5, (bounds[4]+bounds[5])*0.5)
                    canvas.add_text(asset.name, label_pos, color=color, size=12)
                    if i == hi and i < len(faces):
                        velocity = self._parse_velocity_vector_value(str(faces[i].get("u_value", "")))
                        if velocity is not None:
                            size = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4], 1e-9)
                            canvas.add_arrow(np.array(label_pos), velocity, size * 0.18, color=(0.10, 0.45, 1.0), opacity=0.95)
            loc = self._cad_location_values()
            if loc is not None:
                bbox = self._cad_domain_bounds() or (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
                marker_radius = max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4], 1.0) * 0.025
                canvas.add_sphere_marker(loc, marker_radius, color=(1.0, 0.05, 0.02), opacity=0.98)
                canvas.add_text("locationInMesh", loc, color=(1.0, 0.05, 0.02), size=13)
            self._update_cad_domain_status()
            return
        if key == "cylinder":
            highlight_wall = False
            faces = self._get_domain_face_definitions()
            hi = self._highlighted_domain_face
            face_def = faces[hi] if 0 <= hi < len(faces) else None
            highlight_label = face_def.get("label") if face_def else None
            highlight_wall = highlight_label == "圆柱壁面"
            for line in domain_templates.cylinder_preview_polylines(params):
                color = (1.0, 0.85, 0.0) if highlight_wall else (0.2, 0.8, 0.4)
                width = 3.6 if highlight_wall else 2.0
                canvas.add_polyline(line, color=color, width=width, opacity=0.9)
            inlet_c, outlet_c = domain_templates.cylinder_end_centers(params)
            canvas.add_text("inlet", tuple(inlet_c), color=(0.54, 0.82, 0.52), size=14)
            canvas.add_text("outlet", tuple(outlet_c), color=(0.96, 0.53, 0.44), size=14)
            if highlight_label in ("进口端面", "出口端面"):
                at_outlet = highlight_label == "出口端面"
                ring = domain_templates.cylinder_end_circle_points(params, at_outlet)
                canvas.add_polygon(ring, color=(0.96, 0.85, 0.16), opacity=0.42, edge_color=(1.0, 0.85, 0.0))
                self._draw_cylinder_end_arrows(canvas, params, at_outlet, face_def)
            return
        # box-topology (长方体 / 渐变通道) — wireframe + selected-face highlight
        corners = domain_templates.template_corners(key, params)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for a, b in edges:
            canvas.add_polyline(np.array([corners[a], corners[b]]), color=(0.2, 0.8, 0.4), width=2.0, opacity=0.7)
        faces = self._get_domain_face_definitions()
        if 0 <= self._highlighted_domain_face < len(faces):
            face_def = faces[self._highlighted_domain_face]
            quad = domain_templates._BOX_FACE_QUADS.get(face_def.get("label", ""))
            if quad:
                vi = [int(v) for v in quad.split()]
                face_corners = corners[vi]
                canvas.add_polygon(face_corners, color=(0.96, 0.85, 0.16), opacity=0.42, edge_color=(1.0, 0.85, 0.0))
                self._draw_domain_face_velocity_arrows(canvas, face_corners, face_def)

    def _write_multi_solid_stl(self, dest: Path, regions: list[tuple[str, object]]) -> None:
        """Write transformed CAD-domain patches as one ASCII STL with named solids."""
        def _fmt(v: float) -> str:
            return f"{float(v):.9g}"

        with dest.open("w", encoding="utf-8") as handle:
            for raw_name, polydata in regions:
                name = self._safe_openfoam_patch_name(str(raw_name)) or "patch"
                handle.write(f"solid {name}\n")
                for cell_i in range(polydata.GetNumberOfCells()):
                    cell = polydata.GetCell(cell_i)
                    if cell is None or cell.GetNumberOfPoints() < 3:
                        continue
                    p0 = np.array(cell.GetPoints().GetPoint(0), dtype=float)
                    for local_i in range(1, cell.GetNumberOfPoints() - 1):
                        p1 = np.array(cell.GetPoints().GetPoint(local_i), dtype=float)
                        p2 = np.array(cell.GetPoints().GetPoint(local_i + 1), dtype=float)
                        normal = np.cross(p1 - p0, p2 - p0)
                        norm = float(np.linalg.norm(normal))
                        if norm > 1e-14:
                            normal = normal / norm
                        else:
                            normal = np.array([0.0, 0.0, 0.0], dtype=float)
                        handle.write(f"  facet normal {_fmt(normal[0])} {_fmt(normal[1])} {_fmt(normal[2])}\n")
                        handle.write("    outer loop\n")
                        for point in (p0, p1, p2):
                            handle.write(f"      vertex {_fmt(point[0])} {_fmt(point[1])} {_fmt(point[2])}\n")
                        handle.write("    endloop\n")
                        handle.write("  endfacet\n")
                handle.write(f"endsolid {name}\n")


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
        if not self._mesh_imports and not self._cad_domain_imports:
            self._show_error("请先在组1中导入障碍物，或在组2中导入计算域几何体。")
            return
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        template_key = self._current_domain_template_key()
        if template_key == "cad" and not self._cad_domain_imports:
            self._show_error("当前选择 CAD 导入计算域，请先在组2中导入计算域 STL。")
            return
        if template_key == "cad" and self._cad_domain_has_duplicate_surfaces():
            return
        params = self._get_domain_params()
        convert = self._get_domain_unit_convert()
        case_dir = self._current_project.case_dir
        system_dir = case_dir / "system"
        system_dir.mkdir(parents=True, exist_ok=True)

        face_defs = self._get_domain_face_definitions()
        bm = domain_templates.build_block_mesh(template_key, params, face_defs, convert)
        (system_dir / "blockMeshDict").write_text(bm, encoding="utf-8")

        snappy = self._get_snappy_params()
        tri_dir = case_dir / "constant" / "triSurface"
        tri_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        stl_records: list[dict] = []
        for asset in self._mesh_imports:
            if not asset.source_path.exists():
                continue
            dest = tri_dir / asset.source_path.name
            shutil.copy2(asset.source_path, dest)
            stl_records.append({
                "file": dest.name,
                "name": Path(dest.name).stem,
                "type": "wall",
                "level": int(snappy["level"]),
            })

        if template_key == "cad":
            cad_regions: list[dict] = []
            cad_solids: list[tuple[str, object]] = []
            for i, asset in enumerate(self._cad_domain_imports):
                fdef = face_defs[i] if i < len(face_defs) else {
                    "name": asset.name,
                    "type": self._cad_domain_patch_type(asset.name),
                    "refinement_level": int(snappy["level"]),
                    "visible": True,
                }
                if not fdef.get("visible", True):
                    continue
                patch_name = self._safe_openfoam_patch_name(fdef.get("name") or asset.name) or asset.name
                pd = self._transformed_cad_domain_polydata(asset)
                cad_solids.append((patch_name, pd))
                cad_regions.append({
                    "name": patch_name,
                    "type": fdef.get("type") or "patch",
                    "level": int(fdef.get("refinement_level", snappy["level"])),
                })
            if cad_solids:
                dest = tri_dir / "cad_domain.stl"
                self._write_multi_solid_stl(dest, cad_solids)
                stl_records.append({
                    "file": dest.name,
                    "name": "cadDomain",
                    "type": "patch",
                    "level": int(snappy["level"]),
                    "regions": cad_regions,
                })

        # Keep one record per OpenFOAM geometry name so snappy sections stay consistent.
        deduped: dict[str, dict] = {}
        for rec in stl_records:
            deduped[rec["name"]] = rec
        stl_records = list(deduped.values())

        layers_enabled = bool(snappy["n_layers"] > 0 and template_key != "cad")
        shm = (
            "FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }\n"
            "castellatedMesh true;\n"
            "snap            true;\n"
            f"addLayers       {'true' if layers_enabled else 'false'};\n"
            "mergeTolerance 1e-6;\n\n"
            "geometry\n{\n"
        )
        for rec in stl_records:
            shm += f'    {rec["name"]}\n'
            shm += "    {\n"
            shm += "        type triSurfaceMesh;\n"
            shm += f'        file "{rec["file"]}";\n'
            if rec.get("regions"):
                shm += "        regions\n"
                shm += "        {\n"
                for region in rec["regions"]:
                    shm += f'            {region["name"]}\n'
                    shm += "            {\n"
                    shm += f'                name {region["name"]};\n'
                    shm += "            }\n"
                shm += "        }\n"
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
        for rec in stl_records:
            level = int(rec.get("level", snappy["level"]))
            ptype = rec.get("type", "patch")
            shm += f'        {rec["name"]}\n'
            shm += "        {\n"
            shm += f"            level ({level} {level});\n"
            if rec.get("regions"):
                shm += "            regions\n"
                shm += "            {\n"
                for region in rec["regions"]:
                    rlevel = int(region.get("level", level))
                    rtype = region.get("type", "patch")
                    shm += f'                {region["name"]}\n'
                    shm += "                {\n"
                    shm += f"                    level ({rlevel} {rlevel});\n"
                    shm += "                    patchInfo\n"
                    shm += "                    {\n"
                    shm += f"                        type {rtype};\n"
                    shm += "                    }\n"
                    shm += "                }\n"
                shm += "            }\n"
            else:
                shm += "            patchInfo\n"
                shm += "            {\n"
                shm += f"                type {ptype};\n"
                shm += "            }\n"
            shm += "        }\n"
        shm += "    }\n"
        lx, ly, lz = self._domain_location_in_mesh(template_key, params)
        shm += (
            f"    resolveFeatureAngle 30;\n"
            f"    locationInMesh ({lx:.6g} {ly:.6g} {lz:.6g});\n"
            f"    allowFreeStandingZoneFaces true;\n"
            f"}}\n\n"
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
        layer_records = [rec for rec in stl_records if str(rec.get("type", "")).lower() == "wall" and not rec.get("regions")]
        if layers_enabled and layer_records:
            shm += (
                "addLayersControls\n{\n"
                "    relativeSizes true;\n"
                "    layers\n    {\n"
            )
            for rec in layer_records:
                shm += f'        {rec["name"]}\n'
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
            ");\n"
        )
        (system_dir / "snappyHexMeshDict").write_text(shm, encoding="utf-8")
        self._write_zero_field_files(case_dir)
        self._append_log("已生成 blockMeshDict 和 snappyHexMeshDict，启动网格流水线。")
        self._run_mesh_pipeline_command()
        self._set_status("网格文件已生成，正在执行 blockMesh → snappyHexMesh -overwrite → checkMesh...")

    def _is_generated_time_dir(self, name: str) -> bool:
        if name == "0":
            return False
        try:
            float(name)
        except ValueError:
            return False
        return True

    def _clean_case_for_mesh_regeneration(self, case_dir: Path) -> None:
        root = case_dir.resolve()

        def remove_inside(path: Path) -> None:
            if not path.exists():
                return
            target = path.resolve()
            if target == root or root not in target.parents:
                raise OSError(f"拒绝删除 Case 外路径：{target}")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

        remove_inside(case_dir / "constant" / "polyMesh")
        for name in ("cellLevel", "pointLevel", "nSurfaceLayers", "thickness", "thicknessFraction"):
            remove_inside(case_dir / "0" / name)
            remove_inside(case_dir / "constant" / name)

        for child in case_dir.iterdir():
            if child.is_dir() and self._is_generated_time_dir(child.name):
                remove_inside(child)
            elif child.is_dir() and (child.name.startswith("processor") or child.name in {"VTK", "postProcessing"}):
                remove_inside(child)

        self._append_log("已清理旧网格、旧时间步和后处理缓存，准备重新生成网格。")

    def _run_mesh_pipeline_command(self) -> None:
        env_script = self._context.settings_service.load().openfoam_env_script or ""
        case_dir = self._current_project.case_dir
        try:
            self._clean_case_for_mesh_regeneration(case_dir)
        except Exception as error:
            self._append_log(f"清理旧网格/时间步失败：{error}")
            self._show_error(f"清理旧网格/时间步失败：{error}")
            return
        if env_script:
            cmd = (
                f'source "{env_script}" && '
                f"cd {shlex.quote(str(case_dir))} && "
                "blockMesh && snappyHexMesh -overwrite && checkMesh"
            )
        else:
            cmd = (
                f"cd {shlex.quote(str(case_dir))} && "
                "blockMesh && snappyHexMesh -overwrite && checkMesh"
            )
        self._active_process_kind = "meshPipeline"
        self._foam_process = QProcess(self)
        self._foam_process.setProgram("bash")
        self._foam_process.setArguments(["-lc", cmd])
        self._current_process_output = ""
        self._foam_process.readyReadStandardOutput.connect(self._read_process_stdout)
        self._foam_process.readyReadStandardError.connect(self._read_process_stderr)
        self._foam_process.errorOccurred.connect(
            lambda error: self._append_log(f"网格流水线进程启动/运行错误：{error}")
        )
        self._foam_process.finished.connect(
            lambda ec, es: self._on_mesh_pipeline_finished(ec, es))
        self._foam_process.start()
        self._append_log(f"执行网格流水线：blockMesh -> snappyHexMesh -overwrite -> checkMesh")
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
                    idx += 1
                    while idx < len(lines) and "}" not in lines[idx]:
                        idx += 1
                    idx += 1
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
            # split by '(' to separate count from vertex list
            if "(" in line:
                count_str, rest = line.split("(", 1)
                count_str = count_str.strip()
                if count_str.isdigit():
                    count = int(count_str)
                else:
                    continue
                id_str = rest.rstrip(")")
                ids = [int(x) for x in id_str.split() if x]
            else:
                parts = line.split()
                if not parts:
                    continue
                count = int(parts[0])
                ids = [int(x) for x in parts[1:]]
            if count <= 0:
                continue
            if len(ids) != count:
                count = len(ids)
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

    def _on_reset_all_params(self) -> None:
        self._domain_bounds_manual = False
        self._last_checkmesh_output = ""
        self._init_domain_face_table()
        self._rebuild_mesh_import_combo()
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
            "cad_domain_imports": [
                {"name": a.name, "source_path": str(a.source_path)}
                for a in getattr(self, "_cad_domain_imports", [])
            ],
            "cad_domain_transform": self._cad_domain_transform_values() if hasattr(self, "_cad_domain_transform_inputs") else {},
            "cad_location": self._cad_location_values() if hasattr(self, "_cad_location_inputs") else None,
            "domain_faces": [
                {
                    "label": f["label"],
                    "name": f["name"],
                    "type": f["type"],
                    "u_value": f.get("u_value", ""),
                    "p_value": f.get("p_value", ""),
                    "visible": f.get("visible", True),
                    "refinement_level": f.get("refinement_level", 2),
                }
                for f in self._get_domain_face_definitions()
            ],
            "domain": self._get_domain_mesh_params() if self._mesh_imports else {},
            "domain_template": self._current_domain_template_key(),
            "domain_params": self._get_domain_params() if hasattr(self, "_domain_unit_combo") else {},
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
        # restore domain template selection FIRST (face table depends on it)
        template_key = payload.get("domain_template", "box")
        if template_key not in domain_templates.DOMAIN_TEMPLATES:
            template_key = "box"
        self._domain_template_key = template_key
        if hasattr(self, "_mesh_domain_type_combo"):
            idx = self._mesh_domain_type_combo.findData(template_key)
            if idx >= 0:
                self._mesh_domain_type_combo.blockSignals(True)
                self._mesh_domain_type_combo.setCurrentIndex(idx)
                self._mesh_domain_type_combo.blockSignals(False)
        if hasattr(self, "_domain_param_stack"):
            self._domain_param_stack.setCurrentIndex(self._domain_param_page_index(template_key))
        dparams = payload.get("domain_params", {})
        if template_key == "cylinder" and dparams and hasattr(self, "_cyl_radius"):
            self._cyl_radius.setValue(dparams.get("radius", self._cyl_radius.value()))
            self._cyl_length.setValue(dparams.get("length", self._cyl_length.value()))
            self._cyl_center_x.setValue(dparams.get("center_x", self._cyl_center_x.value()))
            self._cyl_center_y.setValue(dparams.get("center_y", self._cyl_center_y.value()))
            self._cyl_center_z.setValue(dparams.get("center_z", self._cyl_center_z.value()))
            ai = self._cyl_axis_combo.findData(dparams.get("axis", "Z"))
            if ai >= 0:
                self._cyl_axis_combo.setCurrentIndex(ai)
            self._cyl_ncirc.setValue(int(dparams.get("n_circ", self._cyl_ncirc.value())))
            self._cyl_nradial.setValue(int(dparams.get("n_radial", self._cyl_nradial.value())))
            self._cyl_naxial.setValue(int(dparams.get("n_axial", self._cyl_naxial.value())))
        if template_key == "nozzle" and dparams and hasattr(self, "_nozzle_hin"):
            self._nozzle_hin.setValue(dparams.get("h_in", self._nozzle_hin.value()))
            self._nozzle_hout.setValue(dparams.get("h_out", self._nozzle_hout.value()))
            self._nozzle_length.setValue(dparams.get("length", self._nozzle_length.value()))
            self._nozzle_width.setValue(dparams.get("width", self._nozzle_width.value()))
            self._nozzle_center_x.setValue(dparams.get("center_x", self._nozzle_center_x.value()))
            self._nozzle_center_y.setValue(dparams.get("center_y", self._nozzle_center_y.value()))
            self._nozzle_center_z.setValue(dparams.get("center_z", self._nozzle_center_z.value()))
            self._nozzle_cx.setValue(int(dparams.get("cells_x", self._nozzle_cx.value())))
            self._nozzle_cy.setValue(int(dparams.get("cells_y", self._nozzle_cy.value())))
            self._nozzle_cz.setValue(int(dparams.get("cells_z", self._nozzle_cz.value())))
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
        self._cad_domain_imports = []
        for item in payload.get("cad_domain_imports", []):
            source_path = Path(item["source_path"])
            if not source_path.exists():
                continue
            asset = self._load_stl_asset(source_path, color=(0.28, 0.62, 1.0))
            if asset is not None:
                asset.name = item.get("name", source_path.stem)
                self._cad_domain_imports.append(asset)
        if hasattr(self, "_cad_domain_transform_inputs"):
            for key, value in payload.get("cad_domain_transform", {}).items():
                spin = self._cad_domain_transform_inputs.get(key)
                if spin is not None:
                    spin.blockSignals(True)
                    spin.setValue(float(value))
                    spin.blockSignals(False)
        self._rebuild_cad_domain_file_list()
        cad_location = payload.get("cad_location")
        if cad_location and len(cad_location) == 3:
            self._set_cad_location_values(tuple(cad_location), update_view=False)
        elif self._cad_domain_imports:
            self._auto_recommend_location_in_mesh(update_view=False)
        self._update_cad_domain_status()
        # restore domain faces (Group 2)
        domain_faces = payload.get("domain_faces", [])
        if domain_faces and hasattr(self, "_domain_face_table"):
            self._init_domain_face_table()
            table = self._domain_face_table
            for i, df in enumerate(domain_faces):
                if i >= table.rowCount():
                    break
                show_combo = table.cellWidget(i, 0)
                if show_combo and "visible" in df:
                    show_combo.setCurrentIndex(0 if df.get("visible", True) else 1)
                combo = table.cellWidget(i, 2)
                if combo:
                    idx = combo.findText(df.get("type", ""))
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                name_edit = table.cellWidget(i, 3)
                if name_edit:
                    name_edit.setText(df.get("name", ""))
                fallback_faces = self._current_template_faces()
                fallback = fallback_faces[i] if i < len(fallback_faces) else {"name": "patch", "type": "patch"}
                default_u, default_p = self._default_domain_face_field_values(
                    df.get("name", fallback["name"]),
                    df.get("type", fallback["type"]),
                )
                u_edit = table.cellWidget(i, 4)
                if u_edit:
                    u_edit.setText(df.get("u_value") or default_u)
                p_edit = table.cellWidget(i, 5)
                if p_edit:
                    p_edit.setText(df.get("p_value") or default_p)
                level_spin = table.cellWidget(i, 6)
                if level_spin and "refinement_level" in df:
                    level_spin.setValue(int(df.get("refinement_level", level_spin.value())))
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
