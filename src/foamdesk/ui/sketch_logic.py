"""Sketch drawing + extrude/revolve logic for the draw-geometry page (方案 A).

This mixin adds a lightweight 2D sketcher on top of the existing
NativeVtkPreviewWidget viewport:

- 选择草图平面 XY / YZ / XZ
- 画点 / 直线 / 圆 / 圆弧 / 矩形 / 多段线
- 闭合轮廓检测
- 草图拉伸 (vtkLinearExtrusionFilter)
- 旋转成型 (vtkRotationalExtrusionFilter)

方案 A 不含约束求解器；精确尺寸通过“编辑尺寸”对话框直接输入。
草图本身是临时的，拉伸/旋转后会烘焙成一个普通 GeometryObject（STL），
直接进入既有的“完成绘制 -> 网格生成”流程。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import vtk
from PySide6.QtWidgets import QInputDialog, QMessageBox


# Tool click requirements (number of clicks needed to commit one entity).
_SKETCH_TOOL_CLICKS = {
    "point": 1,
    "line": 2,
    "rect": 2,
    "circle": 2,
    "arc": 3,
    "polyline": -1,  # unbounded; right-click to finish
}

_SKETCH_TOOL_LABELS = {
    "select": "选择",
    "point": "点",
    "line": "直线",
    "rect": "矩形",
    "circle": "圆",
    "arc": "圆弧",
    "polyline": "多段线",
}

# Plane -> (u-axis, v-axis, normal) in world coordinates.
_SKETCH_PLANES = {
    "XY": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    "YZ": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])),
    "XZ": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
}

_SKETCH_LOOP_SAMPLES = 72


@dataclass
class SketchEntity:
    """One 2D sketch entity, stored in plane (u, v) coordinates."""

    kind: str  # point, line, rect, circle, arc, polyline
    points: list[tuple[float, float]] = field(default_factory=list)
    radius: float = 0.0
    start_angle: float = 0.0  # arc, radians
    end_angle: float = 0.0  # arc, radians


@dataclass
class Sketch:
    """A persistent named sketch that lives in the model tree."""

    name: str
    plane: str = "XY"
    origin: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    entities: list[SketchEntity] = field(default_factory=list)


class SketchLogicMixin:
    """Mixin providing 2D sketch + extrude/revolve for MainWindow."""

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def _init_sketch_state(self) -> None:
        self._sketch_active = False
        self._sketch_plane = "XY"
        self._sketch_tool = "select"
        self._sketch_entities: list[SketchEntity] = []
        self._sketch_pending: list[tuple[float, float]] = []
        self._sketch_cursor_uv: tuple[float, float] | None = None
        self._sketch_drag_start_uv: tuple[float, float] | None = None
        self._sketch_origin: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=float)
        self._sketch_actors: list[object] = []
        self._sketch_observer_tags: list[int] = []
        self._sketch_old_style = None
        self._sketches: list[Sketch] = []
        self._active_sketch_index: int = -1
        self._sketch_counter: int = 0
        self._load_sketches()

    # ------------------------------------------------------------------
    # persistence (sketches live with the project)
    # ------------------------------------------------------------------

    def _sketch_state_path(self) -> str:
        if getattr(self, "_current_project", None) is not None:
            return str(self._current_project.case_dir / "sketch_state.json")
        from pathlib import Path
        return str(Path(__file__).parent.parent.parent.parent / "config" / "sketch_state.json")

    def _serialize_sketch(self, sketch: Sketch) -> dict:
        return {
            "name": sketch.name,
            "plane": sketch.plane,
            "origin": list(sketch.origin),
            "entities": [
                {
                    "kind": e.kind,
                    "points": [list(p) for p in e.points],
                    "radius": e.radius,
                    "start_angle": e.start_angle,
                    "end_angle": e.end_angle,
                }
                for e in sketch.entities
            ],
        }

    def _deserialize_sketch(self, data: dict) -> Sketch:
        entities: list[SketchEntity] = []
        for e in data.get("entities", []):
            entities.append(
                SketchEntity(
                    kind=e.get("kind", "line"),
                    points=[tuple(p) for p in e.get("points", [])],
                    radius=float(e.get("radius", 0.0)),
                    start_angle=float(e.get("start_angle", 0.0)),
                    end_angle=float(e.get("end_angle", 0.0)),
                )
            )
        origin = data.get("origin", [0.0, 0.0, 0.0])
        if not isinstance(origin, list | tuple) or len(origin) != 3:
            origin = [0.0, 0.0, 0.0]
        return Sketch(
            name=data.get("name", "草图"),
            plane=data.get("plane", "XY"),
            origin=[float(origin[0]), float(origin[1]), float(origin[2])],
            entities=entities,
        )

    def _save_sketches(self) -> None:
        if getattr(self, "_suspend_draw_geometry_persist", False):
            return
        import json
        try:
            payload = {
                "counter": self._sketch_counter,
                "sketches": [self._serialize_sketch(s) for s in self._sketches],
            }
            with open(self._sketch_state_path(), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except (OSError, AttributeError):
            pass

    def _load_sketches(self) -> None:
        import json
        from pathlib import Path
        path = self._sketch_state_path()
        if not Path(path).exists():
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self._sketch_counter = int(payload.get("counter", 0))
        self._sketches = [self._deserialize_sketch(d) for d in payload.get("sketches", [])]

    # ------------------------------------------------------------------
    # plane / coordinate helpers
    # ------------------------------------------------------------------

    def _sketch_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u_axis, v_axis, normal = _SKETCH_PLANES[self._sketch_plane]
        origin = np.array(getattr(self, "_sketch_origin", np.array([0.0, 0.0, 0.0], dtype=float)), dtype=float)
        return origin, u_axis.astype(float), v_axis.astype(float), normal.astype(float)

    def _uv_to_world(self, u: float, v: float) -> np.ndarray:
        origin, u_axis, v_axis, _ = self._sketch_basis()
        return origin + u_axis * float(u) + v_axis * float(v)

    def _screen_to_sketch_uv(self, x: float, y: float) -> tuple[float, float] | None:
        renderer = self._modeling_viewport._renderer
        origin, u_axis, v_axis, normal = self._sketch_basis()

        renderer.SetDisplayPoint(float(x), float(y), 0.0)
        renderer.DisplayToWorld()
        near = np.array(renderer.GetWorldPoint(), dtype=float)
        if abs(near[3]) > 1e-12:
            near = near[:3] / near[3]
        else:
            near = near[:3]

        renderer.SetDisplayPoint(float(x), float(y), 1.0)
        renderer.DisplayToWorld()
        far = np.array(renderer.GetWorldPoint(), dtype=float)
        if abs(far[3]) > 1e-12:
            far = far[:3] / far[3]
        else:
            far = far[:3]

        ray = far - near
        denom = float(np.dot(ray, normal))
        if abs(denom) < 1e-12:
            return None
        t = float(np.dot(origin - near, normal)) / denom
        hit = near + t * ray
        return (float(np.dot(hit - origin, u_axis)), float(np.dot(hit - origin, v_axis)))

    # ------------------------------------------------------------------
    # enter / exit sketch mode
    # ------------------------------------------------------------------

    def _new_sketch(self) -> None:
        """Create a new persistent sketch, add it to the model tree, and start editing."""
        if not hasattr(self, "_modeling_viewport") or self._modeling_viewport is None:
            return
        if self._sketch_active:
            self._exit_sketch_mode()
        self._sketch_counter += 1
        plane = self._sketch_plane if self._sketch_plane in _SKETCH_PLANES else "XY"
        sketch = Sketch(name=f"草图{self._sketch_counter}", plane=plane)
        self._sketches.append(sketch)
        self._save_sketches()
        if hasattr(self, "_rebuild_tree"):
            self._rebuild_tree()
        self._enter_sketch_mode(len(self._sketches) - 1)
        self._set_modeling_status(f"已创建 {sketch.name}，开始绘制")

    def _enter_sketch_mode(self, index: int | None = None) -> None:
        if not hasattr(self, "_modeling_viewport") or self._modeling_viewport is None:
            return
        if index is None:
            # No explicit sketch: reuse the active one, or make a new sketch.
            if self._active_sketch_index < 0:
                self._new_sketch()
                return
            index = self._active_sketch_index
        if index < 0 or index >= len(self._sketches):
            return
        if self._sketch_active and self._active_sketch_index == index:
            return
        if self._sketch_active:
            self._exit_sketch_mode()

        self._active_sketch_index = index
        sketch = self._sketches[index]
        self._sketch_plane = sketch.plane if sketch.plane in _SKETCH_PLANES else "XY"
        self._sketch_entities = sketch.entities  # edit in place
        self._sketch_active = True
        self._sketch_tool = "select"
        self._sketch_pending = []
        self._sketch_cursor_uv = None
        self._sketch_drag_start_uv = None
        self._sketch_origin = np.array(sketch.origin, dtype=float)
        if hasattr(self, "_sketch_plane_combo"):
            self._sketch_plane_combo.blockSignals(True)
            self._sketch_plane_combo.setCurrentText(self._sketch_plane)
            self._sketch_plane_combo.blockSignals(False)
        self._orient_camera_to_plane()
        interactor = self._modeling_viewport._interactor
        if self._sketch_old_style is None:
            self._sketch_old_style = interactor.GetInteractorStyle()
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleUser())
        # Mouse tracking ON so we get MouseMove (rubber-band preview) without holding a button.
        self._modeling_viewport._vtk_widget.setMouseTracking(True)
        self._sketch_observer_tags = [
            interactor.AddObserver("LeftButtonPressEvent", self._on_sketch_press, 1.0),
            interactor.AddObserver("LeftButtonReleaseEvent", self._on_sketch_release, 1.0),
            interactor.AddObserver("MouseMoveEvent", self._on_sketch_move, 1.0),
            interactor.AddObserver("RightButtonPressEvent", self._on_sketch_right, 1.0),
        ]
        if hasattr(self, "_sketch_toolbar"):
            self._sketch_toolbar.setVisible(True)
        if hasattr(self, "_rebuild_tree"):
            self._rebuild_tree()  # refresh the ✎ active-sketch marker
        self._refresh_sketch_entity_combo()
        self._render_sketch()
        self._update_sketch_closed_label()
        self._set_modeling_status(f"编辑 {sketch.name}：平面 {self._sketch_plane}，选择左侧工具开始绘制")

    def _exit_sketch_mode(self) -> None:
        if not self._sketch_active:
            return
        # Persist the active sketch's plane + entities before leaving.
        if 0 <= self._active_sketch_index < len(self._sketches):
            self._sketches[self._active_sketch_index].plane = self._sketch_plane
            self._sketches[self._active_sketch_index].origin = [float(v) for v in self._sketch_origin]
        self._save_sketches()
        self._sketch_active = False
        self._sketch_pending = []
        self._sketch_cursor_uv = None
        self._sketch_drag_start_uv = None
        interactor = self._modeling_viewport._interactor
        for tag in self._sketch_observer_tags:
            interactor.RemoveObserver(tag)
        self._sketch_observer_tags = []
        if self._sketch_old_style is not None:
            interactor.SetInteractorStyle(self._sketch_old_style)
            self._sketch_old_style = None
        self._modeling_viewport._vtk_widget.setMouseTracking(False)
        renderer = self._modeling_viewport._renderer
        renderer.GetActiveCamera().ParallelProjectionOff()
        self._clear_sketch_actors()
        self._sketch_entities = []
        self._active_sketch_index = -1
        if hasattr(self, "_sketch_toolbar"):
            self._sketch_toolbar.setVisible(False)
        if hasattr(self, "_rebuild_tree"):
            self._rebuild_tree()
        self._modeling_viewport.render()
        self._set_modeling_status("已退出草图编辑")

    def _delete_sketch(self, index: int) -> None:
        if index < 0 or index >= len(self._sketches):
            return
        was_active = self._sketch_active and self._active_sketch_index == index
        if was_active:
            self._exit_sketch_mode()
        removed = self._sketches.pop(index)
        if self._active_sketch_index > index:
            self._active_sketch_index -= 1
        elif self._active_sketch_index == index:
            self._active_sketch_index = -1
            self._sketch_entities = []
        self._save_sketches()
        if hasattr(self, "_refresh_sketch_entity_combo"):
            self._refresh_sketch_entity_combo()
        if hasattr(self, "_rebuild_tree"):
            self._rebuild_tree()
        if hasattr(self, "_modeling_viewport") and self._modeling_viewport is not None:
            self._modeling_viewport.render()
        self._set_modeling_status(f"已删除 {removed.name}")

    def _orient_camera_to_plane(self) -> None:
        renderer = self._modeling_viewport._renderer
        camera = renderer.GetActiveCamera()
        origin, _u, v_axis, normal = self._sketch_basis()
        bounds = renderer.ComputeVisiblePropBounds()
        valid_bounds = bool(bounds) and all(np.isfinite(bounds)) and bounds[1] >= bounds[0] and bounds[3] >= bounds[2] and bounds[5] >= bounds[4]
        diagonal = 0.0
        if valid_bounds:
            diagonal = float(
                np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
            )
        distance = max(diagonal, 3.0)
        camera.ParallelProjectionOn()
        camera.SetFocalPoint(*origin)
        camera.SetPosition(*(origin + normal * distance))
        camera.SetViewUp(*v_axis)
        camera.SetParallelScale(max(diagonal * 0.65, 1.5))
        renderer.ResetCameraClippingRange()
        self._modeling_viewport.render()

    def _set_sketch_plane(self, plane: str) -> None:
        if plane not in _SKETCH_PLANES:
            return
        self._sketch_plane = plane
        self._sketch_pending = []
        if 0 <= self._active_sketch_index < len(self._sketches):
            self._sketches[self._active_sketch_index].plane = plane
            self._save_sketches()
        if self._sketch_active:
            self._orient_camera_to_plane()
            self._render_sketch()
        self._set_modeling_status(f"草图平面：{plane}")

    def _set_sketch_tool(self, tool: str) -> None:
        self._sketch_tool = tool
        self._sketch_pending = []
        self._sketch_cursor_uv = None
        self._sketch_drag_start_uv = None
        self._render_sketch()
        if tool == "select":
            self._set_modeling_status("选择：按住草图平面拖动，可移动整张草图")
            return
        if tool == "polyline":
            self._set_modeling_status("多段线：依次左键点击顶点，右键结束")
        else:
            self._set_modeling_status(f"绘制工具：{_SKETCH_TOOL_LABELS.get(tool, tool)}")

    # ------------------------------------------------------------------
    # mouse handlers
    # ------------------------------------------------------------------

    def _on_sketch_press(self, _obj, _event) -> None:
        if not self._sketch_active:
            return
        x, y = self._modeling_viewport._interactor.GetEventPosition()
        uv = self._screen_to_sketch_uv(x, y)
        if uv is None:
            return
        if self._sketch_tool == "select":
            self._sketch_drag_start_uv = uv
            self._set_modeling_status("正在移动草图平面")
            return
        self._sketch_pending.append(uv)
        needed = _SKETCH_TOOL_CLICKS.get(self._sketch_tool, 2)
        if needed > 0 and len(self._sketch_pending) >= needed:
            self._commit_pending_entity()
        else:
            self._render_sketch()

    def _on_sketch_move(self, _obj, _event) -> None:
        if not self._sketch_active:
            return
        x, y = self._modeling_viewport._interactor.GetEventPosition()
        uv = self._screen_to_sketch_uv(x, y)
        if uv is None:
            return
        if self._sketch_tool == "select":
            if self._sketch_drag_start_uv is None:
                return
            du = float(uv[0] - self._sketch_drag_start_uv[0])
            dv = float(uv[1] - self._sketch_drag_start_uv[1])
            if abs(du) < 1e-9 and abs(dv) < 1e-9:
                return
            _origin, u_axis, v_axis, _normal = self._sketch_basis()
            self._sketch_origin = self._sketch_origin + u_axis * du + v_axis * dv
            if 0 <= self._active_sketch_index < len(self._sketches):
                self._sketches[self._active_sketch_index].origin = [float(v) for v in self._sketch_origin]
            self._sketch_drag_start_uv = uv
            self._save_sketches()
            self._orient_camera_to_plane()
            self._render_sketch()
            return
        if not self._sketch_pending:
            return
        self._sketch_cursor_uv = uv
        self._render_sketch()

    def _on_sketch_release(self, _obj, _event) -> None:
        if not self._sketch_active:
            return
        if self._sketch_tool == "select" and self._sketch_drag_start_uv is not None:
            self._sketch_drag_start_uv = None
            self._save_sketches()
            self._set_modeling_status("草图平面位置已更新")

    def _on_sketch_right(self, _obj, _event) -> None:
        if not self._sketch_active:
            return
        if self._sketch_tool == "select":
            self._sketch_drag_start_uv = None
            self._render_sketch()
            return
        if self._sketch_tool == "polyline" and len(self._sketch_pending) >= 2:
            self._commit_pending_entity(force=True)
        else:
            self._sketch_pending = []
            self._sketch_cursor_uv = None
            self._render_sketch()

    # ------------------------------------------------------------------
    # entity commit / edit / delete
    # ------------------------------------------------------------------

    def _commit_pending_entity(self, force: bool = False) -> None:
        tool = self._sketch_tool
        pts = list(self._sketch_pending)
        entity: SketchEntity | None = None
        if tool == "point" and len(pts) >= 1:
            entity = SketchEntity(kind="point", points=[pts[0]])
        elif tool == "line" and len(pts) >= 2:
            entity = SketchEntity(kind="line", points=[pts[0], pts[1]])
        elif tool == "rect" and len(pts) >= 2:
            entity = SketchEntity(kind="rect", points=[pts[0], pts[1]])
        elif tool == "circle" and len(pts) >= 2:
            center = pts[0]
            radius = float(np.hypot(pts[1][0] - center[0], pts[1][1] - center[1]))
            entity = SketchEntity(kind="circle", points=[center], radius=radius)
        elif tool == "arc" and len(pts) >= 3:
            center, start, end = pts[0], pts[1], pts[2]
            radius = float(np.hypot(start[0] - center[0], start[1] - center[1]))
            a0 = float(np.arctan2(start[1] - center[1], start[0] - center[0]))
            a1 = float(np.arctan2(end[1] - center[1], end[0] - center[0]))
            entity = SketchEntity(kind="arc", points=[center], radius=radius, start_angle=a0, end_angle=a1)
        elif tool == "polyline" and force and len(pts) >= 2:
            entity = SketchEntity(kind="polyline", points=pts)

        if entity is not None:
            self._sketch_entities.append(entity)
            self._sketch_pending = []
            self._sketch_cursor_uv = None
            self._save_sketches()
            self._refresh_sketch_entity_combo()
            self._render_sketch()
            self._update_sketch_closed_label()
            self._set_modeling_status(f"已添加草图：{_SKETCH_TOOL_LABELS.get(entity.kind, entity.kind)}")

    def _selected_sketch_entity_index(self) -> int:
        if not hasattr(self, "_sketch_entity_combo"):
            return -1
        return int(self._sketch_entity_combo.currentIndex())

    def _delete_selected_sketch_entity(self) -> None:
        index = self._selected_sketch_entity_index()
        if index < 0 or index >= len(self._sketch_entities):
            return
        removed = self._sketch_entities.pop(index)
        self._save_sketches()
        self._refresh_sketch_entity_combo()
        self._render_sketch()
        self._update_sketch_closed_label()
        self._set_modeling_status(f"已删除草图：{_SKETCH_TOOL_LABELS.get(removed.kind, removed.kind)}")

    def _clear_sketch_entities(self) -> None:
        self._sketch_entities.clear()  # keep the same list object the Sketch references
        self._sketch_pending = []
        self._save_sketches()
        self._refresh_sketch_entity_combo()
        self._render_sketch()
        self._update_sketch_closed_label()
        self._set_modeling_status("已清空草图")

    def _edit_selected_sketch_dimension(self) -> None:
        index = self._selected_sketch_entity_index()
        if index < 0 or index >= len(self._sketch_entities):
            self._set_modeling_status("请先在草图列表中选择一个图元")
            return
        entity = self._sketch_entities[index]
        if entity.kind == "circle":
            value, ok = QInputDialog.getDouble(self, "编辑半径", "半径", entity.radius, 0.0001, 1e6, 4)
            if ok:
                entity.radius = float(value)
        elif entity.kind == "rect":
            (u0, v0), (u1, v1) = entity.points
            width, ok = QInputDialog.getDouble(self, "编辑矩形", "宽 (沿 U)", abs(u1 - u0), 0.0001, 1e6, 4)
            if not ok:
                return
            height, ok = QInputDialog.getDouble(self, "编辑矩形", "高 (沿 V)", abs(v1 - v0), 0.0001, 1e6, 4)
            if not ok:
                return
            su = 1.0 if u1 >= u0 else -1.0
            sv = 1.0 if v1 >= v0 else -1.0
            entity.points = [(u0, v0), (u0 + su * width, v0 + sv * height)]
        elif entity.kind == "line":
            (u0, v0), (u1, v1) = entity.points
            current_len = float(np.hypot(u1 - u0, v1 - v0))
            length, ok = QInputDialog.getDouble(self, "编辑直线", "长度", current_len, 0.0001, 1e6, 4)
            if not ok or current_len < 1e-9:
                return
            scale = length / current_len
            entity.points = [(u0, v0), (u0 + (u1 - u0) * scale, v0 + (v1 - v0) * scale)]
        else:
            self._set_modeling_status("该图元暂不支持数值编辑（仅圆/矩形/直线）")
            return
        self._save_sketches()
        self._render_sketch()
        self._update_sketch_closed_label()
        self._set_modeling_status("已更新草图尺寸")

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

    def _clear_sketch_actors(self) -> None:
        renderer = self._modeling_viewport._renderer
        for actor in self._sketch_actors:
            renderer.RemoveActor(actor)
        self._sketch_actors = []

    def _make_sketch_line_actor(self, world_points: np.ndarray, color, width: float) -> object:
        vtk_points = vtk.vtkPoints()
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(world_points))
        for index, point in enumerate(world_points):
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
        return actor

    def _make_sketch_point_actor(self, world_points: np.ndarray, color, size: float) -> object:
        vtk_points = vtk.vtkPoints()
        for point in world_points:
            vtk_points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)
        glyph = vtk.vtkVertexGlyphFilter()
        glyph.SetInputData(poly_data)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetPointSize(size)
        return actor

    def _entity_uv_polyline(self, entity: SketchEntity) -> list[tuple[float, float]]:
        """Sample one entity into an ordered list of (u, v) points (for drawing)."""
        if entity.kind in ("line", "polyline"):
            return list(entity.points)
        if entity.kind == "rect":
            (u0, v0), (u1, v1) = entity.points
            return [(u0, v0), (u1, v0), (u1, v1), (u0, v1), (u0, v0)]
        if entity.kind == "circle":
            cu, cv = entity.points[0]
            angles = np.linspace(0.0, 2.0 * np.pi, _SKETCH_LOOP_SAMPLES, endpoint=True)
            return [(cu + entity.radius * np.cos(a), cv + entity.radius * np.sin(a)) for a in angles]
        if entity.kind == "arc":
            cu, cv = entity.points[0]
            a0, a1 = entity.start_angle, entity.end_angle
            if a1 < a0:
                a1 += 2.0 * np.pi
            samples = max(8, int(_SKETCH_LOOP_SAMPLES * (a1 - a0) / (2.0 * np.pi)))
            angles = np.linspace(a0, a1, samples, endpoint=True)
            return [(cu + entity.radius * np.cos(a), cv + entity.radius * np.sin(a)) for a in angles]
        return list(entity.points)

    def _uv_list_to_world(self, uv_points: list[tuple[float, float]]) -> np.ndarray:
        return np.array([self._uv_to_world(u, v) for (u, v) in uv_points], dtype=float)

    def _add_sketch_reference_grid(self) -> None:
        renderer = self._modeling_viewport._renderer
        grid_points = []
        span = 2.0
        steps = 8
        for i in range(-steps, steps + 1):
            t = span * i / steps
            grid_points.append([self._uv_to_world(-span, t), self._uv_to_world(span, t)])
            grid_points.append([self._uv_to_world(t, -span), self._uv_to_world(t, span)])
        vtk_points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        for segment in grid_points:
            line = vtk.vtkLine()
            start_id = vtk_points.InsertNextPoint(*segment[0])
            end_id = vtk_points.InsertNextPoint(*segment[1])
            line.GetPointIds().SetId(0, start_id)
            line.GetPointIds().SetId(1, end_id)
            cells.InsertNextCell(line)
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)
        poly_data.SetLines(cells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.72, 0.72, 0.72)
        actor.GetProperty().SetOpacity(0.35)
        actor.GetProperty().SetLineWidth(1.0)
        renderer.AddActor(actor)
        self._sketch_actors.append(actor)

    def _render_sketch(self) -> None:
        if not self._sketch_active:
            return
        self._clear_sketch_actors()
        renderer = self._modeling_viewport._renderer
        self._add_sketch_reference_grid()
        sketch_color = (0.95, 0.55, 0.15)
        point_color = (1.0, 0.9, 0.2)

        for entity in self._sketch_entities:
            if entity.kind == "point":
                world = self._uv_list_to_world(entity.points)
                actor = self._make_sketch_point_actor(world, point_color, 9.0)
            else:
                uv_poly = self._entity_uv_polyline(entity)
                world = self._uv_list_to_world(uv_poly)
                if len(world) < 2:
                    continue
                actor = self._make_sketch_line_actor(world, sketch_color, 2.2)
            renderer.AddActor(actor)
            self._sketch_actors.append(actor)

        self._render_sketch_preview()
        self._modeling_viewport.render()

    def _render_sketch_preview(self) -> None:
        if not self._sketch_pending:
            return
        renderer = self._modeling_viewport._renderer
        preview_color = (0.45, 0.85, 1.0)
        pending = list(self._sketch_pending)
        cursor = self._sketch_cursor_uv
        tool = self._sketch_tool

        # Confirmed pending vertices as dots.
        world_pts = self._uv_list_to_world(pending)
        dot_actor = self._make_sketch_point_actor(world_pts, preview_color, 8.0)
        renderer.AddActor(dot_actor)
        self._sketch_actors.append(dot_actor)

        if cursor is None:
            return
        preview = SketchEntity(kind="line")
        if tool in ("line", "polyline"):
            preview = SketchEntity(kind="polyline", points=pending + [cursor])
        elif tool == "rect":
            preview = SketchEntity(kind="rect", points=[pending[0], cursor])
        elif tool == "circle":
            radius = float(np.hypot(cursor[0] - pending[0][0], cursor[1] - pending[0][1]))
            preview = SketchEntity(kind="circle", points=[pending[0]], radius=radius)
        elif tool == "arc" and len(pending) >= 2:
            center, start = pending[0], pending[1]
            radius = float(np.hypot(start[0] - center[0], start[1] - center[1]))
            a0 = float(np.arctan2(start[1] - center[1], start[0] - center[0]))
            a1 = float(np.arctan2(cursor[1] - center[1], cursor[0] - center[0]))
            preview = SketchEntity(kind="arc", points=[center], radius=radius, start_angle=a0, end_angle=a1)
        else:
            return
        uv_poly = self._entity_uv_polyline(preview)
        world = self._uv_list_to_world(uv_poly)
        if len(world) >= 2:
            actor = self._make_sketch_line_actor(world, preview_color, 1.6)
            renderer.AddActor(actor)
            self._sketch_actors.append(actor)

    # ------------------------------------------------------------------
    # closed contour detection
    # ------------------------------------------------------------------

    def _sketch_closed_loop_uv(self) -> list[tuple[float, float]] | None:
        """Return an ordered closed loop of (u, v) points, or None if no closed contour."""
        # Highest priority: a single closed primitive.
        for entity in self._sketch_entities:
            if entity.kind == "circle":
                loop = self._entity_uv_polyline(entity)
                return loop[:-1] if loop else None
            if entity.kind == "rect":
                loop = self._entity_uv_polyline(entity)
                return loop[:-1] if loop else None

        # A self-closing polyline.
        for entity in self._sketch_entities:
            if entity.kind == "polyline" and len(entity.points) >= 3:
                first = np.array(entity.points[0])
                last = np.array(entity.points[-1])
                if float(np.linalg.norm(first - last)) < self._sketch_loop_tol():
                    return list(entity.points[:-1])

        # Chain line / arc / open-polyline segments into a loop.
        segments: list[list[tuple[float, float]]] = []
        for entity in self._sketch_entities:
            if entity.kind == "line":
                segments.append([entity.points[0], entity.points[1]])
            elif entity.kind == "arc":
                segments.append(self._entity_uv_polyline(entity))
            elif entity.kind == "polyline":
                segments.append(list(entity.points))
        return self._chain_segments(segments)

    def _sketch_loop_tol(self) -> float:
        spans = []
        for entity in self._sketch_entities:
            for (u, v) in entity.points:
                spans.append(abs(u))
                spans.append(abs(v))
            if entity.radius:
                spans.append(entity.radius)
        scale = max(spans) if spans else 1.0
        return max(scale * 0.02, 1e-4)

    def _chain_segments(self, segments: list[list[tuple[float, float]]]) -> list[tuple[float, float]] | None:
        if len(segments) < 3:
            return None
        tol = self._sketch_loop_tol()
        remaining = [list(seg) for seg in segments]
        loop = remaining.pop(0)

        def close(a, b) -> bool:
            return float(np.linalg.norm(np.array(a) - np.array(b))) < tol

        progress = True
        while remaining and progress:
            progress = False
            for index, seg in enumerate(remaining):
                if close(loop[-1], seg[0]):
                    loop.extend(seg[1:])
                elif close(loop[-1], seg[-1]):
                    loop.extend(list(reversed(seg))[1:])
                elif close(loop[0], seg[-1]):
                    loop = seg[:-1] + loop
                elif close(loop[0], seg[0]):
                    loop = list(reversed(seg))[:-1] + loop
                else:
                    continue
                remaining.pop(index)
                progress = True
                break
        if not remaining and close(loop[0], loop[-1]):
            return loop[:-1]
        return None

    def _update_sketch_closed_label(self) -> None:
        if not hasattr(self, "_sketch_closed_label"):
            return
        loop = self._sketch_closed_loop_uv() if self._sketch_entities else None
        if loop:
            self._sketch_closed_label.setText("闭合轮廓：✅ 已检测到")
            self._sketch_closed_label.setStyleSheet("color: #89d185; font-weight: 600;")
        else:
            self._sketch_closed_label.setText("闭合轮廓：❌ 未闭合")
            self._sketch_closed_label.setStyleSheet("color: #d7ba7d;")

    # ------------------------------------------------------------------
    # extrude / revolve -> bake into a GeometryObject
    # ------------------------------------------------------------------

    def _loop_polygon_polydata(self, loop_uv: list[tuple[float, float]]) -> vtk.vtkPolyData:
        """Build a polydata containing the closed loop as a single polygon (for capping)."""
        world = self._uv_list_to_world(loop_uv)
        points = vtk.vtkPoints()
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(world))
        for index, point in enumerate(world):
            points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
            polygon.GetPointIds().SetId(index, index)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polygon)
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(cells)
        return poly_data

    def _finalize_solid(self, poly_data: vtk.vtkPolyData) -> vtk.vtkPolyData:
        triangle = vtk.vtkTriangleFilter()
        triangle.SetInputData(poly_data)
        triangle.Update()
        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(triangle.GetOutput())
        clean.Update()
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(clean.GetOutput())
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.SplittingOff()
        normals.Update()
        output = vtk.vtkPolyData()
        output.DeepCopy(normals.GetOutput())
        return output

    def _extrude_sketch(self) -> None:
        loop_uv = self._sketch_closed_loop_uv()
        if not loop_uv or len(loop_uv) < 3:
            QMessageBox.warning(self, "拉伸", "没有检测到闭合轮廓，无法拉伸。\n请先画一个闭合的圆/矩形/多边形。")
            return
        length, ok = QInputDialog.getDouble(self, "草图拉伸", "拉伸长度（沿平面法线）", 1.0, -1e6, 1e6, 4)
        if not ok or abs(length) < 1e-9:
            return
        _origin, _u, _v, normal = self._sketch_basis()
        polygon_poly = self._loop_polygon_polydata(loop_uv)
        extrude = vtk.vtkLinearExtrusionFilter()
        extrude.SetInputData(polygon_poly)
        extrude.SetExtrusionTypeToVectorExtrusion()
        extrude.SetVector(float(normal[0]), float(normal[1]), float(normal[2]))
        extrude.SetScaleFactor(float(length))
        extrude.CappingOn()
        extrude.Update()
        solid = self._finalize_solid(extrude.GetOutput())
        if solid.GetNumberOfPoints() == 0:
            QMessageBox.warning(self, "拉伸", "拉伸结果为空，请检查草图。")
            return
        self._bake_sketch_solid(solid, "拉伸体")

    def _revolve_sketch(self) -> None:
        loop_uv = self._sketch_closed_loop_uv()
        profile_uv: list[tuple[float, float]]
        if loop_uv and len(loop_uv) >= 3:
            profile_uv = loop_uv + [loop_uv[0]]
        else:
            # fall back to a single open profile (polyline / line / arc)
            profile_uv = self._single_open_profile_uv()
            if profile_uv is None or len(profile_uv) < 2:
                QMessageBox.warning(self, "旋转成型", "没有可用于旋转的轮廓。\n请画一个闭合轮廓，或一条多段线/直线作为截面。")
                return

        axis_name, ok = QInputDialog.getItem(
            self, "旋转成型", "旋转轴（全局轴，经过原点）", ["X", "Y", "Z"], 2, False
        )
        if not ok:
            return
        angle, ok = QInputDialog.getDouble(self, "旋转成型", "旋转角度 (°)", 360.0, 1.0, 360.0, 2)
        if not ok:
            return

        axis_map = {"X": np.array([1.0, 0.0, 0.0]), "Y": np.array([0.0, 1.0, 0.0]), "Z": np.array([0.0, 0.0, 1.0])}
        axis = axis_map[axis_name]
        world = self._uv_list_to_world(profile_uv)

        # Map chosen axis -> global Z so vtkRotationalExtrusionFilter (rotates about Z) works.
        rot = self._axis_to_z_matrix(axis)
        local = world @ rot.T

        points = vtk.vtkPoints()
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(local))
        for index, point in enumerate(local):
            points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
            polyline.GetPointIds().SetId(index, index)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyline)
        profile = vtk.vtkPolyData()
        profile.SetPoints(points)
        profile.SetLines(cells)

        revolve = vtk.vtkRotationalExtrusionFilter()
        revolve.SetInputData(profile)
        revolve.SetResolution(max(12, int(angle / 6) + 12))
        revolve.SetAngle(float(angle))
        revolve.CappingOn()
        revolve.Update()

        revolved = revolve.GetOutput()
        if revolved.GetNumberOfPoints() == 0:
            QMessageBox.warning(self, "旋转成型", "旋转结果为空，请确认截面不与旋转轴重合。")
            return
        # Map back from Z-frame to world.
        self._transform_polydata_points(revolved, rot)
        solid = self._finalize_solid(revolved)
        self._bake_sketch_solid(solid, "旋转体")

    def _single_open_profile_uv(self) -> list[tuple[float, float]] | None:
        for entity in self._sketch_entities:
            if entity.kind in ("polyline", "line", "arc"):
                return self._entity_uv_polyline(entity)
        return None

    def _axis_to_z_matrix(self, axis: np.ndarray) -> np.ndarray:
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        z = np.array([0.0, 0.0, 1.0])
        if float(np.dot(axis, z)) > 0.9999:
            return np.eye(3)
        if float(np.dot(axis, z)) < -0.9999:
            return np.diag([1.0, -1.0, -1.0])
        # Rotation that maps `axis` onto z (Rodrigues).
        v = np.cross(axis, z)
        s = float(np.linalg.norm(v))
        c = float(np.dot(axis, z))
        vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))

    def _transform_polydata_points(self, poly_data: vtk.vtkPolyData, rot_to_z: np.ndarray) -> None:
        """Map points from the Z-aligned frame back to world (inverse of rot_to_z)."""
        inverse = rot_to_z.T
        points = poly_data.GetPoints()
        for index in range(points.GetNumberOfPoints()):
            local = np.array(points.GetPoint(index), dtype=float)
            world = inverse @ local
            points.SetPoint(index, float(world[0]), float(world[1]), float(world[2]))
        points.Modified()

    def _bake_sketch_solid(self, poly_data: vtk.vtkPolyData, label: str) -> None:
        from foamdesk.ui.draw_geometry_tab import GeometryObject

        self._modeling_counter[label] = self._modeling_counter.get(label, 0) + 1
        name = f"{label}{self._modeling_counter[label]}"
        actor, wire_actor, point_actor = self._create_modeling_actor_bundle(poly_data)
        obj = GeometryObject(
            name=name,
            kind="stl",
            actor=actor,
            source=poly_data,
            wire_actor=wire_actor,
            point_actor=point_actor,
            section=self._modeling_active_section,
        )
        self._modeling_objects.append(obj)
        self._add_modeling_object_actors(obj)
        self._rebuild_tree()
        self._select_object(len(self._modeling_objects) - 1)
        self._modeling_viewport.render()
        self._save_modeling_state_if_ready()
        self._set_modeling_status(f"已生成 {name}，可在模型树中查看")

    # ------------------------------------------------------------------
    # entity combo refresh (UI helper)
    # ------------------------------------------------------------------

    def _refresh_sketch_entity_combo(self) -> None:
        if not hasattr(self, "_sketch_entity_combo"):
            return
        combo = self._sketch_entity_combo
        combo.blockSignals(True)
        combo.clear()
        for index, entity in enumerate(self._sketch_entities):
            combo.addItem(f"{index + 1}. {_SKETCH_TOOL_LABELS.get(entity.kind, entity.kind)}")
        combo.blockSignals(False)
