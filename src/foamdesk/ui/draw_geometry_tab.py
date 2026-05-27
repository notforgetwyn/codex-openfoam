from __future__ import annotations

from dataclasses import dataclass, field

import vtk
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QMenu,
    QTreeWidgetItem,
)


@dataclass
class GeometryObject:
    name: str
    kind: str  # cube, sphere, cylinder, cone
    actor: object  # vtkActor
    source: object  # vtk source
    visible: bool = True
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


_PRIMITIVE_FACTORIES = {
    "cube": (_create_cube_source, "立方体"),
    "sphere": (_create_sphere_source, "球体"),
    "cylinder": (_create_cylinder_source, "圆柱"),
    "cone": (_create_cone_source, "圆锥"),
}


class DrawGeometryLogicMixin:
    """Mixin providing all modeling logic methods for MainWindow."""

    def _init_modeling_state(self) -> None:
        self._modeling_objects: list[GeometryObject] = []
        self._modeling_selected_index: int = -1
        self._modeling_counter: dict[str, int] = {}

    def _setup_modeling_axes(self) -> None:
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1.5, 1.5, 1.5)
        axes.GetXAxisCaptionActor2D().SetCaption("X")
        axes.GetYAxisCaptionActor2D().SetCaption("Y")
        axes.GetZAxisCaptionActor2D().SetCaption("Z")
        self._modeling_axes_actor = axes
        self._modeling_viewport._renderer.AddActor(axes)
        self._modeling_viewport.render()

    # ------------------------------------------------------------------
    # primitive management
    # ------------------------------------------------------------------

    def _add_primitive(self, kind: str) -> None:
        factory, label = _PRIMITIVE_FACTORIES[kind]
        self._modeling_counter[kind] = self._modeling_counter.get(kind, 0) + 1
        name = f"{label}{self._modeling_counter[kind]}"

        source = factory()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.25, 0.74, 1.0)
        actor.GetProperty().SetInterpolationToPhong()

        obj = GeometryObject(name=name, kind=kind, actor=actor, source=source)
        self._modeling_objects.append(obj)
        self._modeling_viewport._renderer.AddActor(actor)

        self._rebuild_tree()
        self._select_object(len(self._modeling_objects) - 1)
        self._modeling_viewport.render()
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
        self._set_modeling_status(f"已删除 {obj.name}")

    # ------------------------------------------------------------------
    # tree
    # ------------------------------------------------------------------

    def _rebuild_tree(self) -> None:
        tree = self._modeling_tree
        tree.blockSignals(True)
        tree.clear()
        for i, obj in enumerate(self._modeling_objects):
            item = QTreeWidgetItem([obj.name, obj.kind])
            item.setData(0, Qt.ItemDataRole.UserRole, i)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked if obj.visible else Qt.CheckState.Unchecked)
            if i == self._modeling_selected_index:
                tree.setCurrentItem(item)
            tree.addTopLevelItem(item)
        tree.blockSignals(False)

    def _on_tree_item_clicked(self, item: QTreeWidgetItem) -> None:
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        if idx is not None:
            self._select_object(int(idx))

    def _on_tree_item_changed(self, item: QTreeWidgetItem) -> None:
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        i = int(idx)
        if i < 0 or i >= len(self._modeling_objects):
            return
        visible = item.checkState(0) == Qt.CheckState.Checked
        self._modeling_objects[i].visible = visible
        self._modeling_objects[i].actor.SetVisibility(visible)
        self._modeling_viewport.render()

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
        self._block_prop_signals(False)

    def _on_prop_name_changed(self) -> None:
        if self._modeling_selected_index < 0:
            return
        new_name = self._modeling_prop_name.text().strip()
        if new_name:
            self._modeling_objects[self._modeling_selected_index].name = new_name
            self._rebuild_tree()

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
        ):
            w.setEnabled(enabled)

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
