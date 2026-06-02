from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer

from foamdesk.ui.visualization_widgets import NativeVtkViewerDialog


class ResultsLogicMixin:
    def _refresh_result_field_panel(self, show_errors: bool = True) -> None:
        if not hasattr(self, "_result_field_combo"):
            return
        self._result_time_combo.clear()
        if self._current_project is None:
            if show_errors:
                self._show_error("请先新建或打开项目。")
            return
        if not self._result_mesh_available():
            self._result_field_combo.blockSignals(True)
            self._result_field_combo.clear()
            self._result_field_combo.addItems(self.RESULT_FIELDS)
            self._result_field_combo.blockSignals(False)
            self._refresh_result_display_modes()
            self._result_minmax_label.setText("最大/最小值：未生成网格")
            self._set_status("当前 case 还没有生成网格，请先在网格生成页面点击生成网格。")
            return
        try:
            case_info = self._context.openfoam_vtk_service.inspect(self._current_project)
        except (OSError, RuntimeError) as error:
            if show_errors:
                self._show_error(f"刷新结果场失败：{error}")
            return

        available = set(case_info.point_arrays) | set(case_info.cell_arrays)
        supported_fields = [name for name in self.RESULT_FIELDS if name in available]
        if not supported_fields:
            supported_fields = list(self.RESULT_FIELDS)
        current_field = self._result_field_combo.currentText().strip()
        self._result_field_combo.blockSignals(True)
        self._result_field_combo.clear()
        self._result_field_combo.addItems(supported_fields)
        if current_field in supported_fields:
            self._result_field_combo.setCurrentText(current_field)
        elif "U" in supported_fields:
            self._result_field_combo.setCurrentText("U")
        elif supported_fields:
            self._result_field_combo.setCurrentText(supported_fields[0])
        self._result_field_combo.blockSignals(False)
        self._refresh_result_display_modes()

        times = [f"{time:g}" for time in case_info.time_values]
        self._result_time_combo.addItems(times)
        if times:
            self._result_time_combo.setCurrentText(times[-1])
        self._update_result_field_metadata()
        self._append_log(
            "结果场已刷新："
            f"fields={sorted(available)}, supported={self.RESULT_FIELDS}, times={times or ['默认']}"
        )
        self._set_status("结果场已刷新。")

    def _on_result_field_changed(self) -> None:
        self._refresh_result_display_modes()
        self._update_result_field_metadata()

    def _refresh_result_display_modes(self) -> None:
        if not hasattr(self, "_result_field_combo") or not hasattr(self, "_result_display_combo"):
            return
        field_name = self._result_field_combo.currentText().strip()
        allowed_modes = self.RESULT_FIELD_DISPLAY_MODES.get(field_name, self.RESULT_DISPLAY_MODES)
        current_mode = self._result_display_combo.currentText().strip()
        self._result_display_combo.blockSignals(True)
        self._result_display_combo.clear()
        self._result_display_combo.addItems(allowed_modes)
        if current_mode in allowed_modes:
            self._result_display_combo.setCurrentText(current_mode)
        elif allowed_modes:
            self._result_display_combo.setCurrentIndex(0)
        self._result_display_combo.blockSignals(False)

    def _update_result_field_metadata(self) -> None:
        if not hasattr(self, "_result_field_combo"):
            return
        field_name = self._result_field_combo.currentText().strip()
        self._result_unit_label.setText(f"单位：{self.RESULT_FIELD_UNITS.get(field_name, '-')}")
        if self._current_project is None:
            self._result_minmax_label.setText("最大/最小值：未选择项目")
            return
        try:
            _output, field_array, display_name, _selected_time, storage = self._load_result_field_data()
        except (OSError, RuntimeError, ValueError) as error:
            self._result_minmax_label.setText(f"最大/最小值：暂不可用（{error}）")
            return
        values = self._scalar_values(field_array)
        if values.size == 0:
            self._result_minmax_label.setText("最大/最小值：字段为空")
            return
        min_value = float(values.min())
        max_value = float(values.max())
        self._result_minmax_label.setText(
            f"最大/最小值：{display_name} min={min_value:.6g}, max={max_value:.6g}（{storage} 字段）"
        )

    def _load_selected_result_display(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        if not self._result_mesh_available():
            self._show_error("当前 case 还没有生成网格，请先在“网格生成”页面点击“生成网格”。")
            return
        mode = self._result_display_combo.currentText().strip()
        field_name = self._result_field_combo.currentText().strip()
        if mode not in self.RESULT_FIELD_DISPLAY_MODES.get(field_name, []):
            self._show_error(f"{field_name} 不支持 {mode}。请重新选择显示方式。")
            self._refresh_result_display_modes()
            return
        try:
            output, field_array, display_name, selected_time, storage = self._load_result_field_data()
        except (OSError, RuntimeError, ValueError) as error:
            self._show_error(f"加载结果显示失败：{error}")
            return

        color_range = self._selected_result_color_range(field_array)
        self._render_result_display(output, field_array, display_name, selected_time, storage, mode, color_range)
        if mode != "流线 streamlines":
            self._configure_result_animation_source()

    def _compute_inlet_outlet_positions(self):
        """Read constant/polyMesh/{boundary,faces,points} to find
        inlet/outlet patch face centroids. No VTK dependency.
        Returns (inlet_centers: list, outlet_centers: list) or (None, None)."""
        if self._current_project is None:
            return None, None
        case_dir = self._current_project.case_dir
        boundary_path = case_dir / "constant" / "polyMesh" / "boundary"
        faces_path = case_dir / "constant" / "polyMesh" / "faces"
        points_path = case_dir / "constant" / "polyMesh" / "points"
        if not (boundary_path.exists() and faces_path.exists() and points_path.exists()):
            return None, None

        # ── helper: skip FoamFile header and return content after the first '(' ──
        def _skip_foam_header(text: str) -> str:
            idx = text.find("(")
            if idx < 0:
                return text
            return text[idx + 1:]

        # ── 1. parse boundary → {name: (nFaces, startFace)} ──
        b_content = _skip_foam_header(boundary_path.read_text(encoding="utf-8"))
        patch_info: dict[str, tuple[int, int]] = {}
        for pm in re.finditer(r"(\S+)\s*\n\s*\{([^}]+)\}", b_content):
            name = pm.group(1).strip()
            body = pm.group(2)
            if name.isdigit() or name in ("(", ")"):
                continue
            nf_m = re.search(r"nFaces\s+(\d+)", body)
            sf_m = re.search(r"startFace\s+(\d+)", body)
            if nf_m and sf_m:
                patch_info[name] = (int(nf_m.group(1)), int(sf_m.group(1)))

        # ── 2. identify inlet/outlet by name ──
        inlet_names = [n for n in patch_info if "inlet" in n.lower()]
        outlet_names = [n for n in patch_info if "outlet" in n.lower()]
        if not inlet_names and not outlet_names:
            return None, None

        # ── 3. read points ──
        pt_content = _skip_foam_header(points_path.read_text(encoding="utf-8"))
        points = []
        for m in re.finditer(r"\(([^)]+)\)", pt_content):
            parts = m.group(1).split()
            if len(parts) >= 3:
                points.append((float(parts[0]), float(parts[1]), float(parts[2])))
        if not points:
            return None, None

        # ── 4. read faces (4-index format like "4(0 1 5 4)") ──
        f_content = _skip_foam_header(faces_path.read_text(encoding="utf-8"))
        faces = []  # list of (vertex_indices,)
        for m in re.finditer(r"(\d+)\s*\(([^)]+)\)", f_content):
            count = int(m.group(1))
            ids = [int(x) for x in m.group(2).split() if x]
            if len(ids) >= 3 and len(ids) == count:
                faces.append(ids)

        # ── 5. compute patch centroid from its face vertices ──
        def _patch_centroid(patch_name: str):
            info = patch_info.get(patch_name)
            if info is None:
                return None
            nf, sf = info
            sx = sy = sz = 0.0
            n_verts = 0
            for fi in range(sf, min(sf + nf, len(faces))):
                for vi in faces[fi]:
                    if vi < len(points):
                        px, py, pz = points[vi]
                        sx += px; sy += py; sz += pz
                        n_verts += 1
            if n_verts == 0:
                return None
            return (sx / n_verts, sy / n_verts, sz / n_verts)

        inlet_centers = []
        for n in inlet_names:
            c = _patch_centroid(n)
            if c:
                inlet_centers.append(c)
        outlet_centers = []
        for n in outlet_names:
            c = _patch_centroid(n)
            if c:
                outlet_centers.append(c)
        return inlet_centers, outlet_centers

    def _render_result_display(
        self,
        output,
        field_array,
        display_name: str,
        selected_time: str,
        storage: str,
        mode: str,
        color_range: tuple[float, float],
    ) -> None:
        # ensure viewer exists before passing inlet/outlet positions
        self._ensure_native_vtk_viewer()
        io_positions = self._compute_inlet_outlet_positions()
        if io_positions[0] or io_positions[1]:
            self._native_vtk_viewer.set_inlet_outlet_positions(*io_positions)
        self._native_vtk_viewer.set_domain_boundary_polydata(
            self._build_domain_boundary_patch_surface(time_value=self._selected_result_time_value())
        )

        if mode in {"速度云图", "压力云图", "温度云图"}:
            try:
                geometry_surface = self._build_geometry_patch_surface(
                    time_value=self._selected_result_time_value()
                )
                self._native_vtk_viewer.set_geometry_polydata(geometry_surface)
                output, field_array = self._ensure_point_field(output, field_array, storage)
            except RuntimeError as error:
                self._show_error(str(error))
                return
            self._native_vtk_viewer.plot_surface(output, field_array, color_range, display_name)
            face_count = output.GetNumberOfPolys()
            self._finish_result_visualization(
                f"Surface 表面云图已加载到原生 VTK 3D 窗口：field={display_name}, time={selected_time}, faces={face_count}, range={color_range}"
            )
            return
        if mode in {"速度切片", "压力切片", "温度切面", "速度等值线", "压力等值线", "温度等值线"}:
            try:
                mb = self._context.openfoam_vtk_service.build_case_output(
                    self._current_project,
                    time_value=self._selected_result_time_value(),
                )
                vol_data = mb.GetBlock(0) if mb.GetNumberOfBlocks() > 0 else mb
                field_name = self._result_field_combo.currentText().strip()
                arr, display_name, _storage = self._prepare_result_field(vol_data, field_name)
                if arr is None:
                    self._show_error(f"切片需要字段 {field_name}")
                    return
            except (OSError, RuntimeError, ValueError) as error:
                self._show_error(str(error))
                return
            if mode in {"速度等值线", "压力等值线", "温度等值线"}:
                resolved_axis, center = self._native_vtk_viewer.plot_contour_slice(
                    vol_data,
                    arr,
                    color_range,
                    display_name,
                    None,
                    0.5,
                )
                self._finish_result_visualization(
                    f"Contour 等值线已加载：field={display_name}, time={selected_time}, axis={resolved_axis}, center={center:.6g}, range={color_range}"
                )
            else:
                resolved_axis, center = self._native_vtk_viewer.plot_slice(
                    vol_data,
                    arr,
                    color_range,
                    display_name,
                    None,
                    0.5,
                )
                self._finish_result_visualization(
                    f"Slice 切片已加载：field={display_name}, time={selected_time}, axis={resolved_axis}, center={center:.6g}, range={color_range}"
                )
            return
        if mode == "流线 streamlines":
            try:
                streamline_output, vtu_grid, seed_label, seed_count, speed_range = self._build_streamlines_from_vtu("inlet")
                stl_streamline_output, _stl_grid, stl_seed_label, stl_seed_count, stl_speed_range = self._build_streamlines_from_vtu("geometry")
            except (OSError, RuntimeError, ValueError) as error:
                self._show_error(f"生成流线失败：{error}")
                return
            self._native_vtk_viewer.plot_streamlines(
                vtu_grid,
                streamline_output,
                speed_range,
                inlet_label=seed_label,
                stl_streamline_data=(stl_streamline_output, stl_speed_range, stl_seed_label),
            )
            line_count = streamline_output.GetNumberOfLines()
            point_count = streamline_output.GetNumberOfPoints()
            self._finish_result_visualization(
                f"Streamline 流线：入口面={seed_label}, seeds={seed_count}, lines={line_count}, points={point_count}, speedRange={speed_range}; STL近壁={stl_seed_label}, seeds={stl_seed_count}"
            )
            return
        if mode.startswith("Volume"):
            try:
                mb = self._context.openfoam_vtk_service.build_case_output(
                    self._current_project,
                    time_value=self._selected_result_time_value(),
                )
                volume_output = mb.GetBlock(0) if mb.GetNumberOfBlocks() > 0 else mb
                field_name = self._result_field_combo.currentText().strip()
                vol_field, display_name, _storage = self._prepare_result_field(volume_output, field_name)
                if vol_field is None:
                    self._show_error(f"体渲染需要字段 {field_name}")
                    return
            except (OSError, RuntimeError, ValueError) as error:
                self._show_error(str(error))
                return
            self._native_vtk_viewer.plot_volume(volume_output, vol_field, color_range, display_name)
            self._finish_result_visualization(
                f"Volume 体渲染已加载：field={display_name}, time={selected_time}, range={color_range}"
            )
            return

        self._show_error(f"{mode} 已放入界面结构，但 VTK 专项实现还未接入。")

    def _ensure_point_field(self, output, field_array, storage: str):
        if storage == "point":
            return output, field_array
        array_name = field_array.GetName()
        converter = vtk.vtkCellDataToPointData()
        converter.SetInputData(output)
        converter.PassCellDataOn()
        converter.Update()
        converted_output = converter.GetOutput()
        converted_array = converted_output.GetPointData().GetArray(array_name)
        if converted_array is None:
            raise RuntimeError(f"{array_name} 当前是单元字段，转换为点字段失败。")
        return converted_output, converted_array

    def _configure_result_animation_source(self) -> None:
        if not hasattr(self, "_result_time_combo"):
            return
        if self._native_vtk_viewer is None:
            return
        frame_count = self._result_time_combo.count()
        if frame_count <= 1:
            self._native_vtk_viewer.set_animation_source(0, None)
            return

        def render_frame(frame_index: int) -> None:
            if frame_index < 0 or frame_index >= self._result_time_combo.count():
                return
            self._result_time_combo.setCurrentIndex(frame_index)
            mode = self._result_display_combo.currentText().strip()
            try:
                output, field_array, display_name, selected_time, storage = self._load_result_field_data()
            except (OSError, RuntimeError, ValueError) as error:
                self._show_error(f"播放动画失败：{error}")
                if self._native_vtk_viewer is not None:
                    self._native_vtk_viewer.pause_animation()
                return
            color_range = self._selected_result_color_range(field_array)
            self._render_result_display(output, field_array, display_name, selected_time, storage, mode, color_range)

        self._native_vtk_viewer.set_animation_source(frame_count, render_frame)

    def _load_result_field_data(self):
        if self._current_project is None:
            raise RuntimeError("未选择项目")
        if not self._result_mesh_available():
            raise RuntimeError("当前 case 还没有生成网格，请先在“网格生成”页面点击“生成网格”。")
        field_name = self._result_field_combo.currentText().strip()
        selected_time = self._result_time_combo.currentText().strip() or "默认"
        time_value = self._selected_result_time_value()
        geometry = self._context.openfoam_vtk_service.build_geometry_filter(
            self._current_project,
            time_value=time_value,
        )
        output = geometry.GetOutput()
        field_array, display_name, storage = self._prepare_result_field(output, field_name)
        if field_array is None:
            raise RuntimeError(f"当前 Case 没有字段 {field_name}")
        return output, field_array, display_name, selected_time, storage

    def _result_mesh_available(self) -> bool:
        if self._current_project is None:
            return False
        mesh_dir = self._current_project.case_dir / "constant" / "polyMesh"
        return (mesh_dir / "points").exists()

    def _prepare_result_field(self, data_object, field_name: str):
        display_map = {
            "p": "压力 p",
            "U": "|U|",
            "T": "温度 T",
        }
        return self._lookup_result_array(data_object, field_name, display_map.get(field_name, field_name))

    def _lookup_result_array(self, data_object, array_name: str, display_name: str):
        array = data_object.GetPointData().GetArray(array_name)
        storage = "point"
        if array is None:
            array = data_object.GetCellData().GetArray(array_name)
            storage = "cell"
        if array is None:
            return None, display_name, storage
        if array.GetNumberOfComponents() > 1 and display_name == array_name:
            display_name = f"|{array_name}|"
        return array, display_name, storage

    def _selected_result_time_value(self) -> float | None:
        if not hasattr(self, "_result_time_combo"):
            return None
        text = self._result_time_combo.currentText().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _selected_result_color_range(self, field_array) -> tuple[float, float]:
        values = self._scalar_values(field_array)
        if values.size == 0:
            return (0.0, 1.0)
        vmin, vmax = float(values.min()), float(values.max())
        if vmax <= vmin:
            return (vmin - 1.0, vmin + 1.0)
        return (vmin, vmax)

    def _scalar_values(self, field_array) -> np.ndarray:
        values = vtk_to_numpy(field_array)
        if values.ndim == 1:
            return values.astype(float)
        if values.ndim == 2 and values.shape[1] > 1:
            return np.linalg.norm(values, axis=1).astype(float)
        return values.reshape(-1).astype(float)

    def _finish_result_visualization(self, message: str) -> None:
        self._append_log(message)
        if hasattr(self, "_results_text"):
            self._results_text.setPlainText(message)
        self._set_status("结果显示已加载。")

    def _build_streamlines_from_vtu(self, seed_mode: str = "geometry"):
        """Follow Plan.md: read foamToVTK export and generate streamlines.
        Supports XML (.vtu/.vtp) and legacy (.vtk) formats.
        Returns (streamline_output, source_grid, seed_label, seed_count, speed_range)."""
        if self._current_project is None:
            raise RuntimeError("未选择项目")
        case_dir = self._current_project.case_dir
        vtk_dir = case_dir / "VTK"
        if not vtk_dir.exists():
            raise RuntimeError(
                "未找到 VTK 目录。请先运行仿真，仿真结束后会自动执行 foamToVTK 导出。"
            )

        # ── 1. 读取体网格与流场数据 (Plan.md 第22-37行) ──
        grid_data = None
        # try XML .vtu first
        volumes_dir = vtk_dir / "volumes"
        if volumes_dir.exists():
            vtu_files = sorted(volumes_dir.glob("*.vtu"))
            if vtu_files:
                vtu_reader = vtk.vtkXMLUnstructuredGridReader()
                vtu_reader.SetFileName(str(vtu_files[-1]))
                vtu_reader.Update()
                grid_data = vtu_reader.GetOutput()
        # fallback: legacy .vtk
        if grid_data is None:
            vtk_files = sorted(vtk_dir.glob("*.vtk"))
            if not vtk_files:
                raise RuntimeError(f"{vtk_dir} 中没有 .vtu 或 .vtk 体网格文件")
            legacy_reader = vtk.vtkUnstructuredGridReader()
            legacy_reader.SetFileName(str(vtk_files[-1]))
            legacy_reader.Update()
            grid_data = legacy_reader.GetOutput()
        if grid_data is None or grid_data.GetNumberOfPoints() == 0:
            raise RuntimeError("读取体网格文件失败")

        # ── 2. 读取种子源 ──
        if seed_mode == "inlet":
            seed_geometry, seed_label = self._load_inlet_seed_surface(vtk_dir)
        else:
            seed_geometry, seed_label = self._load_geometry_seed_surface(vtk_dir)
        if seed_geometry is None or seed_geometry.GetNumberOfPoints() == 0:
            if seed_mode == "inlet":
                raise RuntimeError("未找到入口面文件。请确认仿真完成并生成了 VTK/inlet/ 目录。")
            raise RuntimeError("未找到几何体表面文件。请先导入 STL，并确认 foamToVTK 已导出几何体边界面。")
        seed_count = seed_geometry.GetNumberOfPoints()

        # ── 3. 流线追踪 (Plan.md 第45-74行) ──
        def _trace_streamlines(seed_source, direction: str):
            stream_tracer = vtkStreamTracer()
            stream_tracer.SetInputData(grid_data)
            stream_tracer.SetInputArrayToProcess(
                0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "U"
            )
            stream_tracer.SetSourceData(seed_source)
            stream_tracer.SetMaximumPropagation(500)
            stream_tracer.SetIntegrationStepUnit(vtk.vtkStreamTracer.LENGTH_UNIT)
            stream_tracer.SetInitialIntegrationStep(0.1)
            stream_tracer.SetMinimumIntegrationStep(0.01)
            stream_tracer.SetMaximumIntegrationStep(1.0)
            if direction == "backward":
                stream_tracer.SetIntegrationDirectionToBackward()
            elif direction == "both":
                stream_tracer.SetIntegrationDirectionToBoth()
            else:
                stream_tracer.SetIntegrationDirectionToForward()
            stream_tracer.Update()
            return stream_tracer.GetOutput()

        def _trace_all_directions(seed_source, directions=("forward",)):
            for direction in directions:
                traced = _trace_streamlines(seed_source, direction)
                if traced is not None and traced.GetNumberOfLines() > 0:
                    return traced
            return None

        if seed_mode == "inlet":
            seed_source = self._densify_seed_points(seed_geometry)
            stream_lines = _trace_all_directions(seed_source, ("forward", "backward", "both"))
            seed_label = f"{seed_label} 入口面"
            seed_count = seed_source.GetNumberOfPoints()
        else:
            flow_direction = self._mean_flow_direction(grid_data)
            near_wall_seed_geometry = self._near_wall_seed_shell(seed_geometry, flow_direction)
            stream_lines = _trace_all_directions(near_wall_seed_geometry)
            if stream_lines is None or stream_lines.GetNumberOfLines() == 0:
                stream_lines = _trace_all_directions(self._offset_seed_surface(seed_geometry, flow_direction=flow_direction))
            seed_label = f"{seed_label} 近壁层"
            seed_count = near_wall_seed_geometry.GetNumberOfPoints() or seed_count
        if stream_lines is None or stream_lines.GetNumberOfLines() == 0:
            raise RuntimeError(f"{seed_label} 没有生成下游方向有效流线，请确认速度场结果和边界面存在。")

        # ── 4. 速度大小着色 (Plan.md 第77-85行: 优先 VelocityMagnitude, fallback |U|) ──
        speed_range = (0.0, 1.0)
        vel_mag = stream_lines.GetPointData().GetArray("VelocityMagnitude")
        if vel_mag is not None:
            rng = vel_mag.GetRange()
            speed_range = (float(rng[0]), float(rng[1]))
        else:
            u_array = stream_lines.GetPointData().GetArray("U")
            if u_array is not None:
                vecs = vtk_to_numpy(u_array)
                if vecs.ndim == 2 and vecs.shape[1] >= 3:
                    speeds = np.linalg.norm(vecs[:, :3], axis=1)
                else:
                    speeds = np.abs(vecs.astype(float))
                smin, smax = float(speeds.min()), float(speeds.max())
                speed_array = numpy_to_vtk(speeds.astype(float), deep=True)
                speed_array.SetName("U_mag")
                stream_lines.GetPointData().AddArray(speed_array)
                stream_lines.GetPointData().SetActiveScalars("U_mag")
                speed_range = (smin, smax)

        return stream_lines, grid_data, seed_label, seed_count, speed_range

    def _load_inlet_seed_surface(self, vtk_dir: Path):
        surfaces_dir = vtk_dir / "surfaces"
        if surfaces_dir.exists():
            inlet_vtp = surfaces_dir / "inlet.vtp"
            if inlet_vtp.exists():
                reader = vtk.vtkXMLPolyDataReader()
                reader.SetFileName(str(inlet_vtp))
                reader.Update()
                return reader.GetOutput(), "inlet"

        inlet_dir = vtk_dir / "inlet"
        if not inlet_dir.exists():
            for child in sorted(vtk_dir.iterdir()):
                if child.is_dir() and "inlet" in child.name.lower():
                    inlet_dir = child
                    break
        if inlet_dir.exists():
            inlet_files = sorted(inlet_dir.glob("*.vtk"))
            if inlet_files:
                reader = vtk.vtkDataSetReader()
                reader.SetFileName(str(inlet_files[-1]))
                reader.Update()
                return self._as_poly_data(reader.GetOutput()), inlet_dir.name
        return None, ""

    def _densify_seed_points(self, seed_geometry):
        points = seed_geometry.GetPoints()
        if points is None or seed_geometry.GetNumberOfPoints() == 0:
            return seed_geometry
        dense_points = vtk.vtkPoints()
        for index in range(seed_geometry.GetNumberOfPoints()):
            dense_points.InsertNextPoint(points.GetPoint(index))
        cells = seed_geometry.GetPolys() or seed_geometry.GetLines()
        if cells is not None:
            raw_cells = vtk_to_numpy(cells.GetData())
            index = 0
            while index < len(raw_cells):
                count = int(raw_cells[index])
                index += 1
                ids = raw_cells[index : index + count].astype(int)
                index += count
                if count < 2:
                    continue
                cell_points = np.array([points.GetPoint(int(point_id)) for point_id in ids], dtype=float)
                centroid = cell_points.mean(axis=0)
                dense_points.InsertNextPoint(float(centroid[0]), float(centroid[1]), float(centroid[2]))
                for edge_index in range(count):
                    p0 = cell_points[edge_index]
                    p1 = cell_points[(edge_index + 1) % count]
                    midpoint = (p0 + p1) * 0.5
                    dense_points.InsertNextPoint(float(midpoint[0]), float(midpoint[1]), float(midpoint[2]))
        bounds = seed_geometry.GetBounds()
        if bounds is not None and all(np.isfinite(bounds)):
            spans = np.array(
                [
                    bounds[1] - bounds[0],
                    bounds[3] - bounds[2],
                    bounds[5] - bounds[4],
                ],
                dtype=float,
            )
            normal_axis = int(np.argmin(spans))
            plane_axes = [axis for axis in range(3) if axis != normal_axis]
            target_grid_points = 38000
            aspect = max(float(spans[plane_axes[0]]), 1e-9) / max(float(spans[plane_axes[1]]), 1e-9)
            count_a = max(24, int(np.sqrt(target_grid_points * aspect)))
            count_b = max(24, int(target_grid_points / count_a))
            fixed_value = (bounds[normal_axis * 2] + bounds[normal_axis * 2 + 1]) * 0.5
            axis_a_values = np.linspace(bounds[plane_axes[0] * 2], bounds[plane_axes[0] * 2 + 1], count_a)
            axis_b_values = np.linspace(bounds[plane_axes[1] * 2], bounds[plane_axes[1] * 2 + 1], count_b)
            for value_a in axis_a_values:
                for value_b in axis_b_values:
                    point = [0.0, 0.0, 0.0]
                    point[normal_axis] = fixed_value
                    point[plane_axes[0]] = float(value_a)
                    point[plane_axes[1]] = float(value_b)
                    dense_points.InsertNextPoint(point)
        dense_seed = vtk.vtkPolyData()
        dense_seed.SetPoints(dense_points)
        return dense_seed

    def _mean_flow_direction(self, grid_data) -> np.ndarray | None:
        u_array = grid_data.GetPointData().GetArray("U")
        if u_array is None:
            u_array = grid_data.GetCellData().GetArray("U")
        if u_array is None:
            return None
        vectors = vtk_to_numpy(u_array)
        if vectors.ndim != 2 or vectors.shape[1] < 3:
            return None
        vectors = vectors[:, :3].astype(float)
        speeds = np.linalg.norm(vectors, axis=1)
        usable = speeds > max(float(speeds.max()) * 0.02, 1e-12) if speeds.size else []
        if not np.any(usable):
            return None
        direction = vectors[usable].mean(axis=0)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            return None
        return direction / norm

    def _load_geometry_seed_surface(self, vtk_dir: Path):
        poly_data = self._build_geometry_patch_surface()
        if poly_data is None or poly_data.GetNumberOfPoints() == 0:
            raise RuntimeError("未能从 constant/polyMesh 中提取几何体表面，请先生成网格并确认障碍物 patch 存在。")
        return poly_data, "polyMesh 几何体表面"

    def _read_exported_geometry_surface(self, vtk_dir: Path, asset):
        names = {
            asset.stored_path.stem.lower(),
            asset.stored_path.name.lower(),
            asset.name.lower(),
            Path(asset.name).stem.lower(),
        }
        surfaces_dir = vtk_dir / "surfaces"
        if surfaces_dir.exists():
            for vtp_path in sorted(surfaces_dir.glob("*.vtp")):
                stem = vtp_path.stem.lower()
                if stem in names or any(name and name in stem for name in names):
                    reader = vtk.vtkXMLPolyDataReader()
                    reader.SetFileName(str(vtp_path))
                    reader.Update()
                    return reader.GetOutput()

        for child in sorted(vtk_dir.iterdir()):
            if not child.is_dir() or child.name in {"volumes", "surfaces"}:
                continue
            child_name = child.name.lower()
            if child_name not in names and not any(name and name in child_name for name in names):
                continue
            vtk_files = sorted(child.glob("*.vtk"))
            if not vtk_files:
                continue
            reader = vtk.vtkDataSetReader()
            reader.SetFileName(str(vtk_files[-1]))
            reader.Update()
            return self._as_poly_data(reader.GetOutput())
        return None

    def _read_stl_seed_surface(self, stl_path: Path):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(str(stl_path))
        reader.Update()
        return reader.GetOutput()

    def _as_poly_data(self, data_object):
        if data_object is None:
            return None
        if isinstance(data_object, vtk.vtkPolyData):
            return data_object
        geometry_filter = vtk.vtkGeometryFilter()
        geometry_filter.SetInputDataObject(data_object)
        geometry_filter.Update()
        return geometry_filter.GetOutput()

    def _near_wall_seed_shell(self, seed_geometry, flow_direction: np.ndarray | None = None):
        pieces = []
        for offset_ratio in (0.004, 0.010, 0.018):
            shifted = self._offset_seed_surface(
                seed_geometry,
                offset_ratio=offset_ratio,
                flow_direction=flow_direction,
            )
            if shifted is not None and shifted.GetNumberOfPoints() > 0:
                pieces.append(shifted)
        if not pieces:
            return seed_geometry
        append_filter = vtk.vtkAppendPolyData()
        for piece in pieces:
            append_filter.AddInputData(piece)
        append_filter.Update()
        return append_filter.GetOutput()

    def _offset_seed_surface(
        self,
        seed_geometry,
        offset_ratio: float = 0.01,
        flow_direction: np.ndarray | None = None,
    ):
        bounds = seed_geometry.GetBounds()
        if bounds is None or not all(np.isfinite(bounds)):
            return seed_geometry
        center = np.array(
            [
                (bounds[0] + bounds[1]) * 0.5,
                (bounds[2] + bounds[3]) * 0.5,
                (bounds[4] + bounds[5]) * 0.5,
            ],
            dtype=float,
        )
        diagonal = float(
            np.linalg.norm(
                [
                    bounds[1] - bounds[0],
                    bounds[3] - bounds[2],
                    bounds[5] - bounds[4],
                ]
            )
        )
        offset = max(diagonal * float(offset_ratio), 1e-5)
        shifted_points = vtk.vtkPoints()
        source_points = seed_geometry.GetPoints()
        for index in range(seed_geometry.GetNumberOfPoints()):
            point = np.array(source_points.GetPoint(index), dtype=float)
            direction = point - center
            norm = float(np.linalg.norm(direction))
            if norm > 1e-12:
                unit_direction = direction / norm
                if flow_direction is not None and float(np.dot(unit_direction, flow_direction)) > 0.25:
                    continue
                point = point + unit_direction * offset
            shifted_points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
        shifted = vtk.vtkPolyData()
        shifted.SetPoints(shifted_points)
        return shifted

    def _ensure_native_vtk_viewer(self) -> None:
        if self._native_vtk_viewer is None:
            self._native_vtk_viewer = NativeVtkViewerDialog(self)
            self._native_vtk_viewer.destroyed.connect(self._clear_native_vtk_viewer)
        if self._current_project is not None:
            self._native_vtk_viewer.set_geometry_assets([])
            self._native_vtk_viewer.set_domain_boundary_polydata(
                self._build_domain_boundary_patch_surface()
            )
            self._native_vtk_viewer.set_geometry_polydata(self._build_geometry_patch_surface())
        self._native_vtk_viewer.show()
        self._native_vtk_viewer.raise_()
        self._native_vtk_viewer.activateWindow()

    def _build_domain_boundary_patch_surface(self, time_value: float | None = None):
        """Extract the real meshed computational-domain boundary patches.

        This is the snappyHexMesh result boundary (for example pipe/cylinder
        inlet, outlet and wall), not the rectangular blockMesh background box.
        """
        if self._current_project is None:
            return None
        boundary_path = (
            self._current_project.case_dir / "constant" / "polyMesh" / "boundary"
        )
        if not boundary_path.exists():
            return None
        patch_names = self._result_domain_boundary_patch_names()
        if not patch_names:
            return None
        try:
            return self._context.openfoam_vtk_service.build_patch_surfaces(
                self._current_project,
                include_patches=patch_names,
                time_value=time_value,
            )
        except (OSError, RuntimeError, ValueError):
            return None

    def _build_geometry_patch_surface(self, time_value: float | None = None):
        """Extract the meshed geometry surface from constant/polyMesh.

        Domain boundary faces (inlet/outlet/walls + the 6 computational-domain
        faces named in the mesh-generation table) are excluded so only the
        immersed geometry patches remain. Returns None when no mesh exists."""
        if self._current_project is None:
            return None
        boundary_path = (
            self._current_project.case_dir / "constant" / "polyMesh" / "boundary"
        )
        if not boundary_path.exists():
            return None
        try:
            return self._context.openfoam_vtk_service.build_patch_surfaces(
                self._current_project,
                exclude_patches=self._domain_boundary_patch_names(),
                time_value=time_value,
            )
        except (OSError, RuntimeError, ValueError):
            return None

    def _result_domain_boundary_patch_names(self) -> tuple[str, ...]:
        """Patch names that represent the user-defined CFD domain surface."""
        names: set[str] = set()
        getter = getattr(self, "_get_domain_face_definitions", None)
        if callable(getter):
            try:
                for face in getter():
                    name = str(face.get("name", "")).strip()
                    if name:
                        names.add(name)
            except (AttributeError, KeyError, TypeError):
                pass
        names.update({"inlet", "outlet", "wall", "walls"})
        blocked_background_names = {
            "background", "front", "back", "top", "bottom", "left", "right",
            "empty", "sym", "symmetry", "symmetryplane",
        }
        return tuple(sorted(name for name in names if name.lower() not in blocked_background_names))

    def _domain_boundary_patch_names(self) -> tuple[str, ...]:
        """Names of computational-domain boundary patches (not geometry)."""
        names = {
            "inlet", "outlet", "walls", "wall", "background",
            "front", "back", "top", "bottom", "left", "right",
        }
        getter = getattr(self, "_get_domain_face_definitions", None)
        if callable(getter):
            try:
                for face in getter():
                    name = str(face.get("name", "")).strip()
                    if name:
                        names.add(name)
            except (AttributeError, KeyError, TypeError):
                pass
        return tuple(names)

    def _is_domain_boundary_stl_asset(self, asset) -> bool:
        """Return True for CAD-domain STL files that should not be shown as geometry.

        The result viewer should render and seed streamlines from the immersed
        geometry only. Mesh generation writes CAD/imported domain boundaries as
        `cad_domain.stl` or named inlet/outlet/wall patches in triSurface, so
        those files must be excluded from the "仅显示STL附近" path.
        """
        candidates = {
            str(getattr(asset, "name", "")).lower(),
            Path(str(getattr(asset, "name", ""))).stem.lower(),
        }
        stored_path = getattr(asset, "stored_path", None)
        if stored_path is not None:
            candidates.add(stored_path.name.lower())
            candidates.add(stored_path.stem.lower())
        source_path = getattr(asset, "source_path", "")
        if source_path:
            source = Path(str(source_path))
            candidates.add(source.name.lower())
            candidates.add(source.stem.lower())

        domain_names = {name.lower() for name in self._domain_boundary_patch_names()}
        domain_names.update({"cad_domain", "caddomain", "domain", "inlet", "outlet", "wall", "walls", "sym", "empty"})
        for name in candidates:
            if not name:
                continue
            if name in domain_names:
                return True
            if name.startswith("domain_patch") or name.startswith("cad_domain"):
                return True
        return False

    def _clear_native_vtk_viewer(self, *_args) -> None:
        self._native_vtk_viewer = None
