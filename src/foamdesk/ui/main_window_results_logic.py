from __future__ import annotations

from pathlib import Path

import numpy as np
import vtk
from matplotlib.figure import Figure
from PySide6.QtWidgets import QMessageBox
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkCommonMath import vtkRungeKutta45
from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer

from foamdesk.ui.visualization_widgets import NativeVtkViewerDialog, VtkViewerDialog


class ResultsLogicMixin:
    def _refresh_solver_metrics_panel(self) -> None:
        if not hasattr(self, "_solver_metric_summary"):
            return
        if not self._current_process_output.strip():
            self._solver_metric_summary.setPlainText("关键指标摘要：尚未运行。")
            return
        metrics = self._context.log_metric_service.parse(self._current_process_output)
        self._solver_metric_summary.setPlainText(
            self._context.log_metric_service.format_summary(metrics)
        )

    def _refresh_results_panel(self) -> None:
        if not hasattr(self, "_results_text"):
            return
        if self._current_project is None:
            self._results_text.setPlainText("请先新建或打开项目。")
            self._set_status("结果索引刷新失败：未选择项目。")
            return

        try:
            result_index = self._context.result_index_service.index(self._current_project)
        except OSError as error:
            self._show_error(f"刷新结果索引失败：{error}")
            return

        self._results_text.setPlainText(self._context.result_index_service.format_index(result_index))
        self._refresh_result_field_panel(show_errors=False)
        self._append_log("结果索引已刷新。")
        self._set_status("结果索引已刷新。")

    def _export_solver_metrics(self) -> bool:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return False
        if not self._current_process_output.strip():
            self._show_error("当前没有可导出的求解日志指标，请先运行一次最小仿真。")
            return False

        metrics = self._context.log_metric_service.parse(self._current_process_output)
        if not metrics.times and not metrics.residuals:
            self._show_error("未从当前日志中识别到可导出的求解指标。")
            return False

        try:
            json_path, csv_path = self._context.metric_export_service.export(
                self._current_project,
                metrics,
            )
        except OSError as error:
            self._show_error(f"导出求解指标失败：{error}")
            return False

        self._append_log(f"求解指标已导出：{json_path}")
        self._append_log(f"残差 CSV 已导出：{csv_path}")
        self._set_status("求解指标导出完成。")
        return True

    def _export_markdown_report(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return
        try:
            result_index = self._context.result_index_service.index(self._current_project)
            try:
                vtk_info = self._context.openfoam_vtk_service.inspect(self._current_project)
            except (OSError, RuntimeError):
                vtk_info = None
            asset_paths = self._export_report_assets()
            report_path = self._context.report_export_service.export_markdown(
                self._current_project,
                result_index,
                vtk_info,
                asset_paths,
            )
        except OSError as error:
            self._show_error(f"导出 Markdown 报告失败：{error}")
            return
        self._append_log(f"Markdown 报告已导出：{report_path}")
        self._results_text.setPlainText(
            "Markdown 报告已导出\n\n"
            f"路径：{report_path}\n\n"
            "当前报告包含：项目/Case、结果索引、VTK 字段、求解指标、残差数据、可视化能力说明和自动嵌入图片。"
        )
        self._set_status("Markdown 报告导出完成。")

    def _export_report_assets(self) -> list[Path]:
        if self._current_project is None:
            return []
        assets_dir = self._current_project.case_dir / "foamdesk_results" / "report_assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        asset_paths: list[Path] = []
        residual_path = self._export_residual_report_asset(assets_dir)
        if residual_path is not None:
            asset_paths.append(residual_path)
        if self._vtk_viewer is not None:
            asset_paths.extend(self._vtk_viewer.export_all_pngs(assets_dir))
        if asset_paths:
            self._append_log(f"报告图片已导出：{len(asset_paths)} 个")
        return asset_paths

    def _export_residual_report_asset(self, assets_dir: Path) -> Path | None:
        if self._current_project is None:
            return None
        try:
            series = self._context.residual_plot_service.load_series(self._current_project)
        except (OSError, ValueError):
            return None
        figure = Figure(figsize=(8, 3.6), tight_layout=True)
        axes = figure.add_subplot(111)
        for field, points in series.items():
            points = sorted(points, key=lambda item: item[0])
            axes.plot(
                [time for time, _residual in points],
                [residual for _time, residual in points],
                marker="o",
                linewidth=1.4,
                markersize=3,
                label=field,
            )
        axes.set_title("Residual Curve")
        axes.set_xlabel("Time")
        axes.set_ylabel("Final residual")
        axes.set_yscale("log")
        axes.grid(True, which="both", linestyle="--", alpha=0.35)
        axes.legend(loc="best")
        path = assets_dir / "residual_curve.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        return path

    def _plot_residual_curve(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return

        try:
            series = self._context.residual_plot_service.load_series(self._current_project)
        except (OSError, ValueError) as error:
            self._show_error(f"绘制残差曲线失败：{error}")
            return

        self._residual_figure.clear()
        axes = self._residual_figure.add_subplot(111)
        for field, points in series.items():
            points = sorted(points, key=lambda item: item[0])
            axes.plot(
                [time for time, _residual in points],
                [residual for _time, residual in points],
                marker="o",
                linewidth=1.4,
                markersize=3,
                label=field,
            )
        axes.set_title("Residual Curve")
        axes.set_xlabel("Time")
        axes.set_ylabel("Final residual")
        axes.set_yscale("log")
        axes.grid(True, which="both", linestyle="--", alpha=0.35)
        axes.legend(loc="best")
        self._residual_canvas.draw()
        self._append_log("残差曲线已绘制。")
        self._set_status("残差曲线已绘制。")

    def _refresh_result_field_panel(self, show_errors: bool = True) -> None:
        if not hasattr(self, "_result_field_combo"):
            return
        self._result_time_combo.clear()
        if self._current_project is None:
            if show_errors:
                self._show_error("请先新建或打开项目。")
            return
        try:
            case_info = self._context.openfoam_vtk_service.inspect(self._current_project)
        except (OSError, RuntimeError) as error:
            if show_errors:
                self._show_error(f"刷新结果场失败：{error}")
            return

        available = set(case_info.point_arrays) | set(case_info.cell_arrays)
        if "U" in available:
            available.add("mag(U)")
        current_field = self._result_field_combo.currentText().strip()
        self._result_field_combo.blockSignals(True)
        self._result_field_combo.clear()
        self._result_field_combo.addItems(self.RESULT_FIELDS)
        if current_field in self.RESULT_FIELDS:
            self._result_field_combo.setCurrentText(current_field)
        elif "p" in available:
            self._result_field_combo.setCurrentText("p")
        elif "U" in available:
            self._result_field_combo.setCurrentText("U")
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
        self._result_color_min_input.setValue(min_value)
        self._result_color_max_input.setValue(max_value if max_value != min_value else min_value + 1.0)
        self._result_minmax_label.setText(
            f"最大/最小值：{display_name} min={min_value:.6g}, max={max_value:.6g}（{storage} 字段）"
        )

    def _load_selected_result_display(self) -> None:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
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
        if not mode.startswith("Streamline"):
            self._configure_result_animation_source()

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
        if mode.startswith("Surface"):
            try:
                output, field_array = self._ensure_point_field(output, field_array, storage)
            except RuntimeError as error:
                self._show_error(str(error))
                return
            self._ensure_native_vtk_viewer()
            self._native_vtk_viewer.plot_surface(output, field_array, color_range, display_name)
            face_count = output.GetNumberOfPolys()
            self._finish_result_visualization(
                f"Surface 表面云图已加载到原生 VTK 3D 窗口：field={display_name}, time={selected_time}, faces={face_count}, range={color_range}"
            )
            return
        if mode.startswith("Contour"):
            try:
                output, field_array = self._ensure_point_field(output, field_array, storage)
            except RuntimeError as error:
                self._show_error(str(error))
                return
            axis = self._result_slice_axis_combo.currentText().strip()
            position = self._result_slice_position_input.value()
            self._ensure_native_vtk_viewer()
            resolved_axis, center = self._native_vtk_viewer.plot_contour(
                output,
                field_array,
                color_range,
                display_name,
                None if axis == "自动" else axis,
                position,
            )
            self._finish_result_visualization(
                f"Contour 等值线已加载到原生 VTK 3D 窗口：field={display_name}, time={selected_time}, axis={resolved_axis}, center={center:.6g}, range={color_range}"
            )
            return
        if mode.startswith("Iso-surface"):
            try:
                output, field_array = self._ensure_point_field(output, field_array, storage)
            except RuntimeError as error:
                self._show_error(str(error))
                return
            self._ensure_native_vtk_viewer()
            self._native_vtk_viewer.plot_iso_surface(
                output,
                field_array,
                color_range,
                display_name,
            )
            self._finish_result_visualization(
                f"Iso-surface 等值面已加载到原生 VTK 3D 窗口：field={display_name}, time={selected_time}, range={color_range}"
            )
            return
        if mode.startswith("Slice"):
            try:
                output, field_array = self._ensure_point_field(output, field_array, storage)
            except RuntimeError as error:
                self._show_error(str(error))
                return
            axis = self._result_slice_axis_combo.currentText().strip()
            position = self._result_slice_position_input.value()
            self._ensure_native_vtk_viewer()
            resolved_axis, center = self._native_vtk_viewer.plot_slice(
                output,
                field_array,
                color_range,
                display_name,
                None if axis == "自动" else axis,
                position,
            )
            self._finish_result_visualization(
                f"Slice 切片已加载到原生 VTK 3D 窗口：field={display_name}, time={selected_time}, axis={resolved_axis}, center={center:.6g}, range={color_range}"
            )
            return
        if mode.startswith("Streamline"):
            field_name = self._result_field_combo.currentText().strip()
            if field_name != "U":
                self._show_error("流线需要速度矢量场 U，不能直接使用 p、T、k 等标量场生成。")
                self._refresh_result_display_modes()
                return
            output, vector_array = self._point_vector_field(output, "U")
            if vector_array is None:
                self._show_error("流线当前需要点字段 U。请先确认当前 Case 输出了 U。")
                return
            try:
                stream_input = self._context.openfoam_vtk_service.build_case_output(
                    self._current_project,
                    time_value=self._selected_result_time_value(),
                )
                streamline_output, main_axis, seed_count, speed_range = self._build_vtk_streamlines(
                    stream_input,
                    output,
                    vector_array,
                )
            except (OSError, RuntimeError, ValueError) as error:
                self._show_error(f"生成流线失败：{error}")
                return
            self._ensure_native_vtk_viewer()
            self._native_vtk_viewer.plot_streamlines(
                output,
                streamline_output,
                color_range if color_range[1] > color_range[0] else speed_range,
            )
            line_count = streamline_output.GetNumberOfLines()
            point_count = streamline_output.GetNumberOfPoints()
            self._finish_result_visualization(
                f"Streamline 流线已加载到原生 VTK 3D 窗口：time={selected_time}, mainAxis={main_axis}, seeds={seed_count}, lines={line_count}, points={point_count}, speedRange={speed_range}"
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

    def _point_vector_field(self, output, array_name: str):
        vector_array = output.GetPointData().GetArray(array_name)
        if vector_array is not None:
            return output, vector_array
        cell_vector = output.GetCellData().GetArray(array_name)
        if cell_vector is None:
            return output, None
        return self._ensure_point_field(output, cell_vector, "cell")

    def _vector_magnitude_range(self, vector_array) -> tuple[float, float]:
        vectors = vtk_to_numpy(vector_array)
        if vectors.size == 0:
            return (0.0, 1.0)
        if vectors.ndim == 1:
            values = np.abs(vectors.astype(float))
        else:
            values = np.linalg.norm(vectors[:, : min(vectors.shape[1], 3)], axis=1)
        minimum = float(values.min())
        maximum = float(values.max())
        if maximum <= minimum:
            return minimum - 1.0, maximum + 1.0
        return minimum, maximum

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
        field_name = self._result_field_combo.currentText().strip()
        selected_time = self._result_time_combo.currentText().strip() or "默认"
        time_value = self._selected_result_time_value()
        geometry = self._context.openfoam_vtk_service.build_geometry_filter(
            self._current_project,
            time_value=time_value,
        )
        output = geometry.GetOutput()
        source_field = "U" if field_name == "mag(U)" else field_name
        field_array = output.GetPointData().GetArray(source_field)
        storage = "point"
        if field_array is None:
            field_array = output.GetCellData().GetArray(source_field)
            storage = "cell"
        if field_array is None:
            raise RuntimeError(f"当前 Case 没有字段 {field_name}")
        display_name = "mag(U)" if field_name == "mag(U)" else (
            field_name if field_array.GetNumberOfComponents() == 1 else f"|{field_name}|"
        )
        return output, field_array, display_name, selected_time, storage

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
        fallback = (float(values.min()), float(values.max()))
        minimum = float(self._result_color_min_input.value())
        maximum = float(self._result_color_max_input.value())
        if maximum <= minimum:
            return fallback if fallback[1] > fallback[0] else (fallback[0] - 1.0, fallback[1] + 1.0)
        return minimum, maximum

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

    def _build_vtk_streamlines(
        self,
        stream_input,
        bounds_poly_data,
        velocity_array,
        seed_resolution_x: int = 12,
        seed_resolution_y: int = 6,
        margin_ratio: float = 0.08,
        length_factor: float = 4.0,
        step_factor: float = 0.01,
    ):
        vtk_points = bounds_poly_data.GetPoints()
        if vtk_points is None:
            points = np.empty((0, 3), dtype=float)
        else:
            points = vtk_to_numpy(vtk_points.GetData())
        vectors = vtk_to_numpy(velocity_array)
        if points.size == 0 or vectors.size == 0 or vectors.ndim != 2 or vectors.shape[1] < 3:
            raise RuntimeError("当前 Case 没有可用于流线追踪的速度点字段。")
        count = min(len(points), len(vectors))
        points = points[:count]
        vectors = vectors[:count, :3]
        speeds = np.linalg.norm(vectors, axis=1)
        usable = speeds > 1e-12
        if not np.any(usable):
            raise RuntimeError("速度场全为 0，无法生成流线。")

        mean_vector = vectors[usable].mean(axis=0)
        axis = int(np.argmax(np.abs(mean_vector)))
        main_axis = "XYZ"[axis]
        bounds_min = points.min(axis=0)
        bounds_max = points.max(axis=0)
        spans = bounds_max - bounds_min
        direction_sign = 1.0 if mean_vector[axis] >= 0 else -1.0
        domain_size = max(float(spans.max()), 1e-9)
        inlet_value = bounds_min[axis] if direction_sign >= 0 else bounds_max[axis]
        seed_plane = inlet_value + direction_sign * max(float(spans[axis]) * 0.015, domain_size * 0.002)
        cross_axes = [index for index in range(3) if index != axis]
        seed_resolution_x = max(4, min(int(seed_resolution_x), 40))
        seed_resolution_y = max(2, min(int(seed_resolution_y), 24))
        length_factor = max(0.5, min(float(length_factor), 10.0))
        step_factor = max(0.002, min(float(step_factor), 0.1))

        seed_axes_values = []
        for cross_axis, requested_count in zip(cross_axes, (seed_resolution_x, seed_resolution_y), strict=True):
            span = float(spans[cross_axis])
            if span <= domain_size * 1e-5:
                seed_axes_values.append(np.array([(bounds_min[cross_axis] + bounds_max[cross_axis]) * 0.5], dtype=float))
                continue
            margin = min(span * max(margin_ratio, 0.0), span * 0.35)
            seed_axes_values.append(
                np.linspace(
                    float(bounds_min[cross_axis] + margin),
                    float(bounds_max[cross_axis] - margin),
                    requested_count,
                    dtype=float,
                )
            )

        seed_points = []
        for first_value in seed_axes_values[0]:
            for second_value in seed_axes_values[1]:
                point = np.zeros(3, dtype=float)
                point[axis] = seed_plane
                point[cross_axes[0]] = first_value
                point[cross_axes[1]] = second_value
                seed_points.append(point)

        vtk_seed_points = vtkPoints()
        for point in seed_points:
            vtk_seed_points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
        seed_data = vtkPolyData()
        seed_data.SetPoints(vtk_seed_points)

        initial_step = max(domain_size * step_factor, domain_size * 0.002)
        tracer = vtkStreamTracer()
        tracer.SetInputDataObject(stream_input)
        tracer.SetSourceData(seed_data)
        tracer.SetIntegrator(vtkRungeKutta45())
        tracer.SetIntegrationDirectionToForward()
        tracer.SetMaximumPropagation(domain_size * length_factor)
        tracer.SetInitialIntegrationStep(initial_step)
        tracer.SetMinimumIntegrationStep(max(initial_step * 0.1, domain_size * 0.0002))
        tracer.SetMaximumIntegrationStep(max(initial_step * 2.5, domain_size * 0.005))
        if hasattr(tracer, "SetMaximumError"):
            tracer.SetMaximumError(1e-6)
        tracer.SetComputeVorticity(False)
        tracer.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "U")
        tracer.Update()
        streamline_output = tracer.GetOutput()
        if streamline_output.GetNumberOfLines() == 0:
            raise RuntimeError("VTK StreamTracer 没有生成有效流线。")
        self._add_streamline_speed_array(streamline_output, "U")
        return streamline_output, main_axis, vtk_seed_points.GetNumberOfPoints(), (
            float(speeds[usable].min()),
            float(speeds[usable].max()),
        )

    def _add_streamline_speed_array(self, streamline_output, vector_name: str) -> None:
        vector_array = streamline_output.GetPointData().GetArray(vector_name)
        if vector_array is None:
            return
        vectors = vtk_to_numpy(vector_array)
        if vectors.size == 0:
            return
        if vectors.ndim == 1:
            speed_values = np.abs(vectors.astype(float))
        else:
            speed_values = np.linalg.norm(vectors[:, : min(vectors.shape[1], 3)], axis=1)
        speed_array = numpy_to_vtk(speed_values.astype(float), deep=True)
        speed_array.SetName("U_mag")
        streamline_output.GetPointData().AddArray(speed_array)
        streamline_output.GetPointData().SetActiveScalars("U_mag")

    def _ensure_vtk_viewer(self) -> None:
        if self._vtk_viewer is None:
            self._vtk_viewer = VtkViewerDialog(self)
        self._vtk_viewer.show()
        self._vtk_viewer.raise_()
        self._vtk_viewer.activateWindow()

    def _ensure_native_vtk_viewer(self) -> None:
        if self._native_vtk_viewer is None:
            self._native_vtk_viewer = NativeVtkViewerDialog(self)
            self._native_vtk_viewer.destroyed.connect(self._clear_native_vtk_viewer)
        if self._current_project is not None:
            try:
                assets = self._context.geometry_import_service.list_assets(self._current_project)
            except (OSError, ValueError):
                assets = []
            self._native_vtk_viewer.set_geometry_assets(assets)
        self._native_vtk_viewer.show()
        self._native_vtk_viewer.raise_()
        self._native_vtk_viewer.activateWindow()

    def _clear_native_vtk_viewer(self, *_args) -> None:
        self._native_vtk_viewer = None
