from __future__ import annotations

from PySide6.QtWidgets import QComboBox

from foamdesk.domain.models import SimulationParameters


class ParametersLogicMixin:
    def _show_parameters(self, parameters: SimulationParameters) -> None:
        self._set_combo_value(self._solver_name_combo, parameters.solver_name)
        self._set_combo_value(self._turbulence_model_combo, parameters.turbulence_model)
        self._set_combo_value(self._numeric_scheme_combo, parameters.numeric_scheme)
        self._set_combo_value(self._fv_solution_preset_combo, parameters.fv_solution_preset)
        self._end_time_input.setText(f"{parameters.end_time:.12g}")
        self._delta_t_input.setText(f"{parameters.delta_t:.12g}")
        self._write_interval_input.setValue(parameters.write_interval)
        if hasattr(self, "_material_combo"):
            self._material_combo.blockSignals(True)
            self._set_combo_value(self._material_combo, parameters.material_name)
            self._material_combo.blockSignals(False)
            self._density_input.setText(f"{parameters.density:.12g}")
            self._viscosity_input.setText(f"{parameters.viscosity:.12g}")
            self._dynamic_viscosity_input.setText(f"{parameters.dynamic_viscosity:.12g}")

    def _read_parameter_inputs(self) -> SimulationParameters:
        base = self._context.case_parameter_service.defaults()
        if self._current_project is not None:
            try:
                base = self._context.case_parameter_service.load(self._current_project)
            except (OSError, ValueError):
                pass
        return SimulationParameters(
            solver_name=str(self._solver_name_combo.currentData()) if hasattr(self, "_solver_name_combo") else base.solver_name,
            end_time=float(self._end_time_input.text().strip()),
            delta_t=float(self._delta_t_input.text().strip()),
            write_interval=self._write_interval_input.value(),
            max_iterations=base.max_iterations,
            residual_tolerance=base.residual_tolerance,
            material_name=str(self._material_combo.currentData()) if hasattr(self, "_material_combo") else base.material_name,
            density=float(self._density_input.text().strip()) if hasattr(self, "_density_input") else base.density,
            viscosity=float(self._viscosity_input.text().strip()) if hasattr(self, "_viscosity_input") else base.viscosity,
            dynamic_viscosity=float(self._dynamic_viscosity_input.text().strip()) if hasattr(self, "_dynamic_viscosity_input") else base.dynamic_viscosity,
            turbulence_model=str(self._turbulence_model_combo.currentData()) if hasattr(self, "_turbulence_model_combo") else base.turbulence_model,
            numeric_scheme=str(self._numeric_scheme_combo.currentData()) if hasattr(self, "_numeric_scheme_combo") else base.numeric_scheme,
            fv_solution_preset=str(self._fv_solution_preset_combo.currentData()) if hasattr(self, "_fv_solution_preset_combo") else base.fv_solution_preset,
        )

    def _apply_material_preset(self) -> None:
        if not hasattr(self, "_material_combo"):
            return
        material_name = str(self._material_combo.currentData())
        if material_name == "custom":
            return
        try:
            preset = self._context.case_parameter_service.material_preset(material_name)
        except ValueError as error:
            self._show_error(str(error))
            return
        self._density_input.setText(f"{preset.density:.12g}")
        self._viscosity_input.setText(f"{preset.viscosity:.12g}")
        self._dynamic_viscosity_input.setText(f"{preset.dynamic_viscosity:.12g}")
        message = "已应用物性预设，点击“补齐边界和物性”后写入 OpenFOAM 物性文件。"
        if hasattr(self, "_physics_prepare_status"):
            self._physics_prepare_status.append(message)
        elif hasattr(self, "_parameter_status_label"):
            self._parameter_status_label.setText(message)

    def _calculate_dynamic_viscosity(self) -> None:
        try:
            density = float(self._density_input.text().strip())
            viscosity = float(self._viscosity_input.text().strip())
        except ValueError:
            self._show_error("rho 和 nu 必须是数字，才能计算 mu。")
            return
        self._dynamic_viscosity_input.setText(f"{density * viscosity:.12g}")
        if hasattr(self, "_material_combo"):
            self._set_combo_value(self._material_combo, "custom")
        message = "已按 rho × nu 计算动力粘度 mu。"
        if hasattr(self, "_physics_prepare_status"):
            self._physics_prepare_status.append(message)
        elif hasattr(self, "_parameter_status_label"):
            self._parameter_status_label.setText(message)

    def _set_combo_value(self, combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index < 0:
            index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _load_case_parameters(self) -> None:
        if self._current_project is None:
            self._parameter_status_label.setText("请先新建或打开项目。")
            self._set_parameter_inputs_enabled(False)
            if hasattr(self, "_solver_select_status_label"):
                self._solver_select_status_label.setText("请先新建或打开项目。")
            self._set_solver_inputs_enabled(False)
            return

        try:
            self._context.project_service.ensure_minimal_case_template(self._current_project)
            parameters = self._context.case_parameter_service.load(self._current_project)
        except (OSError, ValueError) as error:
            self._show_error(f"加载参数失败：{error}")
            return

        self._show_parameters(parameters)
        self._set_parameter_inputs_enabled(True)
        self._set_solver_inputs_enabled(True)
        self._parameter_status_label.setText(f"已加载项目参数：{self._current_project.name}")
        if hasattr(self, "_solver_select_status_label"):
            self._solver_select_status_label.setText(
                f"已加载求解器：{parameters.solver_name}，湍流模型：{parameters.turbulence_model}"
            )
        self._refresh_solver_run_panel()
        self._set_status("参数已加载。")

    def _save_case_parameters(self) -> bool:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return False

        try:
            parameters = self._read_parameter_inputs()
            self._context.project_service.ensure_minimal_case_template(self._current_project)
            self._context.case_parameter_service.save(self._current_project, parameters)
            solver_files = self._context.project_service.ensure_solver_support_files(self._current_project, parameters)
        except ValueError as error:
            self._show_error(f"参数不合法：{error}")
            return False
        except OSError as error:
            self._show_error(f"保存参数失败：{error}")
            return False

        self._parameter_status_label.setText(
            f"参数已保存到 Case：solver={parameters.solver_name}, endTime={parameters.end_time:g}, "
            f"deltaT={parameters.delta_t:g}, writeInterval={parameters.write_interval}, "
            f"material={parameters.material_name}, rho={parameters.density:g}, "
            f"nu={parameters.viscosity:g}, mu={parameters.dynamic_viscosity:g}"
        )
        if hasattr(self, "_solver_select_status_label"):
            self._solver_select_status_label.setText(
                f"求解器配置已保存：{parameters.solver_name} / {parameters.turbulence_model} / "
                f"{parameters.numeric_scheme} / {parameters.fv_solution_preset}"
            )
        self._refresh_solver_run_panel()
        self._append_log(
            "参数已写入 system/controlDict、system/fvSchemes、system/fvSolution、"
            "constant/physicalProperties 和 system/foamdesk_simulation_config.json。"
        )
        relative_files = [str(path.relative_to(self._current_project.case_dir)) for path in solver_files]
        self._append_log("已生成求解器/湍流/patch 边界支持文件：")
        self._append_log("\n".join(f"- {path}" for path in relative_files))
        self._set_status("参数保存完成。")
        return True

    def _save_control_dict_parameters(self) -> bool:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return False
        try:
            self._context.project_service.ensure_minimal_case_template(self._current_project)
            end_time = float(self._end_time_input.text().strip())
            delta_t = float(self._delta_t_input.text().strip())
            write_interval = self._write_interval_input.value()
            if end_time <= 0 or delta_t <= 0 or write_interval <= 0:
                raise ValueError("endTime、deltaT 和 writeInterval 必须大于 0。")
            if delta_t > end_time:
                raise ValueError("deltaT 不能大于 endTime。")
            control_path = self._current_project.case_dir / "system" / "controlDict"
            control_text = control_path.read_text(encoding="utf-8")
            parameter_service = self._context.case_parameter_service
            control_text = parameter_service._replace_assignment(control_text, "endTime", parameter_service._format_float(end_time))
            control_text = parameter_service._replace_assignment(control_text, "deltaT", parameter_service._format_float(delta_t))
            control_text = parameter_service._replace_assignment(control_text, "writeInterval", str(write_interval))
            control_path.write_text(control_text, encoding="utf-8")
        except ValueError as error:
            self._show_error(f"controlDict 参数不合法：{error}")
            return False
        except OSError as error:
            self._show_error(f"保存 controlDict 失败：{error}")
            return False

        self._parameter_status_label.setText(
            f"controlDict 已保存：endTime={end_time:g}, deltaT={delta_t:g}, writeInterval={write_interval}"
        )
        self._append_log("仿真参数页已写入 system/controlDict。")
        self._refresh_solver_run_panel()
        self._set_status("controlDict 参数保存完成。")
        return True

    def _restore_default_parameters(self) -> None:
        parameters = self._context.case_parameter_service.defaults()
        self._show_parameters(parameters)
        self._parameter_status_label.setText("已恢复默认参数，点击“保存参数到 Case”后生效。")
        self._set_parameter_inputs_enabled(self._current_project is not None)
        self._set_solver_inputs_enabled(self._current_project is not None)
        if hasattr(self, "_solver_select_status_label"):
            self._solver_select_status_label.setText("已恢复默认求解器配置，点击“保存求解器配置”后生效。")
        self._refresh_solver_run_panel()

    def _refresh_solver_run_panel(self, status_text: str | None = None) -> None:
        if not hasattr(self, "_solver_status_label"):
            return

        self._solver_status_label.setText(f"状态：{status_text or '空闲'}")
        self._refresh_solver_metrics_panel()
        if hasattr(self, "_solver_diagnostic_text"):
            self._solver_diagnostic_text.setPlainText(f"最近诊断：\n{self._last_diagnostic_summary}")
        if self._current_project is None:
            self._solver_project_label.setText("当前项目：未选择")
            self._solver_case_path_label.setText("Case 路径：未选择")
            self._solver_parameter_summary.setPlainText("参数摘要：请先新建或打开项目。")
            return

        self._solver_project_label.setText(f"当前项目：{self._current_project.name}")
        self._solver_case_path_label.setText(f"Case 路径：{self._current_project.case_dir}")
        try:
            parameters = self._context.case_parameter_service.load(self._current_project)
        except (OSError, ValueError) as error:
            self._solver_parameter_summary.setPlainText(f"参数摘要加载失败：{error}")
            return

        self._solver_parameter_summary.setPlainText(
            "参数摘要：\n"
            f"- solver：{parameters.solver_name}\n"
            f"- endTime：{parameters.end_time:g}\n"
            f"- deltaT：{parameters.delta_t:g}\n"
            f"- writeInterval：{parameters.write_interval}\n"
            f"- maxIterations：{parameters.max_iterations}\n"
            f"- residualTolerance：{parameters.residual_tolerance:g}\n"
            f"- material：{parameters.material_name}\n"
            f"- rho：{parameters.density:g}\n"
            f"- nu：{parameters.viscosity:g}\n"
            f"- mu：{parameters.dynamic_viscosity:g}\n"
            f"- turbulence：{parameters.turbulence_model}\n"
            f"- fvSchemes：{parameters.numeric_scheme}\n"
            f"- fvSolution：{parameters.fv_solution_preset}\n"
            "\n"
            "输出位置：当前 Case 目录下的时间步目录、constant/polyMesh 和日志面板。"
        )
