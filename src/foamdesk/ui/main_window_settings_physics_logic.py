from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import QTableWidgetItem

from foamdesk.domain.models import SimulationParameters
from foamdesk.ui.theme import THEMES, build_stylesheet


class SettingsPhysicsLogicMixin:
    def _apply_settings_theme(self) -> None:
        settings = self._context.settings_service.load()
        self._theme_index = self._theme_names.index(settings.theme_name)
        self.setStyleSheet(
            build_stylesheet(
                settings.theme_name,
                settings.background_color,
                settings.font_family,
                settings.font_size,
            )
        )
        if hasattr(self, "_theme_combo"):
            self._theme_combo.setCurrentText(settings.theme_name)
            self._background_color_input.setText(settings.background_color)
            self._workspace_input.setText(str(settings.workspace_dir))
            self._env_script_input.setText(settings.openfoam_env_script or "")
            self._font_combo.setCurrentFont(QFont(settings.font_family))
            self._font_size_input.setValue(settings.font_size)

    def _cycle_theme(self) -> None:
        self._theme_index = (self._theme_index + 1) % len(self._theme_names)
        self._set_theme(self._theme_names[self._theme_index])

    def _set_theme(self, theme_name: str) -> None:
        palette = THEMES[theme_name]
        settings = self._context.settings_service.load()
        updated_settings = type(settings)(
            workspace_dir=settings.workspace_dir,
            openfoam_env_script=settings.openfoam_env_script,
            theme_name=theme_name,
            background_color=palette.window_bg,
            font_family=settings.font_family,
            font_size=settings.font_size,
            last_project_path=settings.last_project_path,
        )
        self._context.settings_service.save(updated_settings)
        self._apply_settings_theme()
        self._refresh_environment_panels()
        self._set_status(f"主题已切换为 {theme_name}。")

    def _save_settings(self) -> None:
        background_color = self._background_color_input.text().strip() or "#1e1e1e"
        env_script = self._env_script_input.text().strip() or None
        settings = self._context.settings_service.load()
        updated_settings = type(settings)(
            workspace_dir=settings.workspace_dir.__class__(self._workspace_input.text().strip()),
            openfoam_env_script=env_script,
            theme_name=self._theme_combo.currentText(),
            background_color=background_color,
            font_family=self._font_combo.currentFont().family(),
            font_size=self._font_size_input.value(),
            last_project_path=settings.last_project_path,
        )
        self._context.settings_service.save(updated_settings)
        self._apply_settings_theme()
        self._refresh_environment_panels()
        self._set_status("设置已保存。")

    def _refresh_status_bar(self) -> None:
        status = self._context.environment_detector.detect()
        version_text = status.foam_version if status.is_available else "未就绪"
        self._version_label.setText(f"OpenFOAM: {version_text}")
        self._refresh_environment_panels(status)

    def _refresh_environment_panels(self, status=None) -> None:
        if status is None or isinstance(status, bool):
            status = self._context.environment_detector.detect()
        status_flag = "可用" if status.is_available else "不可用"
        self._environment_text.setPlainText(
            "OpenFOAM 环境检查结果\n\n"
            f"- 状态：{status_flag}\n"
            f"- bash 路径：{status.bash_path or '未找到'}\n"
            f"- 环境脚本：{status.env_script_path or '未配置'}\n"
            f"- OpenFOAM 版本：{status.foam_version or '未知'}\n"
            f"- 说明：{status.detail}\n"
        )
        self._append_log(f"环境检查完成：{status_flag}，OpenFOAM={status.foam_version or '未知'}")

    def _refresh_physics_prepare_panel(self) -> None:
        if not hasattr(self, "_physics_prepare_status"):
            return
        if self._current_project is None:
            self._physics_prepare_status.setPlainText("请先新建或打开项目。")
            return

        case_dir = self._current_project.case_dir
        physics_files = [
            case_dir / "0" / "U",
            case_dir / "0" / "p",
            case_dir / "constant" / "physicalProperties",
        ]
        missing = [path for path in physics_files if not path.exists()]
        try:
            parameters = self._context.case_parameter_service.load(self._current_project)
            parameter_text = (
                f"rho={parameters.density:g}, nu={parameters.viscosity:g}, "
                f"solver={parameters.solver_name}, turbulence={parameters.turbulence_model}"
            )
        except (OSError, ValueError) as error:
            parameter_text = f"参数尚未完整：{error}"

        lines = [
            "求解器准备状态",
            "",
            f"- 当前项目：{self._current_project.name}",
            f"- 当前 Case：{self._current_project.case_name}",
            f"- Case 路径：{case_dir}",
            f"- 参数/物性来源：{parameter_text}",
            "",
            "本页负责的边界条件和物性文件：",
        ]
        for path in physics_files:
            lines.append(f"- {'OK' if path.exists() else '缺失'}：{path.relative_to(case_dir)}")

        if missing:
            lines.extend(
                [
                    "",
                    "下一步建议：点击“补齐边界和物性”，生成 0/U、0/p 和 physicalProperties。",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "下一步建议：进入“求解器选择”和“仿真参数”确认配置，然后进入“求解运行”。",
                ]
            )
        self._populate_boundary_table()
        row_count = self._boundary_table.rowCount() if hasattr(self, "_boundary_table") else 0
        lines.extend(["", f"Patch 表格：已刷新 {row_count} 个边界。"])
        self._physics_prepare_status.setPlainText("\n".join(lines))
        self._set_status(f"求解器准备状态已刷新，发现 {row_count} 个边界。")
        self._append_log(f"求解器准备状态已刷新：{self._current_project.name}/{self._current_project.case_name}，patch={row_count}")

    def _prepare_physics_files(self) -> bool:
        if self._current_project is None:
            self._show_error("请先新建或打开项目。")
            return False
        try:
            self._context.project_service.ensure_minimal_case_template(self._current_project)
            parameters = self._read_parameter_inputs()
            self._context.case_parameter_service.save(self._current_project, parameters)
            synced_files = self._context.project_service.ensure_solver_support_files(
                self._current_project,
                parameters,
            )
            custom_files = self._apply_boundary_table_to_case(parameters)
        except (OSError, ValueError) as error:
            self._show_error(f"求解器准备失败：{error}")
            return False

        self._append_log("求解器准备完成：已补齐 0/U、0/p 和 physicalProperties。")
        written_files = list(dict.fromkeys([*synced_files, *custom_files]))
        if written_files:
            relative_files = [str(path.relative_to(self._current_project.case_dir)) for path in written_files]
            self._append_log("\n".join(f"- {path}" for path in relative_files))
        self._load_case_parameters()
        self._refresh_physics_prepare_panel()
        self._refresh_solver_run_panel("求解器准备完成")
        self._set_status("求解器准备完成。")
        return True

    def _physics_boundary_config_path(self) -> Path | None:
        if self._current_project is None:
            return None
        return self._current_project.case_dir / "system" / "foamdesk_boundary_table.json"

    def _physics_boundary_names(self) -> tuple[str, ...]:
        if self._current_project is None:
            return ()
        names = list(
            self._context.project_service._extract_boundary_names(
                self._current_project.case_dir / "system" / "blockMeshDict"
            )
        )
        if not names:
            names = ["inlet", "outlet", "fixedWalls"]
        stl_dir = self._current_project.case_dir / "constant" / "triSurface"
        if stl_dir.exists() and any(path.suffix.lower() == ".stl" for path in stl_dir.iterdir()):
            if "importedGeometry" not in names:
                names.append("importedGeometry")
        return tuple(dict.fromkeys(name for name in names if name.strip()))

    def _default_boundary_row(self, name: str) -> dict:
        lower = name.lower()
        if "inlet" in lower:
            return {"patch": name, "role": "入口", "u_type": "fixedValue", "u_value": "(1 0 0)", "p_type": "zeroGradient", "p_value": ""}
        if "outlet" in lower:
            return {"patch": name, "role": "出口", "u_type": "zeroGradient", "u_value": "", "p_type": "fixedValue", "p_value": "0"}
        if "symmetry" in lower:
            return {"patch": name, "role": "对称面", "u_type": "symmetryPlane", "u_value": "", "p_type": "symmetryPlane", "p_value": ""}
        return {"patch": name, "role": "壁面/物体", "u_type": "noSlip", "u_value": "", "p_type": "zeroGradient", "p_value": ""}

    def _load_boundary_table_config(self) -> dict[str, dict]:
        path = self._physics_boundary_config_path()
        if path is None or not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        rows = payload.get("patches", [])
        if not isinstance(rows, list):
            return {}
        result = {}
        for row in rows:
            if isinstance(row, dict) and row.get("patch"):
                result[str(row["patch"])] = row
        return result

    def _populate_boundary_table(self) -> None:
        if not hasattr(self, "_boundary_table") or self._current_project is None:
            return
        names = self._physics_boundary_names()
        saved_rows = self._load_boundary_table_config()
        self._boundary_table.blockSignals(True)
        self._boundary_table.setRowCount(0)
        for row_index, name in enumerate(names):
            row = {**self._default_boundary_row(name), **saved_rows.get(name, {})}
            self._boundary_table.insertRow(row_index)
            values = [
                row.get("patch", name),
                row.get("role", ""),
                row.get("u_type", ""),
                row.get("u_value", ""),
                row.get("p_type", ""),
                row.get("p_value", ""),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._boundary_table.setItem(row_index, column, item)
        self._boundary_table.blockSignals(False)

    def _read_boundary_table_rows(self) -> list[dict]:
        if not hasattr(self, "_boundary_table"):
            return []
        rows = []
        for row_index in range(self._boundary_table.rowCount()):
            values = []
            for column in range(self._boundary_table.columnCount()):
                item = self._boundary_table.item(row_index, column)
                values.append(item.text().strip() if item else "")
            if not values[0]:
                continue
            rows.append(
                {
                    "patch": values[0],
                    "role": values[1],
                    "u_type": values[2] or "zeroGradient",
                    "u_value": values[3],
                    "p_type": values[4] or "zeroGradient",
                    "p_value": values[5],
                }
            )
        return rows

    def _apply_boundary_table_to_case(self, parameters: SimulationParameters) -> list[Path]:
        if self._current_project is None:
            return []
        rows = self._read_boundary_table_rows()
        if not rows:
            self._populate_boundary_table()
            rows = self._read_boundary_table_rows()
        case_dir = self._current_project.case_dir
        zero_dir = case_dir / "0"
        constant_dir = case_dir / "constant"
        system_dir = case_dir / "system"
        zero_dir.mkdir(parents=True, exist_ok=True)
        constant_dir.mkdir(parents=True, exist_ok=True)
        system_dir.mkdir(parents=True, exist_ok=True)
        u_path = zero_dir / "U"
        p_path = zero_dir / "p"
        physical_path = constant_dir / "physicalProperties"
        config_path = self._physics_boundary_config_path()
        u_path.write_text(self._custom_velocity_field(rows), encoding="utf-8")
        p_path.write_text(self._custom_pressure_field(rows), encoding="utf-8")
        physical_path.write_text(
            self._context.case_parameter_service._default_properties_text(
                "physicalProperties",
                self._context.case_parameter_service._format_float(parameters.viscosity),
                self._context.case_parameter_service._format_float(parameters.density),
                self._context.case_parameter_service._format_float(parameters.dynamic_viscosity),
            ),
            encoding="utf-8",
        )
        written = [u_path, p_path, physical_path]
        if config_path is not None:
            config_path.write_text(
                json.dumps({"version": 1, "patches": rows}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            written.append(config_path)
        return written

    def _custom_velocity_field(self, rows: list[dict]) -> str:
        boundary_field = "\n".join(self._custom_velocity_entry(row) for row in rows)
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}

dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);

boundaryField
{{
{boundary_field}
}}
"""

    def _custom_pressure_field(self, rows: list[dict]) -> str:
        boundary_field = "\n".join(self._custom_pressure_entry(row) for row in rows)
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}}

dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{{
{boundary_field}
}}
"""

    def _custom_velocity_entry(self, row: dict) -> str:
        patch = row["patch"]
        boundary_type = row.get("u_type", "zeroGradient")
        value = row.get("u_value", "").strip() or "(0 0 0)"
        if boundary_type == "fixedValue":
            body = f"type            fixedValue;\n        value           uniform {value};"
        elif boundary_type == "zeroGradient":
            body = "type            zeroGradient;"
        elif boundary_type == "symmetryPlane":
            body = "type            symmetryPlane;"
        else:
            body = f"type            {boundary_type};"
        return f"""    {patch}
    {{
        {body}
    }}"""

    def _custom_pressure_entry(self, row: dict) -> str:
        patch = row["patch"]
        boundary_type = row.get("p_type", "zeroGradient")
        value = row.get("p_value", "").strip() or "0"
        if boundary_type == "fixedValue":
            body = f"type            fixedValue;\n        value           uniform {value};"
        elif boundary_type == "symmetryPlane":
            body = "type            symmetryPlane;"
        else:
            body = "type            zeroGradient;"
        return f"""    {patch}
    {{
        {body}
    }}"""
