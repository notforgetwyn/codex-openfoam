from __future__ import annotations

import re
import shlex
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox, QTreeWidgetItem

from foamdesk.domain.models import SimulationProject
from foamdesk.ui.startup_window import StartupWindow


class ProjectProcessLogicMixin:
    def _save_current_state(self) -> None:
        self._save_settings()
        self._append_log("保存：当前设置已写入本地配置。")

    def _create_project(self) -> None:
        name, ok = QInputDialog.getText(self, "新建项目", "项目名称")
        if not ok:
            return
        try:
            project = self._context.project_service.create_project(name)
        except ValueError as error:
            self._show_error(str(error))
            return

        self._activate_project(project, "项目创建完成。")
        self._append_log(f"已创建项目：{project.path}")
        self._set_status("项目创建完成。")

    def _open_project(self) -> None:
        settings = self._context.settings_service.load()
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "打开 FoamDesk 项目",
            str(settings.workspace_dir / "projects"),
        )
        if not selected_dir:
            return
        try:
            project = self._context.project_service.open_project(Path(selected_dir))
        except ValueError as error:
            self._show_error(str(error))
            return

        self._activate_project(project, "项目打开完成。")
        self._append_log(f"已打开项目：{project.path}")
        self._set_status("项目打开完成。")

    def _activate_project(self, project: SimulationProject, status_text: str) -> None:
        self._current_project = project
        self._clear_case_runtime_state()
        self._context.project_service.remember_project(project)
        self._refresh_project_tree()
        self._workspace_tabs.setCurrentIndex(self.TAB_PROJECT_HOME)
        self._case_label.setText(f"当前 Case: {project.name}/{project.case_name}")
        self._refresh_geometry_panel()
        self._refresh_project_home_summary()
        self._load_boundaries_into_table()
        self._load_mesh_workflow_state()
        self._restore_project_result_state()
        self._append_log(f"当前项目：{project.path}")
        self._set_status(status_text)

    def _refresh_project_home_summary(self) -> None:
        if not hasattr(self, "_project_home_summary"):
            return
        if self._current_project is None:
            self._project_home_summary.setPlainText("请先选择或创建项目。")
            return

        project = self._current_project
        case_dir = project.case_dir
        lines = [
            "当前 Case 概览",
            "",
            f"- 项目名称：{project.name}",
            f"- Case 名称：{project.case_name}",
            f"- 项目路径：{project.path}",
            f"- Case 路径：{case_dir}",
            "",
            "OpenFOAM 关键文件：",
        ]

        required_files = [
            case_dir / "system" / "blockMeshDict",
            case_dir / "system" / "snappyHexMeshDict",
            case_dir / "system" / "controlDict",
            case_dir / "system" / "fvSchemes",
            case_dir / "system" / "fvSolution",
            case_dir / "0" / "U",
            case_dir / "0" / "p",
            case_dir / "constant" / "physicalProperties",
        ]
        for path in required_files:
            status = "OK" if path.exists() else "缺失"
            lines.append(f"- {status}：{path.relative_to(case_dir)}")

        lines.extend(["", "几何/STL："])
        try:
            assets = self._context.geometry_import_service.list_assets(project)
        except (OSError, ValueError) as error:
            assets = []
            lines.append(f"- 读取 STL 清单失败：{error}")
        if assets:
            for asset in assets:
                transform = asset.transform
                transform_text = ""
                if transform is not None:
                    transform_text = (
                        f"，scale={transform.scale:g}，"
                        f"translate=({transform.translate[0]:g}, {transform.translate[1]:g}, {transform.translate[2]:g})，"
                        f"rotate=({transform.rotate_degrees[0]:g}, {transform.rotate_degrees[1]:g}, {transform.rotate_degrees[2]:g})"
                    )
                lines.append(f"- {asset.name}：{asset.stored_path.name}{transform_text}")
        else:
            lines.append("- 当前 Case 暂无导入 STL。")

        mesh_dir = case_dir / "constant" / "polyMesh"
        result_times = [
            path.name
            for path in case_dir.iterdir()
            if path.is_dir() and self._is_openfoam_time_dir(path.name)
        ] if case_dir.exists() else []
        result_times = sorted(result_times, key=self._openfoam_time_sort_key)
        lines.extend(
            [
                "",
                "网格与结果：",
                f"- 网格目录：{'已生成' if mesh_dir.exists() else '未生成'}（constant/polyMesh）",
                f"- 时间步目录：{', '.join(result_times) if result_times else '暂无'}",
            ]
        )

        try:
            case_info = self._context.openfoam_vtk_service.inspect(project)
        except (OSError, RuntimeError, ValueError) as error:
            lines.append(f"- 结果字段：暂不可读（{error}）")
        else:
            fields = sorted(set(case_info.point_arrays) | set(case_info.cell_arrays))
            times = [f"{time:g}" for time in case_info.time_values]
            lines.append(f"- 可视化字段：{', '.join(fields) if fields else '暂无'}")
            lines.append(f"- 可视化时间步：{', '.join(times) if times else '暂无'}")

        lines.extend(
            [
                "",
                "建议下一步：",
                "- 如果缺少 blockMeshDict：进入“绘制几何”生成 blockMeshDict + STL。",
                "- 如果缺少 U/p/物性：进入“求解器准备”补齐边界和物性。",
                "- 如果网格未生成：进入“可视化网格生成”页生成 snappyHexMeshDict 并运行网格流程。",
                "- 如果已有结果字段：进入“结果”选择字段和显示方式查看 3D 结果。",
            ]
        )
        self._project_home_summary.setPlainText("\n".join(lines))

    def _is_openfoam_time_dir(self, name: str) -> bool:
        try:
            float(name)
        except ValueError:
            return False
        return True

    def _openfoam_time_sort_key(self, name: str) -> float:
        try:
            return float(name)
        except ValueError:
            return -1.0

    def _clear_case_runtime_state(self) -> None:
        self._current_process_output = ""
        self._last_diagnostic_summary = "暂无诊断。"
        if hasattr(self, "_solver_metric_summary"):
            self._solver_metric_summary.setPlainText("关键指标摘要：尚未运行。")
        if hasattr(self, "_solver_diagnostic_text"):
            self._solver_diagnostic_text.setPlainText("最近诊断：\n暂无诊断。")

    def _create_case(self) -> None:
        if self._current_project is None:
            self._show_error("请先选择项目。")
            return
        name, ok = QInputDialog.getText(self, "新增 Case", "Case 名称")
        if not ok:
            return
        try:
            project = self._context.project_service.create_case(self._current_project, name)
        except ValueError as error:
            self._show_error(str(error))
            return
        self._activate_project(project, "Case 创建完成。")
        self._append_log(f"已创建 Case：{project.case_dir}")

    def _delete_case(self) -> None:
        if self._current_project is None:
            self._show_error("请先选择项目。")
            return
        case_name = self._current_project.case_name
        cases = self._context.project_service.list_cases(self._current_project)
        if len(cases) <= 1:
            self._show_error("不能删除最后一个 Case。")
            return
        confirm = QMessageBox.question(
            self,
            "删除 Case",
            f"确定要删除 Case \"{case_name}\" 吗？\n\n此操作不可撤销，Case 目录将被永久删除。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            deleted_index = cases.index(case_name)
        except ValueError:
            self._show_error(f"Case \"{case_name}\" 不在项目列表中。")
            return
        fallback = cases[deleted_index - 1] if deleted_index > 0 else cases[deleted_index + 1]
        try:
            self._context.project_service.delete_case(self._current_project, case_name)
        except ValueError as error:
            self._show_error(str(error))
            return
        self._append_log(f"已删除 Case：{case_name}")
        project = self._context.project_service.switch_case(self._current_project, fallback)
        self._activate_project(project, f"Case 已回退到：{fallback}")

    def _return_to_project_selection(self) -> None:
        app = QApplication.instance()
        old_quit_on_close = app.quitOnLastWindowClosed() if app else True
        if app:
            app.setQuitOnLastWindowClosed(False)
        current_project = self._current_project
        self.close()
        QApplication.processEvents()
        startup_window = StartupWindow(self._context)
        selected_project = current_project
        status_text = "已取消项目选择。"
        if startup_window.exec() == 1 and startup_window.selected_project is not None:
            selected_project = startup_window.selected_project
            status_text = "已从项目选择页切换项目。"

        if selected_project is None:
            if app:
                app.setQuitOnLastWindowClosed(old_quit_on_close)
            return

        new_window = self.__class__(self._context, initial_project=selected_project)
        if app:
            app._foamdesk_main_window = new_window
        new_window.show()
        new_window.raise_()
        new_window.activateWindow()
        new_window._set_status(status_text)
        if app:
            app.setQuitOnLastWindowClosed(old_quit_on_close)

    def _restore_project_result_state(self) -> None:
        if self._current_project is None:
            return
        try:
            result_index = self._context.result_index_service.index(self._current_project)
        except OSError:
            return

        residuals_csv = self._current_project.case_dir / "foamdesk_results" / "residuals.csv"
        metrics_json = self._current_project.case_dir / "foamdesk_results" / "metrics.json"
        if result_index.latest_time is None and not residuals_csv.exists():
            self._task_text.setPlainText("任务状态：当前项目暂无求解结果")
            self._clear_case_runtime_state()
            return

        lines = [
            "任务状态：已加载项目已有结果",
            f"最新时间步：{result_index.latest_time or '无'}",
            f"网格目录：{'已生成' if result_index.has_mesh else '未生成'}",
            f"残差 CSV：{'已存在' if residuals_csv.exists() else '未生成'}",
            f"指标 JSON：{'已存在' if metrics_json.exists() else '未生成'}",
            "",
            "说明：打开项目只读取已有文件，不会自动重新求解；只有点击“运行”才会重新执行 OpenFOAM。",
        ]
        self._task_text.setPlainText("\n".join(lines))

    def _refresh_project_tree(self) -> None:
        if not hasattr(self, "_project_tree"):
            return
        self._project_tree.clear()
        if self._current_project is None:
            empty_item = QTreeWidgetItem(["未选择项目"])
            empty_item.setDisabled(True)
            self._project_tree.addTopLevelItem(empty_item)
            return

        project_item = QTreeWidgetItem([self._current_project.name])
        project_item.setData(0, Qt.ItemDataRole.UserRole, "")
        for case_name in self._context.project_service.list_cases(self._current_project):
            case_item = QTreeWidgetItem([case_name])
            case_item.setData(0, Qt.ItemDataRole.UserRole, case_name)
            if case_name == self._current_project.case_name:
                case_item.setText(0, f"{case_name}  ✓")
            project_item.addChild(case_item)
        self._project_tree.addTopLevelItem(project_item)
        self._project_tree.expandAll()

    def _on_project_tree_item_clicked(self, item: QTreeWidgetItem) -> None:
        case_name = item.data(0, Qt.ItemDataRole.UserRole)
        if not case_name or self._current_project is None:
            return
        try:
            project = self._context.project_service.switch_case(self._current_project, str(case_name))
        except ValueError as error:
            self._show_error(str(error))
            return
        self._activate_project(project, "Case 已切换。")
        self._append_log(f"当前 Case 目录：{project.case_dir}")

    def _search_projects(self) -> None:
        keyword, ok = QInputDialog.getText(self, "搜索项目", "项目名称关键字")
        if not ok:
            return
        normalized = keyword.strip().lower()
        root_count = self._project_tree.topLevelItemCount()
        for index in range(root_count):
            item = self._project_tree.topLevelItem(index)
            visible = not normalized or normalized in item.text(0).lower()
            item.setHidden(not visible)
        self._set_status("项目搜索已应用。")

    def _run_minimal_simulation(self) -> None:
        if self._foam_process and self._foam_process.state() != QProcess.ProcessState.NotRunning:
            self._show_error("已有任务正在运行，请先停止当前任务。")
            return
        if self._current_project is None:
            self._show_error("请先新建或打开一个项目。")
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

        repaired_files = self._context.project_service.ensure_minimal_case_template(self._current_project)
        if repaired_files:
            relative_files = [str(path.relative_to(self._current_project.case_dir)) for path in repaired_files]
            self._append_log("已补齐旧项目缺失的最小仿真文件：")
            self._append_log("\n".join(f"- {path}" for path in relative_files))

        block_mesh_dict = self._current_project.case_dir / "system" / "blockMeshDict"
        if not block_mesh_dict.exists():
            self._show_error("当前 case 缺少 system/blockMeshDict。")
            return

        self._workspace_tabs.setCurrentIndex(self.TAB_SOLVER_RUN)
        self._bottom_tabs.setCurrentIndex(0)
        self._task_text.setPlainText("任务状态：最小仿真运行中")
        self._current_process_output = ""
        self._last_diagnostic_summary = "本次任务正在运行，暂无失败诊断。"
        self._active_process_kind = "minimal"
        self._set_status("最小仿真运行中。")

        command = (
            f"source {shlex.quote(status.env_script_path)} >/dev/null 2>&1 && "
            f"cd {shlex.quote(str(self._current_project.case_dir))} && "
            f"blockMesh && {shlex.quote(parameters.solver_name)}"
        )
        self._foam_process = QProcess(self)
        self._foam_process.setProgram("bash")
        self._foam_process.setArguments(["-lc", command])
        self._foam_process.readyReadStandardOutput.connect(self._read_process_stdout)
        self._foam_process.readyReadStandardError.connect(self._read_process_stderr)
        self._foam_process.finished.connect(self._on_process_finished)
        self._foam_process.start()
        self._append_log(f"启动最小仿真：{self._current_project.case_dir}")
        self._append_log(f"执行流程：blockMesh -> {parameters.solver_name}")

    def _stop_current_process(self) -> None:
        if not self._foam_process or self._foam_process.state() == QProcess.ProcessState.NotRunning:
            self._task_text.setPlainText("任务状态：空闲")
            self._set_status("当前没有正在运行的任务。")
            return
        self._foam_process.terminate()
        if not self._foam_process.waitForFinished(3000):
            self._foam_process.kill()
        self._task_text.setPlainText("任务状态：已停止")
        self._active_process_kind = "idle"
        self._set_status("任务已停止。")

    def _read_process_stdout(self) -> None:
        if self._foam_process:
            output = bytes(self._foam_process.readAllStandardOutput()).decode(errors="replace")
            self._current_process_output += output
            self._append_log(output)

    def _read_process_stderr(self) -> None:
        if self._foam_process:
            output = bytes(self._foam_process.readAllStandardError()).decode(errors="replace")
            self._current_process_output += output
            self._append_log(output)

    def _on_process_finished(self, exit_code: int, _exit_status) -> None:
        process_kind = self._active_process_kind
        self._active_process_kind = "idle"
        if exit_code == 0 and process_kind == "simulationPipeline":
            self._task_text.setPlainText("任务状态：一键仿真流水线完成")
            self._last_diagnostic_summary = "一键仿真流水线正常完成，没有失败诊断。"
            self._refresh_geometry_panel()
            self._set_status("一键仿真流水线完成。")
        elif exit_code == 0 and process_kind == "preprocess":
            summary = self._format_check_mesh_summary(self._current_process_output)
            self._task_text.setPlainText("任务状态：一键前处理完成")
            self._last_diagnostic_summary = "一键前处理完成。\n\n" + summary
            self._problem_text.setPlainText(self._last_diagnostic_summary)
            self._refresh_geometry_panel()
            self._set_status("一键前处理完成。")
        elif exit_code == 0 and process_kind == "checkMesh":
            summary = self._format_check_mesh_summary(self._current_process_output)
            self._task_text.setPlainText("任务状态：checkMesh 完成")
            self._last_diagnostic_summary = summary
            self._problem_text.setPlainText(summary)
            self._set_status("checkMesh 完成。")
        elif exit_code == 0 and process_kind == "snappyHexMesh":
            self._task_text.setPlainText("任务状态：snappyHexMesh 完成")
            self._last_diagnostic_summary = "snappyHexMesh 正常完成，没有失败诊断。"
            self._refresh_geometry_panel()
            self._set_status("snappyHexMesh 完成。")
        elif exit_code == 0:
            self._task_text.setPlainText("任务状态：最小仿真完成")
            self._last_diagnostic_summary = "本次任务正常完成，没有失败诊断。"
            self._set_status("最小仿真完成。")
        else:
            self._update_diagnostics(exit_code)
            label = self._process_label(process_kind)
            failed_step = self._detect_failed_pipeline_step(process_kind, self._current_process_output)
            if failed_step:
                advice = self._pipeline_step_advice(failed_step)
                self._last_diagnostic_summary = (
                    f"失败步骤：{failed_step}\n\n{advice}\n\n{self._last_diagnostic_summary}"
                )
                self._problem_text.setPlainText(self._last_diagnostic_summary)
                self._bottom_tabs.setCurrentIndex(2)
            self._task_text.setPlainText(f"任务状态：{label}失败，退出码 {exit_code}")
    
            self._set_status(f"{label}失败，退出码 {exit_code}。")

    def _process_label(self, process_kind: str) -> str:
        labels = {
            "simulationPipeline": "一键仿真流水线",
            "preprocess": "一键前处理",
            "checkMesh": "checkMesh",
            "snappyHexMesh": "snappyHexMesh",
            "minimal": "最小仿真",
        }
        return labels.get(process_kind, "OpenFOAM 任务")

    def _detect_failed_pipeline_step(self, process_kind: str, output: str) -> str | None:
        if process_kind not in {"preprocess", "simulationPipeline"}:
            return None
        step_names = {
            "blockMesh": "blockMesh 背景网格生成",
            "snappyHexMesh": "snappyHexMesh 贴体网格生成",
            "checkMesh": "checkMesh 网格质量检查",
            "icoFoam": "icoFoam 最小求解",
            "simpleFoam": "simpleFoam 稳态求解",
            "pisoFoam": "pisoFoam 瞬态求解",
        }
        matches = re.findall(r"FOAMDESK_STEP:([A-Za-z0-9_]+)", output)
        if not matches:
            return "未知步骤，未识别到 FoamDesk 步骤标记"
        return step_names.get(matches[-1], matches[-1])

    def _pipeline_step_advice(self, failed_step: str) -> str:
        if "blockMesh" in failed_step:
            return (
                "修复建议：\n"
                "- 检查 `system/blockMeshDict` 是否存在并且语法正确。\n"
                "- 确认背景网格区域要包住 STL 几何，否则 snappyHexMesh 后续无法贴体。\n"
                "- 如果你刚新建 Case，可以先运行最小仿真验证 blockMesh 是否能单独通过。"
            )
        if "snappyHexMesh" in failed_step:
            return (
                "修复建议：\n"
                "- 检查 STL 是否已经导入到 `constant/triSurface`，文件名是否和 `snappyHexMeshDict` 一致。\n"
                "- 检查 `locationInMesh` 是否位于流体区域内部；这个点选错会导致网格区域判断失败。\n"
                "- 先降低最大加密等级，例如从 4 降到 2，减少网格生成压力。\n"
                "- 如果启用了边界层 addLayers，先关闭边界层再试。"
            )
        if "checkMesh" in failed_step:
            return (
                "修复建议：\n"
                "- 查看日志中的 `Failed`、`severely non-orthogonal`、`skewness` 等关键词。\n"
                "- 降低 snappy 加密等级或关闭边界层，先得到可用网格。\n"
                "- 如果非正交角或扭曲度过高，需要调整背景网格、STL 几何质量或 snappy 参数。"
            )
        if any(solver in failed_step for solver in ("icoFoam", "simpleFoam", "pisoFoam")):
            return (
                "修复建议：\n"
                "- 检查 `0/U` 和 `0/p` 的边界名称是否和网格 boundary 文件一致。\n"
                "- 如果 snappyHexMesh 生成了新的 patch，求解场文件也必须包含对应边界条件。\n"
                "- 检查 `system/controlDict`、`fvSchemes`、`fvSolution` 是否完整。\n"
                "- 先确认 checkMesh 通过，再运行求解器。"
            )
        return (
            "修复建议：\n"
            "- 查看底部日志中最后一个 OpenFOAM 报错块。\n"
            "- 优先检查当前 Case 的 system、constant、0 目录是否完整。"
        )

    def _format_check_mesh_summary(self, output: str) -> str:
        lines = ["checkMesh 网格质量检查摘要", ""]
        if "Mesh OK." in output:
            lines.append("总体结论：通过，OpenFOAM 输出 Mesh OK。")
        elif "Failed" in output or "failed" in output:
            lines.append("总体结论：存在失败检查，需要查看日志中的 Failed 项。")
        else:
            lines.append("总体结论：未识别到明确 Mesh OK，请查看完整日志。")

        checks = [
            ("点数量", r"points:\s+([0-9]+)"),
            ("面数量", r"faces:\s+([0-9]+)"),
            ("单元数量", r"cells:\s+([0-9]+)"),
            ("边界 patch 数量", r"boundary patches:\s+([0-9]+)"),
            ("最大长宽比", r"Max aspect ratio\s*=\s*([0-9.eE+-]+)"),
            ("最大非正交角", r"Mesh non-orthogonality Max:\s*([0-9.eE+-]+)"),
            ("最大扭曲度", r"Max skewness\s*=\s*([0-9.eE+-]+)"),
        ]
        for label, pattern in checks:
            match = re.search(pattern, output)
            if match:
                lines.append(f"{label}：{match.group(1)}")

        failed_lines = [
            line.strip()
            for line in output.splitlines()
            if "Failed" in line or "failed" in line or "Error" in line
        ]
        if failed_lines:
            lines.extend(["", "需要关注："])
            lines.extend(f"- {line}" for line in failed_lines[:8])

        lines.extend(
            [
                "",
                "说明：checkMesh 是 OpenFOAM 的网格体检工具。它不是求解流体，而是检查当前网格是否适合后续计算。",
            ]
        )
        return "\n".join(lines)

    def _update_diagnostics(self, exit_code: int) -> None:
        diagnostics = self._context.log_diagnostic_service.diagnose(self._current_process_output)
        summary = self._context.log_diagnostic_service.format_diagnostics(diagnostics)
        fatal_block = self._context.log_diagnostic_service.extract_fatal_error_block(
            self._current_process_output
        )
        if fatal_block:
            summary = f"{summary}\n\nOpenFOAM 原始致命错误：\n{fatal_block}"
        self._last_diagnostic_summary = f"退出码：{exit_code}\n\n{summary}"
        self._problem_text.setPlainText(self._last_diagnostic_summary)
        self._bottom_tabs.setCurrentIndex(2)
        self._append_log("已生成失败诊断。")

    def _show_current_case_path(self) -> None:
        if self._current_project is None:
            self._show_error("当前没有打开的项目。")
            return
        self._workspace_tabs.setCurrentIndex(self.TAB_PROJECT_HOME)
        self._append_log(f"当前 Case 目录：{self._current_project.case_dir}")
        self._set_status("已输出当前 Case 目录。")

    def _show_stage_summary(self) -> None:
        self._append_log(
            "当前 Sprint：Sprint 2 UI 可用性与设置系统；下一阶段：Sprint 3 项目管理与 OpenFOAM 最小执行闭环。"
        )
        self._set_status("已输出当前阶段说明。")

    def _show_error(self, message: str) -> None:
        self._problem_text.setPlainText(message)
        self._bottom_tabs.setCurrentIndex(2)
        self._append_log(f"错误：{message}")
        QMessageBox.warning(self, "FoamDesk", message)

    def _append_log(self, message: str) -> None:
        if hasattr(self, "_log_text"):
            self._log_text.append(message)

    def _set_status(self, message: str) -> None:
        self._task_label.setText(f"任务状态: {message}")
        self._append_log(message)
    # --- Draw geometry persistence ---
