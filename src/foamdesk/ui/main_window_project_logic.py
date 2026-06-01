from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox, QTreeWidgetItem

from foamdesk.domain.models import SimulationProject
from foamdesk.ui.startup_window import StartupWindow


class ProjectProcessLogicMixin:
    def _before_project_change(self) -> None:
        if getattr(self, "_current_project", None) is None:
            return
        if hasattr(self, "_save_modeling_state_if_ready"):
            self._save_modeling_state_if_ready()
        if hasattr(self, "_clear_draw_geometry_cache"):
            self._clear_draw_geometry_cache()

    def _save_current_state(self) -> None:
        self._save_settings()
        if hasattr(self, "_save_modeling_state_if_ready"):
            self._save_modeling_state_if_ready()
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
        self._before_project_change()
        self._current_project = project
        self._clear_case_runtime_state()
        self._context.project_service.remember_project(project)
        self._refresh_project_tree()
        self._case_label.setText(f"当前 Case: {project.name}/{project.case_name}")
        self._refresh_geometry_panel()
        self._refresh_project_home_summary()
        self._init_modeling_state()
        self._load_sim_config_state()
        self._load_solver_run_state()
        self._init_mesh_import_state()
        self._load_mesh_workflow_state()
        self._restore_project_result_state()
        self._append_log(f"当前项目：{project.path}")
        self._set_status(status_text)
        # refresh the currently visible tab with new case data
        current_idx = self._workspace_tabs.currentIndex()
        self._on_workspace_tab_changed(current_idx)

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
        self._before_project_change()
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

    def _stop_current_process(self) -> None:
        if not self._foam_process or self._foam_process.state() == QProcess.ProcessState.NotRunning:
            self._task_text.setPlainText("任务状态：空闲")
            self._set_status("当前没有正在运行的任务。")
            return
        process_kind = self._active_process_kind
        if process_kind == "simulation" and hasattr(self, "_mark_simulation_paused"):
            self._sim_pause_requested = True
        self._foam_process.terminate()
        if not self._foam_process.waitForFinished(3000):
            self._foam_process.kill()
        self._task_text.setPlainText("任务状态：已停止")
        self._active_process_kind = "idle"
        if process_kind == "simulation" and hasattr(self, "_mark_simulation_paused"):
            self._mark_simulation_paused()
            return
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







    def _show_current_case_path(self) -> None:
        if self._current_project is None:
            self._show_error("当前没有打开的项目。")
            return
        import subprocess, sys, os
        case_dir = str(self._current_project.case_dir)
        try:
            if sys.platform == "win32":
                os.startfile(case_dir)
            elif sys.platform == "linux":
                subprocess.Popen(["xdg-open", case_dir])
            else:
                subprocess.Popen(["open", case_dir])
            self._set_status(f"已打开文件夹：{case_dir}")
        except Exception as e:
            self._show_error(f"无法打开文件夹：{e}")

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
