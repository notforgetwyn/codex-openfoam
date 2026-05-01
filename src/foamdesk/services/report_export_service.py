from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from foamdesk.domain.models import CaseResultIndex, OpenFoamVtkCaseInfo, SimulationProject


class ReportExportService:
    """Creates a first Markdown report for one FoamDesk case."""

    def export_markdown(
        self,
        project: SimulationProject,
        result_index: CaseResultIndex,
        vtk_info: OpenFoamVtkCaseInfo | None,
    ) -> Path:
        results_dir = project.case_dir / "foamdesk_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        report_path = results_dir / "report.md"
        report_path.write_text(
            self._build_markdown(project, result_index, vtk_info),
            encoding="utf-8",
        )
        return report_path

    def _build_markdown(
        self,
        project: SimulationProject,
        result_index: CaseResultIndex,
        vtk_info: OpenFoamVtkCaseInfo | None,
    ) -> str:
        metrics = self._load_metrics(project.case_dir / "foamdesk_results" / "metrics.json")
        residual_csv_path = project.case_dir / "foamdesk_results" / "residuals.csv"
        lines = [
            "# FoamDesk 仿真报告 v1",
            "",
            f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 1. 项目与 Case",
            "",
            f"- 项目名称：{project.name}",
            f"- Case 名称：{project.case_name}",
            f"- 项目路径：`{project.path}`",
            f"- Case 路径：`{project.case_dir}`",
            "",
            "## 2. 结果索引",
            "",
            f"- 网格目录：{'已生成' if result_index.has_mesh else '未生成'}",
            f"- 网格路径：`{result_index.mesh_path}`",
            f"- 最新时间步：{result_index.latest_time or '无'}",
            f"- 时间步目录：{', '.join(result_index.time_directories) if result_index.time_directories else '未发现'}",
            "",
            "### 字段文件",
            "",
        ]
        if result_index.field_files:
            lines.extend(f"- `{field}`" for field in result_index.field_files)
        else:
            lines.append("- 未发现字段文件")

        lines.extend([
            "",
            "## 3. VTK 可视化数据",
            "",
        ])
        if vtk_info is None:
            lines.append("- VTK 数据读取失败或当前 Case 暂无可视化数据。")
        else:
            lines.extend([
                f"- VTK marker：`{vtk_info.marker_file}`",
                f"- Block 数量：{vtk_info.block_count}",
                f"- 可用时间步：{', '.join(f'{time:g}' for time in vtk_info.time_values) if vtk_info.time_values else '未发现'}",
                f"- 点字段：{', '.join(vtk_info.point_arrays) if vtk_info.point_arrays else '未发现'}",
                f"- 单元字段：{', '.join(vtk_info.cell_arrays) if vtk_info.cell_arrays else '未发现'}",
            ])

        lines.extend([
            "",
            "## 4. 求解指标",
            "",
        ])
        if metrics:
            times = metrics.get("times") or []
            residuals = metrics.get("residuals") or []
            lines.extend([
                f"- 时间步数量：{len(times)}",
                f"- Courant 最大值：{metrics.get('courant_max') if metrics.get('courant_max') is not None else '未记录'}",
                f"- continuity global：{metrics.get('latest_continuity_global') if metrics.get('latest_continuity_global') is not None else '未记录'}",
                f"- 残差记录数量：{len(residuals)}",
            ])
            if residuals:
                latest = residuals[-1]
                lines.extend([
                    "",
                    "### 最新残差",
                    "",
                    f"- 字段：{latest.get('field')}",
                    f"- 初始残差：{latest.get('initial')}",
                    f"- 最终残差：{latest.get('final')}",
                    f"- 迭代次数：{latest.get('iterations')}",
                ])
        else:
            lines.append("- 未发现 `metrics.json`，请先运行仿真并导出求解指标。")

        lines.extend([
            "",
            "## 5. 残差数据",
            "",
            f"- 残差 CSV：`{residual_csv_path}`" if residual_csv_path.exists() else "- 未发现残差 CSV。",
            "",
            "## 6. 当前可视化能力",
            "",
            "- 压力点云图：结果 -> 云图 -> 加载压力云图",
            "- 压力表面云图：结果 -> 云图 -> 加载压力表面云图",
            "- 速度矢量箭头：结果 -> 速度场 -> 加载速度箭头",
            "- 速度切面：结果 -> 速度场 -> 加载速度切面",
            "- 速度流线：结果 -> 速度场 -> 加载速度流线",
            "- PNG 导出：3D 视图窗口 -> 导出 PNG",
            "",
            "## 7. 说明",
            "",
            "本报告是 FoamDesk v1 Markdown 报告，当前先汇总项目、Case、字段、时间步、残差和可视化能力。后续可扩展为自动嵌入图片、导出 PDF、生成工程结论。",
            "",
        ])
        return "\n".join(lines)

    def _load_metrics(self, path: Path) -> dict[str, object] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
