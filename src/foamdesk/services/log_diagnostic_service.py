from __future__ import annotations

import re

from foamdesk.domain.models import OpenFoamDiagnostic


class OpenFoamLogDiagnosticService:
    """Turns common OpenFOAM failures into beginner-readable diagnostics."""

    def diagnose(self, output: str) -> list[OpenFoamDiagnostic]:
        diagnostics: list[OpenFoamDiagnostic] = []
        diagnostics.extend(self._diagnose_missing_file(output))
        diagnostics.extend(self._diagnose_patch_field(output))
        diagnostics.extend(self._diagnose_pressure_reference(output))
        diagnostics.extend(self._diagnose_fatal_error(output))
        return self._dedupe(diagnostics)

    def format_diagnostics(self, diagnostics: list[OpenFoamDiagnostic]) -> str:
        if not diagnostics:
            return "未识别到已知 OpenFOAM 错误类型，请查看日志面板中的原始输出。"

        blocks: list[str] = []
        for index, diagnostic in enumerate(diagnostics, start=1):
            blocks.append(
                f"诊断 {index}：{diagnostic.title}\n"
                f"原因：{diagnostic.detail}\n"
                f"建议：{diagnostic.suggestion}"
            )
        return "\n\n".join(blocks)

    def _diagnose_missing_file(self, output: str) -> list[OpenFoamDiagnostic]:
        diagnostics: list[OpenFoamDiagnostic] = []
        for match in re.finditer(r'cannot find file "([^"]+)"', output, flags=re.IGNORECASE):
            missing_path = match.group(1)
            diagnostics.append(
                OpenFoamDiagnostic(
                    title="缺少 OpenFOAM 必需文件",
                    detail=f"OpenFOAM 找不到文件：{missing_path}",
                    suggestion="重新运行前让 FoamDesk 自动补齐最小 case 文件；如果是自定义 case，请检查目录结构和文件名。",
                )
            )
        return diagnostics

    def _diagnose_patch_field(self, output: str) -> list[OpenFoamDiagnostic]:
        match = re.search(r"Cannot find patchField entry for (\S+)", output)
        if not match:
            return []
        patch_name = match.group(1)
        return [
            OpenFoamDiagnostic(
                title="边界条件名称不匹配",
                detail=f"`blockMeshDict` 中存在边界 `{patch_name}`，但 `0/U` 或 `0/p` 没有对应字段。",
                suggestion="运行前 FoamDesk 会尝试按 blockMeshDict 同步 U/p 边界；如果仍失败，请检查边界名称是否拼写一致。",
            )
        ]

    def _diagnose_pressure_reference(self, output: str) -> list[OpenFoamDiagnostic]:
        if "Unable to set reference cell for field p" not in output:
            return []
        return [
            OpenFoamDiagnostic(
                title="压力参考值缺失",
                detail="压力场 `p` 没有设置参考单元或参考点，OpenFOAM 无法确定压力基准。",
                suggestion="在 `system/fvSolution` 的 `PISO` 中配置 `pRefCell 0;` 和 `pRefValue 0;`。",
            )
        ]

    def _diagnose_fatal_error(self, output: str) -> list[OpenFoamDiagnostic]:
        if "FOAM FATAL ERROR" not in output and "FOAM FATAL IO ERROR" not in output:
            return []
        return [
            OpenFoamDiagnostic(
                title="OpenFOAM 致命错误",
                detail="OpenFOAM 在读取 case、生成网格或求解时遇到不可恢复错误。",
                suggestion="优先查看上方更具体的诊断；如果没有具体诊断，请把日志中的 FOAM FATAL 段落作为排查依据。",
            )
        ]

    def _dedupe(self, diagnostics: list[OpenFoamDiagnostic]) -> list[OpenFoamDiagnostic]:
        seen: set[tuple[str, str]] = set()
        result: list[OpenFoamDiagnostic] = []
        for diagnostic in diagnostics:
            key = (diagnostic.title, diagnostic.detail)
            if key in seen:
                continue
            seen.add(key)
            result.append(diagnostic)
        return result
