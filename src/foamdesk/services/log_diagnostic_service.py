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
        diagnostics.extend(self._diagnose_numerical_instability(output))
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

    def extract_fatal_error_block(self, output: str, context_lines: int = 12) -> str | None:
        """Return the original FOAM FATAL block so users can report the real error."""
        lines = output.splitlines()
        start_index: int | None = None
        for index, line in enumerate(lines):
            if "FOAM FATAL ERROR" in line or "FOAM FATAL IO ERROR" in line:
                start_index = max(0, index - 1)
                break
        if start_index is None:
            return None

        end_index = min(len(lines), start_index + context_lines)
        for index in range(start_index + 1, len(lines)):
            line = lines[index]
            if "FOAM exiting" in line:
                end_index = index + 1
                break
            if index > start_index + 2 and line.startswith("End"):
                end_index = index
                break
        block = "\n".join(lines[start_index:end_index]).strip()
        return block or None

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

    def _diagnose_numerical_instability(self, output: str) -> list[OpenFoamDiagnostic]:
        courant_values = [
            float(match.group(1))
            for match in re.finditer(r"Courant Number mean:\s*[0-9.eE+-]+\s+max:\s*([0-9.eE+-]+)", output)
        ]
        has_sigfpe = "sigFpe::sigHandler" in output or "Floating point exception" in output
        has_huge_residual = bool(re.search(r"Final residual = [0-9.]+e\+[0-9]{2,}", output))
        max_courant = max(courant_values) if courant_values else None
        if not has_sigfpe and not has_huge_residual and (max_courant is None or max_courant <= 1.0):
            return []

        detail_parts = ["求解过程中出现数值不稳定或浮点异常。"]
        if max_courant is not None:
            detail_parts.append(f"日志中最大 Courant Number 约为 {max_courant:g}。")
        if has_huge_residual:
            detail_parts.append("速度或压力残差突然变成极大值，说明解已经发散。")
        if has_sigfpe:
            detail_parts.append("OpenFOAM 捕获到 sigFpe 浮点异常。")
        return [
            OpenFoamDiagnostic(
                title="数值发散或时间步过大",
                detail=" ".join(detail_parts),
                suggestion="先把 `仿真参数` 里的 `deltaT` 降低，例如从 0.005 改到 0.001 或 0.0005；同时确认入口速度不要过大，并优先用 upwind 稳定格式。",
            )
        ]

    def _diagnose_fatal_error(self, output: str) -> list[OpenFoamDiagnostic]:
        if "FOAM FATAL ERROR" not in output and "FOAM FATAL IO ERROR" not in output:
            return []
        fatal_block = self.extract_fatal_error_block(output, context_lines=8)
        detail = "OpenFOAM 在读取 case、生成网格或求解时遇到不可恢复错误。"
        if fatal_block:
            first_specific_line = self._first_specific_fatal_line(fatal_block)
            if first_specific_line:
                detail = f"{detail} 关键原文：{first_specific_line}"
        return [
            OpenFoamDiagnostic(
                title="OpenFOAM 致命错误",
                detail=detail,
                suggestion="查看下方 `OpenFOAM 原始致命错误` 段落，优先按其中的文件名、字段名、patch 名或字典关键字修复。",
            )
        ]

    def _first_specific_fatal_line(self, fatal_block: str) -> str | None:
        ignored_prefixes = (
            "--> FOAM FATAL",
            "From function",
            "in file",
        )
        for line in fatal_block.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(ignored_prefixes):
                continue
            if stripped == "FOAM exiting":
                continue
            return stripped
        return None

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
