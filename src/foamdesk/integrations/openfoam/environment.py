from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from foamdesk.domain.models import OpenFOAMEnvironmentStatus
from foamdesk.services.settings_service import AppSettingsService


class OpenFOAMEnvironmentDetector:
    """Detects whether the current WSL environment can run OpenFOAM commands."""

    CANDIDATE_ENV_SCRIPTS = (
        "/opt/openfoam/etc/bashrc",
        "/usr/lib/openfoam/openfoam*/etc/bashrc",
        "/usr/lib/openfoam*/etc/bashrc",
    )

    def __init__(self, settings_service: AppSettingsService) -> None:
        self._settings_service = settings_service

    def detect(self) -> OpenFOAMEnvironmentStatus:
        bash_path = shutil.which("bash") or ""
        if not bash_path:
            return OpenFOAMEnvironmentStatus(
                is_available=False,
                bash_path="",
                env_script_path=None,
                foam_version=None,
                detail="未找到 bash，当前环境不符合 WSL/Linux 运行前提。",
            )

        settings = self._settings_service.load()
        env_script_path = settings.openfoam_env_script or self._find_env_script()
        if not env_script_path:
            return OpenFOAMEnvironmentStatus(
                is_available=False,
                bash_path=bash_path,
                env_script_path=None,
                foam_version=None,
                detail="未找到 OpenFOAM 环境脚本，请在设置中指定 bashrc 路径。",
            )

        command = (
            f"source '{env_script_path}' >/dev/null 2>&1 && "
            "printf '%s' \"${WM_PROJECT_VERSION:-unknown}\""
        )
        completed = subprocess.run(
            ["bash", "-lc", command],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return OpenFOAMEnvironmentStatus(
                is_available=False,
                bash_path=bash_path,
                env_script_path=env_script_path,
                foam_version=None,
                detail=completed.stderr.strip() or "OpenFOAM 环境脚本加载失败。",
            )

        version = completed.stdout.strip() or "unknown"
        return OpenFOAMEnvironmentStatus(
            is_available=True,
            bash_path=bash_path,
            env_script_path=env_script_path,
            foam_version=version,
            detail="OpenFOAM 环境可用。",
        )

    def _find_env_script(self) -> str | None:
        for candidate in self.CANDIDATE_ENV_SCRIPTS:
            if "*" not in candidate:
                if Path(candidate).exists():
                    return candidate
                continue

            parent = Path(candidate.split("*", 1)[0]).parent
            if not parent.exists():
                continue

            matches = sorted(str(path) for path in parent.glob(Path(candidate).name))
            if matches:
                return matches[0]
        return None

