from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from foamdesk.domain.models import SimulationProject


@dataclass(slots=True)
class GeometryAsset:
    name: str
    format: str
    source_path: str
    stored_path: Path
    size_bytes: int
    imported_at: str


class GeometryImportService:
    """Imports geometry files into the current OpenFOAM case."""

    SUPPORTED_EXTENSIONS = {".stl"}

    def import_stl(self, project: SimulationProject, source_path: Path) -> GeometryAsset:
        source = source_path.expanduser().resolve()
        if not source.exists() or not source.is_file():
            raise ValueError("几何文件不存在。")
        if source.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError("当前 MVP 只支持 STL。STEP/IGES/CATIA/SolidWorks 需要后续接入 CAD 内核。")
        if source.stat().st_size <= 0:
            raise ValueError("STL 文件为空，无法导入。")

        target_dir = project.case_dir / "constant" / "triSurface"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = self._unique_target_path(target_dir, self._safe_name(source.name))
        shutil.copy2(source, target_path)

        asset = GeometryAsset(
            name=target_path.name,
            format="STL",
            source_path=str(source),
            stored_path=target_path,
            size_bytes=target_path.stat().st_size,
            imported_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._append_manifest(project, asset)
        return asset

    def list_assets(self, project: SimulationProject) -> list[GeometryAsset]:
        manifest_path = self._manifest_path(project)
        if not manifest_path.exists():
            return []
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

        assets: list[GeometryAsset] = []
        for item in payload.get("assets", []):
            try:
                stored_path = project.case_dir / str(item["stored_path"])
                assets.append(
                    GeometryAsset(
                        name=str(item["name"]),
                        format=str(item["format"]),
                        source_path=str(item["source_path"]),
                        stored_path=stored_path,
                        size_bytes=int(item["size_bytes"]),
                        imported_at=str(item["imported_at"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return assets

    def format_assets(self, project: SimulationProject) -> str:
        assets = self.list_assets(project)
        snappy_dict = project.case_dir / "system" / "snappyHexMeshDict"
        if not assets:
            return (
                "当前 Case 暂无已导入几何。\n\n"
                "MVP 支持：STL 导入到 constant/triSurface。\n"
                "后续扩展：STEP/IGES/CATIA/SolidWorks 需要接入 OCCT/CAD 内核。\n\n"
                f"snappyHexMeshDict：{'已生成' if snappy_dict.exists() else '未生成'}"
            )

        lines = [
            "已导入几何清单",
            "",
            f"Case：{project.name}/{project.case_name}",
            f"triSurface：{project.case_dir / 'constant' / 'triSurface'}",
            f"snappyHexMeshDict：{snappy_dict if snappy_dict.exists() else '未生成'}",
            "",
        ]
        for index, asset in enumerate(assets, start=1):
            lines.extend(
                [
                    f"{index}. {asset.name}",
                    f"   - 格式：{asset.format}",
                    f"   - 大小：{asset.size_bytes} bytes",
                    f"   - 导入时间：{asset.imported_at}",
                    f"   - Case 内路径：{asset.stored_path}",
                    f"   - 原始路径：{asset.source_path}",
                ]
            )
        return "\n".join(lines)

    def generate_snappy_hex_mesh_dict(
        self,
        project: SimulationProject,
        asset_name: str | None = None,
    ) -> Path:
        assets = [
            asset
            for asset in self.list_assets(project)
            if asset.format.upper() == "STL" and asset.stored_path.exists()
        ]
        if not assets:
            raise ValueError("当前 Case 没有可用于 snappyHexMesh 的 STL 几何。")

        if asset_name:
            matching_assets = [asset for asset in assets if asset.name == asset_name]
            if not matching_assets:
                raise ValueError(f"未找到 STL 几何：{asset_name}")
            asset = matching_assets[0]
        else:
            asset = assets[0]

        system_dir = project.case_dir / "system"
        system_dir.mkdir(parents=True, exist_ok=True)
        dict_path = system_dir / "snappyHexMeshDict"
        dict_path.write_text(self._snappy_hex_mesh_dict(asset.name), encoding="utf-8")
        self._write_snappy_config(project, asset, dict_path)
        return dict_path

    def _append_manifest(self, project: SimulationProject, asset: GeometryAsset) -> None:
        manifest_path = self._manifest_path(project)
        payload = {"assets": []}
        if manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {"assets": []}
        payload.setdefault("assets", [])
        payload["assets"].append(
            {
                "name": asset.name,
                "format": asset.format,
                "source_path": asset.source_path,
                "stored_path": str(asset.stored_path.relative_to(project.case_dir)),
                "size_bytes": asset.size_bytes,
                "imported_at": asset.imported_at,
            }
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _manifest_path(self, project: SimulationProject) -> Path:
        return project.case_dir / "constant" / "triSurface" / "geometry_manifest.json"

    def _write_snappy_config(self, project: SimulationProject, asset: GeometryAsset, dict_path: Path) -> None:
        config_path = project.case_dir / "constant" / "triSurface" / "snappy_config.json"
        payload = {
            "asset_name": asset.name,
            "dict_path": str(dict_path.relative_to(project.case_dir)),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scope": "MVP starter snappyHexMeshDict; config generation only.",
        }
        config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _safe_name(self, name: str) -> str:
        safe = "".join(character if character.isalnum() or character in ("-", "_", ".") else "_" for character in name)
        return safe or "geometry.stl"

    def _unique_target_path(self, target_dir: Path, file_name: str) -> Path:
        target = target_dir / file_name
        if not target.exists():
            return target
        stem = target.stem
        suffix = target.suffix
        counter = 2
        while True:
            candidate = target_dir / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def _snappy_hex_mesh_dict(self, stl_name: str) -> str:
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}

castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
    {stl_name}
    {{
        type triSurfaceMesh;
        name importedGeometry;
    }}
}}

castellatedMeshControls
{{
    maxLocalCells 100000;
    maxGlobalCells 2000000;
    minRefinementCells 0;
    nCellsBetweenLevels 3;

    features
    (
    );

    refinementSurfaces
    {{
        importedGeometry
        {{
            level (1 2);
        }}
    }}

    resolveFeatureAngle 30;

    refinementRegions
    {{
    }}

    locationInMesh (0.5 0.5 0.5);
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
}}

addLayersControls
{{
    relativeSizes true;
    layers
    {{
    }}
    expansionRatio 1.0;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}}

meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-15;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.02;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}

debug 0;
mergeTolerance 1e-6;
"""
