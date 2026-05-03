from __future__ import annotations

import json
import math
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
    transform: StlTransform | None = None


@dataclass(slots=True)
class StlTransform:
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0
    rotate_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def is_identity(self) -> bool:
        return (
            self.scale == 1.0
            and self.translate == (0.0, 0.0, 0.0)
            and self.rotate_degrees == (0.0, 0.0, 0.0)
        )


@dataclass(slots=True)
class SnappyHexMeshSettings:
    min_refinement_level: int = 1
    max_refinement_level: int = 2
    location_in_mesh: tuple[float, float, float] = (0.5, 0.5, 0.5)
    add_layers: bool = False
    final_layer_thickness: float = 0.3


class GeometryImportService:
    """Imports geometry files into the current OpenFOAM case."""

    SUPPORTED_EXTENSIONS = {".stl"}

    def import_stl(
        self,
        project: SimulationProject,
        source_path: Path,
        transform: StlTransform | None = None,
    ) -> GeometryAsset:
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
        resolved_transform = transform or StlTransform()
        if resolved_transform.is_identity:
            shutil.copy2(source, target_path)
        else:
            self._write_transformed_ascii_stl(source, target_path, resolved_transform)

        asset = GeometryAsset(
            name=target_path.name,
            format="STL",
            source_path=str(source),
            stored_path=target_path,
            size_bytes=target_path.stat().st_size,
            imported_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._append_manifest(project, asset, resolved_transform)
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
                        transform=self._transform_from_manifest(item.get("transform")),
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
                    f"   - 变换：scale={asset.transform.scale if asset.transform else 1.0}, "
                    f"translate={asset.transform.translate if asset.transform else (0.0, 0.0, 0.0)}, "
                    f"rotate={asset.transform.rotate_degrees if asset.transform else (0.0, 0.0, 0.0)}",
                ]
            )
        return "\n".join(lines)

    def update_stl_transform(
        self,
        project: SimulationProject,
        asset_name: str,
        transform: StlTransform,
    ) -> GeometryAsset:
        manifest_path = self._manifest_path(project)
        if not manifest_path.exists():
            raise ValueError("当前 Case 没有几何 manifest，无法编辑 STL。")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assets = payload.get("assets", [])
        for item in assets:
            if str(item.get("name", "")) != asset_name:
                continue
            stored_path = project.case_dir / str(item["stored_path"])
            source_path = Path(str(item.get("source_path", "")))
            base_path = source_path if source_path.exists() else stored_path
            if not base_path.exists():
                raise ValueError(f"STL 源文件和 Case 文件都不存在：{asset_name}")
            if transform.is_identity and base_path != stored_path:
                shutil.copy2(base_path, stored_path)
            elif transform.is_identity:
                pass
            else:
                self._write_transformed_ascii_stl(base_path, stored_path, transform)
            item["size_bytes"] = stored_path.stat().st_size
            item["transform"] = {
                "translate": list(transform.translate),
                "scale": transform.scale,
                "rotate_degrees": list(transform.rotate_degrees),
            }
            manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return GeometryAsset(
                name=str(item["name"]),
                format=str(item["format"]),
                source_path=str(item["source_path"]),
                stored_path=stored_path,
                size_bytes=int(item["size_bytes"]),
                imported_at=str(item["imported_at"]),
                transform=transform,
            )
        raise ValueError(f"未找到 STL：{asset_name}")

    def generate_snappy_hex_mesh_dict(
        self,
        project: SimulationProject,
        asset_name: str | None = None,
        settings: SnappyHexMeshSettings | None = None,
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
        resolved_settings = settings or SnappyHexMeshSettings()
        dict_path.write_text(self._snappy_hex_mesh_dict(asset.name, resolved_settings), encoding="utf-8")
        self._write_snappy_config(project, asset, dict_path, resolved_settings)
        return dict_path

    def load_snappy_settings(self, project: SimulationProject) -> SnappyHexMeshSettings | None:
        config_path = self._snappy_config_path(project)
        if not config_path.exists():
            return None
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            raw_settings = payload.get("settings", {})
            location = raw_settings.get("location_in_mesh", [0.5, 0.5, 0.5])
            if not isinstance(location, list | tuple) or len(location) != 3:
                location = [0.5, 0.5, 0.5]
            return SnappyHexMeshSettings(
                min_refinement_level=int(raw_settings.get("min_refinement_level", 1)),
                max_refinement_level=int(raw_settings.get("max_refinement_level", 2)),
                location_in_mesh=(
                    float(location[0]),
                    float(location[1]),
                    float(location[2]),
                ),
                add_layers=bool(raw_settings.get("add_layers", False)),
                final_layer_thickness=float(raw_settings.get("final_layer_thickness", 0.3)),
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def _append_manifest(self, project: SimulationProject, asset: GeometryAsset, transform: StlTransform) -> None:
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
                "transform": {
                    "translate": list(transform.translate),
                    "scale": transform.scale,
                    "rotate_degrees": list(transform.rotate_degrees),
                },
            }
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _transform_from_manifest(self, payload) -> StlTransform:
        if not isinstance(payload, dict):
            return StlTransform()
        translate = payload.get("translate", [0.0, 0.0, 0.0])
        rotate = payload.get("rotate_degrees", [0.0, 0.0, 0.0])
        if not isinstance(translate, list | tuple) or len(translate) != 3:
            translate = [0.0, 0.0, 0.0]
        if not isinstance(rotate, list | tuple) or len(rotate) != 3:
            rotate = [0.0, 0.0, 0.0]
        try:
            return StlTransform(
                translate=(float(translate[0]), float(translate[1]), float(translate[2])),
                scale=float(payload.get("scale", 1.0)),
                rotate_degrees=(float(rotate[0]), float(rotate[1]), float(rotate[2])),
            )
        except (TypeError, ValueError):
            return StlTransform()

    def _manifest_path(self, project: SimulationProject) -> Path:
        return project.case_dir / "constant" / "triSurface" / "geometry_manifest.json"

    def _snappy_config_path(self, project: SimulationProject) -> Path:
        return project.case_dir / "constant" / "triSurface" / "snappy_config.json"

    def _write_snappy_config(
        self,
        project: SimulationProject,
        asset: GeometryAsset,
        dict_path: Path,
        settings: SnappyHexMeshSettings,
    ) -> None:
        config_path = self._snappy_config_path(project)
        payload = {
            "asset_name": asset.name,
            "dict_path": str(dict_path.relative_to(project.case_dir)),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scope": "MVP starter snappyHexMeshDict; config generation only.",
            "settings": {
                "min_refinement_level": settings.min_refinement_level,
                "max_refinement_level": settings.max_refinement_level,
                "location_in_mesh": list(settings.location_in_mesh),
                "add_layers": settings.add_layers,
                "final_layer_thickness": settings.final_layer_thickness,
            },
        }
        config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _safe_name(self, name: str) -> str:
        safe = "".join(character if character.isalnum() or character in ("-", "_", ".") else "_" for character in name)
        return safe or "geometry.stl"

    def _write_transformed_ascii_stl(self, source: Path, target: Path, transform: StlTransform) -> None:
        try:
            lines = source.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise ValueError("当前 STL 变换 MVP 只支持 ASCII STL。二进制 STL 请先不使用平移/缩放。") from error

        tx, ty, tz = transform.translate
        scale = float(transform.scale)
        rotation = self._rotation_matrix(transform.rotate_degrees)
        transformed_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("vertex "):
                parts = stripped.split()
                if len(parts) != 4:
                    raise ValueError("STL vertex 行格式异常，无法执行平移/缩放。")
                x, y, z = self._transform_vertex(
                    (float(parts[1]), float(parts[2]), float(parts[3])),
                    scale,
                    rotation,
                    (tx, ty, tz),
                )
                indent = line[: len(line) - len(line.lstrip())]
                transformed_lines.append(f"{indent}vertex {x:.9g} {y:.9g} {z:.9g}")
            else:
                transformed_lines.append(line)
        target.write_text("\n".join(transformed_lines) + "\n", encoding="utf-8")

    def _rotation_matrix(self, rotate_degrees: tuple[float, float, float]) -> tuple[tuple[float, float, float], ...]:
        rx, ry, rz = (math.radians(value) for value in rotate_degrees)
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        return (
            (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
            (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
            (-sy, cy * sx, cy * cx),
        )

    def _transform_vertex(
        self,
        vertex: tuple[float, float, float],
        scale: float,
        rotation: tuple[tuple[float, float, float], ...],
        translate: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        x, y, z = (value * scale for value in vertex)
        rx = rotation[0][0] * x + rotation[0][1] * y + rotation[0][2] * z
        ry = rotation[1][0] * x + rotation[1][1] * y + rotation[1][2] * z
        rz = rotation[2][0] * x + rotation[2][1] * y + rotation[2][2] * z
        return rx + translate[0], ry + translate[1], rz + translate[2]

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

    def _snappy_hex_mesh_dict(self, stl_name: str, settings: SnappyHexMeshSettings) -> str:
        min_level = max(0, int(settings.min_refinement_level))
        max_level = max(min_level, int(settings.max_refinement_level))
        location = tuple(float(value) for value in settings.location_in_mesh)
        add_layers = "true" if settings.add_layers else "false"
        layer_block = (
            """
        importedGeometry
        {
            nSurfaceLayers 2;
        }
"""
            if settings.add_layers
            else ""
        )
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}

castellatedMesh true;
snap            true;
addLayers       {add_layers};

geometry
{{
    importedGeometry
    {{
        type triSurfaceMesh;
        file "{stl_name}";
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
            level ({min_level} {max_level});
        }}
    }}

    resolveFeatureAngle 30;

    refinementRegions
    {{
    }}

    locationInMesh ({location[0]:.6g} {location[1]:.6g} {location[2]:.6g});
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
{layer_block}
    }}
    expansionRatio 1.0;
    finalLayerThickness {settings.final_layer_thickness:.6g};
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
