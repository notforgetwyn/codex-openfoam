from __future__ import annotations

from pathlib import Path

from vtkmodules.vtkCommonDataModel import vtkCompositeDataSet, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkAppendPolyData
from vtkmodules.vtkFiltersGeometry import (
    vtkCompositeDataGeometryFilter,
    vtkGeometryFilter,
)
from vtkmodules.vtkIOGeometry import vtkOpenFOAMReader

from foamdesk.domain.models import OpenFoamVtkCaseInfo, SimulationProject


class OpenFoamVtkService:
    """Loads OpenFOAM case data through VTK's native OpenFOAM reader."""

    MARKER_FILE_NAME = "foamdesk.foam"

    def inspect(self, project: SimulationProject) -> OpenFoamVtkCaseInfo:
        self._ensure_mesh_exists(project)
        marker_file = self.ensure_marker_file(project)
        reader = self._build_reader(marker_file)
        reader.UpdateInformation()
        self._enable_all_arrays(reader)
        time_values = self._time_values(reader)
        reader.Update()
        output = reader.GetOutput()
        geometry = vtkCompositeDataGeometryFilter()
        geometry.SetInputDataObject(output)
        geometry.Update()
        poly_data = geometry.GetOutput()
        return OpenFoamVtkCaseInfo(
            marker_file=marker_file,
            time_values=time_values,
            block_count=output.GetNumberOfBlocks(),
            point_arrays=self._array_names(poly_data.GetPointData()),
            cell_arrays=self._array_names(poly_data.GetCellData()),
        )

    def build_geometry_filter(
        self,
        project: SimulationProject,
        time_value: float | None = None,
    ) -> vtkCompositeDataGeometryFilter:
        marker_file = self.ensure_marker_file(project)
        reader = self._build_reader(marker_file)
        reader.UpdateInformation()
        self._enable_all_arrays(reader)
        if time_value is not None:
            reader.SetTimeValue(time_value)
        geometry = vtkCompositeDataGeometryFilter()
        geometry.SetInputConnection(reader.GetOutputPort())
        if time_value is not None:
            geometry.UpdateTimeStep(time_value)
        else:
            geometry.Update()
        return geometry

    def build_case_output(
        self,
        project: SimulationProject,
        time_value: float | None = None,
    ):
        marker_file = self.ensure_marker_file(project)
        reader = self._build_reader(marker_file)
        reader.UpdateInformation()
        self._enable_all_arrays(reader)
        if time_value is not None:
            reader.SetTimeValue(time_value)
        if time_value is not None:
            reader.UpdateTimeStep(time_value)
        else:
            reader.Update()
        output = reader.GetOutputDataObject(0)
        output.Register(None)
        return output

    def build_patch_surfaces(
        self,
        project: SimulationProject,
        exclude_patches: tuple[str, ...] = (),
        include_patches: tuple[str, ...] | None = None,
        time_value: float | None = None,
    ) -> vtkPolyData:
        """Extract boundary patch surfaces from constant/polyMesh as a single
        polydata. By default returns all patches; pass include_patches to keep
        only specific ones or exclude_patches to drop named patches
        (case-insensitive). This reflects the *meshed* geometry, not the raw STL."""
        marker_file = self.ensure_marker_file(project)
        reader = self._build_reader(marker_file)
        reader.SetCreateCellToPoint(1)
        reader.UpdateInformation()
        reader.SetPatchArrayStatus("internalMesh", 0)
        patch_names: list[str] = []
        for index in range(reader.GetNumberOfPatchArrays()):
            name = reader.GetPatchArrayName(index)
            if name == "internalMesh":
                reader.SetPatchArrayStatus(name, 0)
                continue
            patch_names.append(name)
            reader.SetPatchArrayStatus(name, 1)
        if time_value is not None:
            reader.SetTimeValue(time_value)
        reader.Update()
        output = reader.GetOutputDataObject(0)

        exclude_lower = {p.lower() for p in exclude_patches}
        include_lower = {p.lower() for p in include_patches} if include_patches else None

        append = vtkAppendPolyData()
        piece_count = 0
        iterator = output.NewIterator()
        iterator.InitTraversal()
        while not iterator.IsDoneWithTraversal():
            block = iterator.GetCurrentDataObject()
            meta = iterator.GetCurrentMetaData()
            block_name = ""
            if meta is not None and meta.Has(vtkCompositeDataSet.NAME()):
                block_name = str(meta.Get(vtkCompositeDataSet.NAME()))
            leaf = block_name.split("/")[-1].lower()
            iterator.GoToNextItem()
            if block is None:
                continue
            if include_lower is not None and leaf not in include_lower:
                continue
            if leaf in exclude_lower:
                continue
            geometry_filter = vtkGeometryFilter()
            geometry_filter.SetInputData(block)
            geometry_filter.Update()
            piece = geometry_filter.GetOutput()
            if piece is not None and piece.GetNumberOfPoints() > 0:
                append.AddInputData(piece)
                piece_count += 1
        if piece_count == 0:
            return vtkPolyData()
        append.Update()
        return append.GetOutput()

    def ensure_marker_file(self, project: SimulationProject) -> Path:
        marker_file = project.case_dir / self.MARKER_FILE_NAME
        marker_file.touch(exist_ok=True)
        return marker_file

    def _ensure_mesh_exists(self, project: SimulationProject) -> None:
        mesh_dir = project.case_dir / "constant" / "polyMesh"
        if not (mesh_dir / "points").exists():
            raise RuntimeError("当前 case 还没有生成网格，请先在“网格生成”页面点击“生成网格”。")

    def _build_reader(self, marker_file: Path) -> vtkOpenFOAMReader:
        reader = vtkOpenFOAMReader()
        reader.SetFileName(str(marker_file))
        return reader

    def _enable_all_arrays(self, reader: vtkOpenFOAMReader) -> None:
        for index in range(reader.GetNumberOfCellArrays()):
            reader.SetCellArrayStatus(reader.GetCellArrayName(index), 1)
        for index in range(reader.GetNumberOfPointArrays()):
            reader.SetPointArrayStatus(reader.GetPointArrayName(index), 1)

    def _time_values(self, reader: vtkOpenFOAMReader) -> list[float]:
        values = reader.GetTimeValues()
        if values is None:
            return []
        return [values.GetValue(index) for index in range(values.GetNumberOfTuples())]

    def _array_names(self, data) -> list[str]:
        return [data.GetArrayName(index) for index in range(data.GetNumberOfArrays())]
