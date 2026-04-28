from __future__ import annotations

from pathlib import Path

from vtkmodules.vtkFiltersGeometry import vtkCompositeDataGeometryFilter
from vtkmodules.vtkIOGeometry import vtkOpenFOAMReader

from foamdesk.domain.models import OpenFoamVtkCaseInfo, SimulationProject


class OpenFoamVtkService:
    """Loads OpenFOAM case data through VTK's native OpenFOAM reader."""

    MARKER_FILE_NAME = "foamdesk.foam"

    def inspect(self, project: SimulationProject) -> OpenFoamVtkCaseInfo:
        marker_file = self.ensure_marker_file(project)
        reader = self._build_reader(marker_file)
        reader.UpdateInformation()
        time_values = self._time_values(reader)
        reader.Update()
        output = reader.GetOutput()
        return OpenFoamVtkCaseInfo(
            marker_file=marker_file,
            time_values=time_values,
            block_count=output.GetNumberOfBlocks(),
        )

    def build_geometry_filter(self, project: SimulationProject) -> vtkCompositeDataGeometryFilter:
        marker_file = self.ensure_marker_file(project)
        reader = self._build_reader(marker_file)
        geometry = vtkCompositeDataGeometryFilter()
        geometry.SetInputConnection(reader.GetOutputPort())
        geometry.Update()
        return geometry

    def ensure_marker_file(self, project: SimulationProject) -> Path:
        marker_file = project.case_dir / self.MARKER_FILE_NAME
        marker_file.touch(exist_ok=True)
        return marker_file

    def _build_reader(self, marker_file: Path) -> vtkOpenFOAMReader:
        reader = vtkOpenFOAMReader()
        reader.SetFileName(str(marker_file))
        return reader

    def _time_values(self, reader: vtkOpenFOAMReader) -> list[float]:
        values = reader.GetTimeValues()
        if values is None:
            return []
        return [values.GetValue(index) for index in range(values.GetNumberOfTuples())]
