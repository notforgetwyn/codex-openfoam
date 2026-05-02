# Test STL Geometries

This folder contains simple STL files for FoamDesk manual testing.

## small_obstacle_cube.stl

- Closed ASCII STL cube.
- Bounds: `(0.65, 0.65, 0.65)` to `(0.85, 0.85, 0.85)`.
- Intended default fluid point: `locationInMesh = (0.5, 0.5, 0.5)`.
- Use it to test: STL import, `snappyHexMeshDict` generation, `snappyHexMesh`, `checkMesh`, and the one-click pipelines.
