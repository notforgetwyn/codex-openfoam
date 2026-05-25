# FoamDesk Test STL Geometries

This folder contains simple ASCII STL files for FoamDesk manual testing.

## Domain Templates

FoamDesk will use these domain sizes as the first built-in presets:

- Simple domain: `1 x 1 x 1`
- Medium wind tunnel: `4 x 2 x 2`
- Advanced long wind tunnel: `10 x 4 x 4`

In CFD terms, the domain is the fluid box. The STL is the solid obstacle inside that fluid box.

## STL Files

### small_obstacle_cube.stl

- Closed ASCII STL cube.
- Bounds: `(0.65, 0.65, 0.65)` to `(0.85, 0.85, 0.85)`.
- Best domain: simple `1 x 1 x 1`.
- Suggested `locationInMesh`: `(0.5, 0.5, 0.5)`.
- Use it to test STL import, snappyHexMeshDict generation, snappyHexMesh, checkMesh, and one-click pipelines.

### simple_center_cube.stl

- Closed ASCII STL cube.
- Bounds: `(0.40, 0.40, 0.40)` to `(0.60, 0.60, 0.60)`.
- Best domain: simple `1 x 1 x 1`.
- Suggested `locationInMesh`: `(0.2, 0.2, 0.2)` or another point outside the cube but inside the fluid box.
- Use it to test the smallest solid obstacle in the unit cube domain.

### medium_cylinder_obstacle.stl

- Closed ASCII STL cylinder aligned with the X axis.
- Bounds: approximately `(1.55, 0.72, 0.72)` to `(2.45, 1.28, 1.28)`.
- Best domain: medium wind tunnel `4 x 2 x 2`.
- Suggested `locationInMesh`: `(0.5, 1.0, 1.0)`.
- Use it to test a pipe-like or round obstacle in a small wind tunnel.

### medium_ramp_wedge.stl

- Closed ASCII STL wedge/ramp.
- Bounds: `(1.55, 0.65, 0.15)` to `(2.45, 1.35, 0.75)`.
- Best domain: medium wind tunnel `4 x 2 x 2`.
- Suggested `locationInMesh`: `(0.5, 1.0, 1.0)`.
- Use it to test an inclined obstacle with a simple non-box shape.

### advanced_simplified_vehicle.stl

- Closed ASCII STL simplified vehicle shape.
- Includes a body box, cabin box, and four simplified wheel cylinders.
- Bounds: approximately `(4.0, 1.39, 0.29)` to `(6.2, 2.68, 1.45)`.
- Best domain: advanced long wind tunnel `10 x 4 x 4`.
- Suggested `locationInMesh`: `(1.0, 2.0, 2.0)`.
- Use it to test a larger external-flow scenario similar to wind over a vehicle.
