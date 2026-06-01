"""Computational-domain templates that generate blockMeshDict (本期: 长方体 / 圆柱 / 风洞).

Pure geometry + dict generation, no Qt — so it can be unit-tested and validated
with a real `blockMesh` run headless. The mesh-generation page picks a template,
collects parameters + a face table (name/type/U/p per logical face), and asks the
template to emit a complete blockMeshDict with correctly named boundary patches.
"""

from __future__ import annotations

import numpy as np

BOUNDARY_TYPE_OPTIONS = ["patch", "wall", "symmetry", "empty", "cyclic"]

_HEADER = "FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }\n"


def _fmt(value: float) -> str:
    return f"{float(value):.6g}"


def _vertices_block(vertices: list[tuple[float, float, float]]) -> str:
    lines = "".join(f"    ({_fmt(p[0])} {_fmt(p[1])} {_fmt(p[2])})\n" for p in vertices)
    return f"vertices\n(\n{lines});\n\n"


def _boundary_block(faces: list[dict], quad_map: dict[str, list[str]]) -> str:
    """Merge logical faces by (name, type) and emit the boundary section."""
    merged: dict[tuple[str, str], list[str]] = {}
    order: list[tuple[str, str]] = []
    for face in faces:
        name = (face.get("name") or face.get("type") or "patch").strip()
        btype = face.get("type", "patch")
        key = (name, btype)
        if key not in merged:
            merged[key] = []
            order.append(key)
        quads = quad_map.get(face["label"], [])
        if isinstance(quads, str):
            quads = [quads]
        merged[key].extend(quads)
    out = "boundary\n(\n"
    for name, btype in order:
        faces_str = " ".join(f"({q})" for q in merged[(name, btype)])
        out += f"    {name}\n    {{\n        type {btype};\n        faces ({faces_str});\n    }}\n"
    out += ");\n"
    return out


# ──────────────────────────────────────────────────────────────────────────
# Box (轴对齐长方体) — refactor of the original hard-coded generator
# ──────────────────────────────────────────────────────────────────────────

# Logical face -> blockMesh vertex quad (matches the original DOMAIN_FACE_DEFAULTS winding).
_BOX_FACE_QUADS = {
    "X-min": "0 4 7 3",
    "X-max": "1 2 6 5",
    "Y-min": "0 1 2 3",
    "Y-max": "4 5 6 7",
    "Z-min": "0 1 5 4",
    "Z-max": "3 2 6 7",
}

BOX_DEFAULT_FACES = [
    {"label": "X-min", "name": "inlet", "type": "patch"},
    {"label": "X-max", "name": "outlet", "type": "patch"},
    {"label": "Y-min", "name": "walls", "type": "wall"},
    {"label": "Y-max", "name": "walls", "type": "wall"},
    {"label": "Z-min", "name": "walls", "type": "wall"},
    {"label": "Z-max", "name": "walls", "type": "wall"},
]

# Nozzle / diffuser — same hex topology as the box, but the inlet/outlet
# half-heights differ so the duct tapers (converging or diverging).
NOZZLE_DEFAULT_FACES = [
    {"label": "X-min", "name": "inlet", "type": "patch"},
    {"label": "X-max", "name": "outlet", "type": "patch"},
    {"label": "Y-min", "name": "walls", "type": "wall"},
    {"label": "Y-max", "name": "walls", "type": "wall"},
    {"label": "Z-min", "name": "walls", "type": "wall"},
    {"label": "Z-max", "name": "walls", "type": "wall"},
]

CAD_BACKGROUND_FACES = [
    {"label": "X-min", "name": "background", "type": "patch"},
    {"label": "X-max", "name": "background", "type": "patch"},
    {"label": "Y-min", "name": "background", "type": "patch"},
    {"label": "Y-max", "name": "background", "type": "patch"},
    {"label": "Z-min", "name": "background", "type": "patch"},
    {"label": "Z-max", "name": "background", "type": "patch"},
]

CAD_DEFAULT_FACES: list[dict] = []


def box_corners(params: dict) -> np.ndarray:
    x0, x1 = params["x_min"], params["x_max"]
    y0, y1 = params["y_min"], params["y_max"]
    z0, z1 = params["z_min"], params["z_max"]
    return np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=float,
    )


def nozzle_corners(params: dict) -> np.ndarray:
    """8 corners of a tapered duct: flow along +X, taper in Y, depth in Z."""
    length = float(params["length"])
    width = float(params["width"])
    hin = float(params["h_in"]) / 2.0
    hout = float(params["h_out"]) / 2.0
    points = np.array(
        [
            [0.0, -hin, 0.0], [length, -hout, 0.0], [length, hout, 0.0], [0.0, hin, 0.0],
            [0.0, -hin, width], [length, -hout, width], [length, hout, width], [0.0, hin, width],
        ],
        dtype=float,
    )
    return points + _center_offset(params, points)


def template_corners(key: str, params: dict) -> np.ndarray:
    """8 hex corners for box-topology templates (box / nozzle), for preview + highlight."""
    if key == "nozzle":
        return nozzle_corners(params)
    return box_corners(params)


def _build_hex_dict(corners: np.ndarray, nx: int, ny: int, nz: int,
                    faces: list[dict], convert: float) -> str:
    text = _HEADER + f"convertToMeters {_fmt(convert)};\n\n"
    text += _vertices_block([tuple(c) for c in corners])
    text += (
        "blocks\n(\n"
        f"    hex (0 1 2 3 4 5 6 7) ({int(nx)} {int(ny)} {int(nz)}) simpleGrading (1 1 1)\n"
        ");\n\n"
        "edges\n(\n);\n\n"
    )
    text += _boundary_block(faces, _BOX_FACE_QUADS)
    text += "\nmergePatchPairs\n(\n);\n"
    return text


def build_box_dict(params: dict, faces: list[dict], convert: float) -> str:
    return _build_hex_dict(box_corners(params), params["cells_x"], params["cells_y"],
                           params["cells_z"], faces, convert)


def build_nozzle_dict(params: dict, faces: list[dict], convert: float) -> str:
    return _build_hex_dict(nozzle_corners(params), params["cells_x"], params["cells_y"],
                           params["cells_z"], faces, convert)


def build_cad_background_dict(params: dict, _faces: list[dict], convert: float) -> str:
    """Build a simple background box around imported CAD-domain STL patches.

    The CAD domain surfaces themselves become snappyHexMesh refinement surfaces;
    blockMesh still needs a background mesh, so we emit a padded box with one
    background patch instead of using the STL patch names here.
    """
    required = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
    if not all(k in params for k in required):
        params = {
            "x_min": -1.0, "x_max": 1.0,
            "y_min": -1.0, "y_max": 1.0,
            "z_min": -1.0, "z_max": 1.0,
            "cells_x": 20, "cells_y": 20, "cells_z": 20,
        }
    return _build_hex_dict(
        box_corners(params),
        int(params.get("cells_x", 20)),
        int(params.get("cells_y", 20)),
        int(params.get("cells_z", 20)),
        CAD_BACKGROUND_FACES,
        convert,
    )


# ──────────────────────────────────────────────────────────────────────────
# Cylinder (O-grid)
# ──────────────────────────────────────────────────────────────────────────

CYLINDER_DEFAULT_FACES = [
    {"label": "进口端面", "name": "inlet", "type": "patch"},
    {"label": "出口端面", "name": "outlet", "type": "patch"},
    {"label": "圆柱壁面", "name": "walls", "type": "wall"},
]

_CYL_ANGLES_DEG = [45.0, 135.0, 225.0, 315.0]
_CYL_MID_DEG = [90.0, 180.0, 270.0, 0.0]  # arc midpoints between consecutive outer points


def _axis_transform(point: tuple[float, float, float], axis: str) -> tuple[float, float, float]:
    """Map a local (x, y, z=axial) point to global coords; proper rotation (det +1)."""
    x, y, z = point
    if axis == "X":
        return (z, x, y)
    if axis == "Y":
        return (y, z, x)
    return (x, y, z)  # Z


def _center_offset(params: dict, local_points: np.ndarray) -> np.ndarray:
    if not all(k in params for k in ("center_x", "center_y", "center_z")):
        return np.zeros(3, dtype=float)
    mins = local_points.min(axis=0)
    maxs = local_points.max(axis=0)
    current = (mins + maxs) * 0.5
    target = np.array([float(params["center_x"]), float(params["center_y"]), float(params["center_z"])], dtype=float)
    return target - current


def _cylinder_raw_outer_points(radius: float, length: float, axis: str, n: int = 32) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    return np.array(
        [_axis_transform((radius * np.cos(t), radius * np.sin(t), zc), axis) for zc in (0.0, length) for t in theta],
        dtype=float,
    )


def cylinder_vertices(params: dict) -> list[tuple[float, float, float]]:
    radius = float(params["radius"])
    length = float(params["length"])
    axis = params.get("axis", "Z")
    inner = radius * 0.5
    angles = np.deg2rad(_CYL_ANGLES_DEG)
    verts: list[tuple[float, float, float]] = []
    for zc in (0.0, length):
        for a in angles:  # inner square corners
            verts.append(_axis_transform((inner * np.cos(a), inner * np.sin(a), zc), axis))
        for a in angles:  # outer circle points
            verts.append(_axis_transform((radius * np.cos(a), radius * np.sin(a), zc), axis))
    points = np.array(verts, dtype=float)
    offset = _center_offset(params, _cylinder_raw_outer_points(radius, length, axis))
    return [tuple(p) for p in points + offset]


def _cylinder_quad_map() -> dict[str, list[str]]:
    inlet = ["0 1 2 3"] + [f"{k} {(k + 1) % 4} {4 + (k + 1) % 4} {4 + k}" for k in range(4)]
    outlet = ["8 9 10 11"] + [f"{8 + k} {8 + (k + 1) % 4} {12 + (k + 1) % 4} {12 + k}" for k in range(4)]
    walls = [f"{4 + k} {4 + (k + 1) % 4} {12 + (k + 1) % 4} {12 + k}" for k in range(4)]
    return {"进口端面": inlet, "出口端面": outlet, "圆柱壁面": walls}


def build_cylinder_dict(params: dict, faces: list[dict], convert: float) -> str:
    radius = float(params["radius"])
    length = float(params["length"])
    axis = params.get("axis", "Z")
    n_circ = int(params.get("n_circ", 10))
    n_radial = int(params.get("n_radial", 6))
    n_axial = int(params.get("n_axial", 20))

    verts = cylinder_vertices(params)
    text = _HEADER + f"convertToMeters {_fmt(convert)};\n\n"
    text += _vertices_block(verts)

    # blocks: 1 center + 4 outer
    text += "blocks\n(\n"
    text += f"    hex (0 1 2 3 8 9 10 11) ({n_circ} {n_circ} {n_axial}) simpleGrading (1 1 1)\n"
    for k in range(4):
        nx = (k + 1) % 4
        # bottom face wound CCW (normal toward +axis / top layer) so the hex is not inside-out
        block = f"{k} {4 + k} {4 + nx} {nx} {8 + k} {12 + k} {12 + nx} {8 + nx}"
        text += f"    hex ({block}) ({n_radial} {n_circ} {n_axial}) simpleGrading (1 1 1)\n"
    text += ");\n\n"

    # arc edges on the outer circle (bottom + top)
    mids = np.deg2rad(_CYL_MID_DEG)
    text += "edges\n(\n"
    for side, zc in enumerate((0.0, length)):
        base = 4 + side * 8
        for k in range(4):
            nx = (k + 1) % 4
            mp = np.array(_axis_transform((radius * np.cos(mids[k]), radius * np.sin(mids[k]), zc), axis), dtype=float)
            mp = mp + _center_offset(params, _cylinder_raw_outer_points(radius, length, axis))
            text += f"    arc {base + k} {base + nx} ({_fmt(mp[0])} {_fmt(mp[1])} {_fmt(mp[2])})\n"
    text += ");\n\n"

    text += _boundary_block(faces, _cylinder_quad_map())
    text += "\nmergePatchPairs\n(\n);\n"
    return text


def cylinder_preview_polylines(params: dict) -> list[np.ndarray]:
    """World-coordinate polylines for drawing a cylinder preview (2 end circles + axial lines)."""
    radius = float(params["radius"])
    length = float(params["length"])
    axis = params.get("axis", "Z")
    theta = np.linspace(0.0, 2.0 * np.pi, 64)
    offset = _center_offset(params, _cylinder_raw_outer_points(radius, length, axis, n=64))
    lines: list[np.ndarray] = []
    for zc in (0.0, length):
        circle = np.array(
            [_axis_transform((radius * np.cos(t), radius * np.sin(t), zc), axis) for t in theta],
            dtype=float,
        ) + offset
        lines.append(circle)
    for a in np.deg2rad([0.0, 90.0, 180.0, 270.0]):
        p0 = np.array(_axis_transform((radius * np.cos(a), radius * np.sin(a), 0.0), axis), dtype=float) + offset
        p1 = np.array(_axis_transform((radius * np.cos(a), radius * np.sin(a), length), axis), dtype=float) + offset
        lines.append(np.array([p0, p1], dtype=float))
    return lines


def cylinder_end_centers(params: dict) -> tuple[tuple, tuple]:
    """Return (inlet_center, outlet_center) in world coords."""
    length = float(params["length"])
    axis = params.get("axis", "Z")
    radius = float(params.get("radius", 1.0))
    offset = _center_offset(params, _cylinder_raw_outer_points(radius, length, axis))
    raw = np.array([_axis_transform((0.0, 0.0, 0.0), axis), _axis_transform((0.0, 0.0, length), axis)], dtype=float)
    shifted = raw + offset
    return (tuple(shifted[0]), tuple(shifted[1]))


def cylinder_end_circle_points(params: dict, at_outlet: bool, n: int = 64) -> np.ndarray:
    """Closed ring of world-coordinate points on one end face of the cylinder."""
    radius = float(params["radius"])
    length = float(params["length"])
    axis = params.get("axis", "Z")
    zc = length if at_outlet else 0.0
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    raw = np.array(
        [_axis_transform((radius * np.cos(t), radius * np.sin(t), zc), axis) for t in theta],
        dtype=float,
    )
    return raw + _center_offset(params, _cylinder_raw_outer_points(radius, length, axis, n=n))


def cylinder_axis_direction(params: dict) -> np.ndarray:
    """Unit vector pointing from inlet (z=0) toward outlet (z=L) in world coords."""
    axis = params.get("axis", "Z")
    tip = np.array(_axis_transform((0.0, 0.0, 1.0), axis), dtype=float)
    base = np.array(_axis_transform((0.0, 0.0, 0.0), axis), dtype=float)
    direction = tip - base
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])


# ──────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────

# param kind: ("bounds" = the existing xmin..zmax + cells panel, "cylinder" = R/L/axis panel)
DOMAIN_TEMPLATES = {
    "box": {
        "label": "长方体 Box",
        "param_kind": "bounds",
        "default_faces": BOX_DEFAULT_FACES,
        "builder": build_box_dict,
    },
    "cylinder": {
        "label": "圆柱 Cylinder",
        "param_kind": "cylinder",
        "default_faces": CYLINDER_DEFAULT_FACES,
        "builder": build_cylinder_dict,
    },
    "nozzle": {
        "label": "渐变通道（喷管/扩压器）",
        "param_kind": "nozzle",
        "default_faces": NOZZLE_DEFAULT_FACES,
        "builder": build_nozzle_dict,
    },
    "cad": {
        "label": "CAD 导入计算域",
        "param_kind": "cad",
        "default_faces": CAD_DEFAULT_FACES,
        "builder": build_cad_background_dict,
    },
}


def template_default_faces(key: str) -> list[dict]:
    spec = DOMAIN_TEMPLATES.get(key, DOMAIN_TEMPLATES["box"])
    return [dict(f) for f in spec["default_faces"]]


def build_block_mesh(key: str, params: dict, faces: list[dict], convert: float) -> str:
    spec = DOMAIN_TEMPLATES.get(key, DOMAIN_TEMPLATES["box"])
    return spec["builder"](params, faces, convert)
