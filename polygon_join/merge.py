"""Coplanar polygon merging.

Polygons that lie on (nearly) the same plane cannot be joined through
plane-plane intersection: the intersection line of parallel planes is
undefined or numerically unstable. When such polygons touch or overlap
within the join tolerance, the correct operation is a 2D boolean union.

Small gaps between members are bridged with a morphological closing
(buffer outward by tol/2, union, buffer inward by tol/2) using mitred
offsets so rectangular corners stay sharp. Members whose union is still
disconnected after closing remain separate polygons.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from .model import PlanarPolygon, fit_plane, plane_frame


def _shape_2d(pts2):
    shape = ShapelyPolygon(pts2)
    if not shape.is_valid:
        shape = shape.buffer(0)
    return shape


def _plane_offset(a: PlanarPolygon, b: PlanarPolygon) -> float:
    """Largest distance from either polygon's vertices to the other's plane."""
    da = float(np.max(np.abs((b.vertices - a.origin) @ a.normal)))
    db = float(np.max(np.abs((a.vertices - b.origin) @ b.normal)))
    return max(da, db)


def merge_coplanar_polygons(polys, tol, cos_parallel_min, simplify_tol=1e-7):
    """Merge groups of coplanar polygons whose boundaries touch within tol.

    polys:            list of flattened PlanarPolygon
    tol:              join tolerance; used as both the maximum plane offset
                      and the maximum 2D boundary gap between group members
    cos_parallel_min: pairs with |n_i . n_j| above this value are treated as
                      parallel (same threshold that excludes them from the
                      plane-plane intersection path)
    simplify_tol:     tolerance for removing redundant union vertices

    Returns (new polygon list, merge report list). Polygons that belong to
    no merge group are passed through unchanged, in input order.
    """
    n = len(polys)
    parent = list(range(n))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = polys[i], polys[j]
            if abs(float(pi.normal @ pj.normal)) < cos_parallel_min:
                continue  # not parallel: handled by the intersection path
            if _plane_offset(pi, pj) > tol:
                continue  # parallel but on distinct planes (e.g. floor/ceiling)
            si = _shape_2d(pi.to_2d(pi.vertices))
            sj = _shape_2d(pi.to_2d(pj.vertices))  # both in i's frame
            if si.distance(sj) > tol:
                continue  # coplanar but too far apart to be one surface
            parent[find(i)] = find(j)

    groups = defaultdict(list)
    for k in range(n):
        groups[find(k)].append(k)

    result, report = [], []
    for key in sorted(groups, key=lambda g: min(groups[g])):
        members = groups[key]
        if len(members) == 1:
            result.append(polys[members[0]])
            continue

        group = [polys[k] for k in members]
        origin, normal = fit_plane(np.vstack([g.vertices for g in group]))
        if float(normal @ group[0].normal) < 0.0:
            normal = -normal  # keep the first member's orientation

        u, v = plane_frame(normal)

        def to_2d(pts):
            rel = pts - origin
            return np.column_stack([rel @ u, rel @ v])

        shapes = [_shape_2d(to_2d(g.vertices)) for g in group]
        # Morphological closing: bridges boundary gaps up to ~tol wide.
        closed = unary_union([s.buffer(tol / 2.0, join_style=2) for s in shapes])
        closed = closed.buffer(-tol / 2.0, join_style=2)

        parts = list(closed.geoms) if isinstance(closed, MultiPolygon) else [closed]
        parts = [p for p in parts if isinstance(p, ShapelyPolygon) and not p.is_empty]
        if not parts:  # numerically degenerate: keep the originals
            result.extend(group)
            continue

        base_name = "+".join(g.name for g in group)
        for part_idx, part in enumerate(parts):
            part = orient(part.simplify(simplify_tol), 1.0)  # CCW == +normal winding
            ring = np.asarray(part.exterior.coords[:-1], dtype=float)
            verts3 = origin + np.outer(ring[:, 0], u) + np.outer(ring[:, 1], v)
            name = base_name if len(parts) == 1 else f"{base_name}_{part_idx}"
            result.append(PlanarPolygon(name, verts3))
            report.append({
                "result": name,
                "members": [g.name for g in group],
                "dropped_holes": len(part.interiors),
            })
    return result, report
