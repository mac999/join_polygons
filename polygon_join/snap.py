"""Core polygon joining algorithm.

Pipeline:
 1. Fit a least-squares plane (SVD) to every polygon and project its
    vertices onto that plane.
 2. Merge coplanar polygons whose boundaries touch within the join
    tolerance (same-plane contacts are a 2D union problem, not a
    plane-plane intersection problem) -- see merge.py.
 3. For every remaining polygon pair, intersect the two fitted planes.
    The pair is adjacent when both boundaries come within join_tolerance
    of the intersection line AND their near-line spans overlap along the
    line (this rejects polygons that touch the infinite line at disjoint
    locations).
 4. Snap each polygon in its own 2D plane frame:
    - insert new vertices where two neighbor lines meet inside an edge,
      so N:1 contacts (e.g. several wall planes meeting one floor edge)
      can reshape the shared polygon; its vertex count may grow
    - a vertex near one line is projected onto it (extend or trim)
    - a vertex near two lines snaps to the line-line intersection, which
      is the triple-plane corner point shared by three polygons
 5. Optionally clip material protruding past a neighbor line, restricted
    to the shared adjacency span so that non-convex polygons keep parts
    that legitimately extend beyond the (infinite) line elsewhere.
 6. Remove duplicate and collinear vertices.
"""
from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations

import numpy as np
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon

from .merge import merge_coplanar_polygons
from .model import PlanarPolygon


# ------------------------------------------------------------ geometry utils

def plane_plane_line(o1, n1, o2, n2, parallel_eps=1e-9):
    """Intersection line of two planes as (point, unit direction), or None."""
    direction = np.cross(n1, n2)
    norm = np.linalg.norm(direction)
    if norm < parallel_eps:
        return None
    direction = direction / norm
    d1, d2 = float(n1 @ o1), float(n2 @ o2)
    c = float(n1 @ n2)
    denom = 1.0 - c * c
    a = (d1 - d2 * c) / denom
    b = (d2 - d1 * c) / denom
    point = a * n1 + b * n2
    return point, direction


def segment_line_distance(a, b, line_pt, line_dir):
    """Minimum distance between 3D segment [a, b] and an infinite line."""
    proj = np.eye(3) - np.outer(line_dir, line_dir)
    w0 = proj @ (a - line_pt)
    d = proj @ (b - a)
    dd = float(d @ d)
    if dd < 1e-16:
        return float(np.linalg.norm(w0))
    s = float(np.clip(-(w0 @ d) / dd, 0.0, 1.0))
    return float(np.linalg.norm(w0 + s * d))


def polygon_line_distance(vertices, line_pt, line_dir):
    """Minimum distance between a closed polygon boundary and a line."""
    n = len(vertices)
    return min(
        segment_line_distance(vertices[i], vertices[(i + 1) % n], line_pt, line_dir)
        for i in range(n)
    )


def segment_segment_distance(p1, q1, p2, q2):
    """Minimum distance between two 3D segments (standard clamping)."""
    d1, d2 = q1 - p1, q2 - p2
    r = p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a < 1e-16 and e < 1e-16:
        return float(np.linalg.norm(r))
    if a < 1e-16:
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = float(d1 @ r)
        if e < 1e-16:
            t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > 1e-16 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    closest1 = p1 + s * d1
    closest2 = p2 + t * d2
    return float(np.linalg.norm(closest1 - closest2))


def boundary_boundary_distance(verts_a, verts_b):
    """Minimum distance between two polygon boundaries."""
    na, nb = len(verts_a), len(verts_b)
    best = math.inf
    for i in range(na):
        a1, a2 = verts_a[i], verts_a[(i + 1) % na]
        for j in range(nb):
            b1, b2 = verts_b[j], verts_b[(j + 1) % nb]
            best = min(best, segment_segment_distance(a1, a2, b1, b2))
    return best


def _near_span(vertices, line_pt, line_dir, tol):
    """Parameter span (tmin, tmax) along the line where the polygon boundary
    lies within tol of the line, or None if it never comes that close.

    For each boundary segment, the sub-interval within tol is found by
    solving |perp(p(s))|^2 <= tol^2, a quadratic in the segment parameter.
    """
    proj = np.eye(3) - np.outer(line_dir, line_dir)
    n = len(vertices)
    tmin, tmax = math.inf, -math.inf
    for i in range(n):
        a, b = vertices[i], vertices[(i + 1) % n]
        w0 = proj @ (a - line_pt)
        wd = proj @ (b - a)
        qa = float(wd @ wd)
        qb = 2.0 * float(w0 @ wd)
        qc = float(w0 @ w0) - tol * tol
        if qa < 1e-16:
            if qc > 0.0:
                continue
            s0, s1 = 0.0, 1.0
        else:
            disc = qb * qb - 4.0 * qa * qc
            if disc < 0.0:
                continue
            root = math.sqrt(disc)
            s0 = max(0.0, (-qb - root) / (2.0 * qa))
            s1 = min(1.0, (-qb + root) / (2.0 * qa))
            if s0 > s1:
                continue
        for s in (s0, s1):
            t = float(line_dir @ (a + s * (b - a) - line_pt))
            tmin = min(tmin, t)
            tmax = max(tmax, t)
    if tmin > tmax:
        return None
    return tmin, tmax


# ----------------------------------------------------------- 2D line helpers

def _line_to_2d(poly: PlanarPolygon, line_pt, line_dir):
    """Express a 3D line lying in the polygon's plane in its 2D frame.

    The plane frame is orthonormal, so the line parameter t measured along
    the 3D unit direction carries over to the 2D representation unchanged.
    """
    p2 = poly.to_2d(line_pt.reshape(1, 3))[0]
    q2 = poly.to_2d((line_pt + line_dir).reshape(1, 3))[0]
    d2 = q2 - p2
    d2 = d2 / np.linalg.norm(d2)
    return p2, d2


def _dist_2d(pt, line2):
    p2, d2 = line2
    rel = pt - p2
    return abs(d2[0] * rel[1] - d2[1] * rel[0])


def _intersect_2d(l1, l2, parallel_eps=1e-9):
    (p1, d1), (p2, d2) = l1, l2
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < parallel_eps:
        return None
    rel = p2 - p1
    t = (rel[0] * d2[1] - rel[1] * d2[0]) / cross
    return p1 + t * d1


def _active(pt, line2, span, ext):
    """True when pt projects inside the line's adjacency span (with slack)."""
    p2, d2 = line2
    t = float((pt - p2) @ d2)
    return span[0] - ext <= t <= span[1] + ext


# ------------------------------------------------- refine / snap / clip / tidy

def _corner_candidates(constraints, ext, dedupe_tol):
    """Intersection points of neighbor-line pairs near their adjacency spans.

    These are the triple-plane corner points the polygon boundary should
    pass through.
    """
    corners = []
    for (l1, s1), (l2, s2) in combinations(constraints, 2):
        c = _intersect_2d(l1, l2)
        if c is None:
            continue
        if not (_active(c, l1, s1, ext) and _active(c, l2, s2, ext)):
            continue
        if all(np.linalg.norm(c - prev) > dedupe_tol for prev in corners):
            corners.append(c)
    return corners


def _insert_corner_vertices_2d(pts2, corners, tol):
    """Insert corner points into the edges they pass through.

    This handles N:1 contacts: when several neighbor planes meet along one
    edge of this polygon, the edge must gain a vertex at each plane-to-plane
    transition. Corners already represented by a nearby endpoint are skipped;
    the endpoint itself will snap onto the corner instead.
    """
    if not corners:
        return pts2
    out = []
    n = len(pts2)
    for k in range(n):
        a, b = pts2[k], pts2[(k + 1) % n]
        out.append(a)
        d = b - a
        length2 = float(d @ d)
        if length2 < 1e-16:
            continue
        cand = []
        for c in corners:
            t = float((c - a) @ d) / length2
            if not 0.0 < t < 1.0:
                continue
            if np.linalg.norm(c - (a + t * d)) > tol:
                continue
            if np.linalg.norm(c - a) <= tol or np.linalg.norm(c - b) <= tol:
                continue
            cand.append((t, c))
        out.extend(c for _, c in sorted(cand, key=lambda x: x[0]))
    return np.asarray(out)


def _snap_vertices_2d(pts2, constraints, tol, corner_snap, corner_slack, ext):
    """Snap vertices to nearby active lines (1 line: project, 2+: corner)."""
    snapped = pts2.copy()
    for k, pt in enumerate(pts2):
        near = sorted(
            ((dist, line2) for line2, span in constraints
             if (dist := _dist_2d(pt, line2)) <= tol and _active(pt, line2, span, ext)),
            key=lambda x: x[0],
        )
        if not near:
            continue
        target = None
        if corner_snap and len(near) >= 2:
            base = near[0][1]
            for _, other in near[1:]:
                corner = _intersect_2d(base, other)
                if corner is not None and np.linalg.norm(corner - pt) <= tol * corner_slack:
                    target = corner
                    break
        if target is None:
            p2, d2 = near[0][1]
            rel = pt - p2
            target = p2 + (rel @ d2) * d2
        snapped[k] = target
    return snapped


def _signed_area(pts2):
    x, y = pts2[:, 0], pts2[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _clip_overshoot_2d(pts2, constraints, eps):
    """Remove material protruding past each neighbor line.

    The cut is bounded to the shared adjacency span along the line, so a
    non-convex polygon (e.g. an L-shaped merged floor) keeps the parts
    that legitimately lie beyond the infinite line outside the contact
    region. The side to keep is the one holding more polygon area.
    """
    shape = ShapelyPolygon(pts2)
    if not shape.is_valid:
        shape = shape.buffer(0)
    if shape.is_empty:
        return pts2
    depth = float(max(pts2.max(axis=0) - pts2.min(axis=0))) * 10.0 + 10.0
    for (p2, d2), (t0, t1) in constraints:
        if t1 - t0 < 1e-9:
            continue  # point-contact pair: nothing meaningful to clip
        normal = np.array([-d2[1], d2[0]])
        a0, a1 = p2 + d2 * t0, p2 + d2 * t1

        def side_rect(side):
            # eps offset keeps vertices snapped exactly onto the line alive
            b0, b1 = a0 + side * normal * eps, a1 + side * normal * eps
            return ShapelyPolygon([b0, b1,
                                   b1 + side * normal * depth,
                                   b0 + side * normal * depth])

        pos, neg = side_rect(1.0), side_rect(-1.0)
        overshoot = neg if shape.intersection(pos).area >= shape.intersection(neg).area else pos
        clipped = shape.difference(overshoot)
        if isinstance(clipped, MultiPolygon):
            clipped = max(clipped.geoms, key=lambda g: g.area)
        if isinstance(clipped, ShapelyPolygon) and not clipped.is_empty:
            shape = clipped
    out = np.asarray(shape.exterior.coords[:-1], dtype=float)
    if _signed_area(out) * _signed_area(pts2) < 0.0:
        out = out[::-1]  # shapely may flip the ring; restore input winding
    return out


def _dedupe(pts2, tol):
    """Remove consecutive duplicate vertices."""
    keep = []
    for pt in pts2:
        if not keep or np.linalg.norm(pt - keep[-1]) > tol:
            keep.append(pt)
    if len(keep) > 1 and np.linalg.norm(keep[0] - keep[-1]) <= tol:
        keep.pop()
    return np.asarray(keep)


def _remove_collinear(pts2, sin_tol):
    """Drop vertices whose incident edges continue straight ahead.

    Only forward-continuing vertices (dot > 0) are removed; reversal spikes
    are kept so genuine geometry is never silently deleted.
    """
    n = len(pts2)
    if n < 4 or sin_tol <= 0.0:
        return pts2
    keep = []
    for k in range(n):
        a, b, c = pts2[k - 1], pts2[k], pts2[(k + 1) % n]
        v1, v2 = b - a, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            keep.append(b)
            continue
        cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
        if float(v1 @ v2) > 0.0 and cross <= sin_tol * n1 * n2:
            continue
        keep.append(b)
    return np.asarray(keep) if len(keep) >= 3 else pts2


# --------------------------------------------------------------- main entry

def join_polygons(polygons, config):
    """Join a list of PlanarPolygon within tolerance.

    polygons: list of PlanarPolygon
    config:   parameter dict (see config.json / README)
    Returns (joined polygon list, report dict).
    """
    tol = float(config.get("join_tolerance", 0.2))
    min_angle = math.radians(float(config.get("plane_angle_min_deg", 10.0)))
    corner_snap = bool(config.get("corner_snap", True))
    corner_slack = float(config.get("corner_slack", 1.6))
    do_clip = bool(config.get("clip_beyond_lines", True))
    clip_eps = float(config.get("clip_epsilon", 1e-9))
    dedupe_tol = float(config.get("dedupe_tolerance", 1e-7))
    collinear_tol = float(config.get("collinear_sin_tolerance", 1e-8))
    do_merge = bool(config.get("merge_coplanar", True))

    cos_max = math.cos(min_angle)
    ext = tol * corner_slack  # slack for span checks so corners stay active

    flat = [p.flattened() for p in polygons]

    # 1) coplanar groups become single polygons (union, gap-closing)
    merges = []
    if do_merge:
        flat, merges = merge_coplanar_polygons(flat, tol, cos_max, dedupe_tol)

    # 2) adjacency detection: plane intersection line + overlapping near spans
    neighbors = defaultdict(list)  # polygon index -> [(line3, shared span)]
    pair_info = []
    for i, j in combinations(range(len(flat)), 2):
        pi, pj = flat[i], flat[j]
        if abs(float(pi.normal @ pj.normal)) > cos_max:
            continue  # near-parallel: merged above or intentionally ignored
        line = plane_plane_line(pi.origin, pi.normal, pj.origin, pj.normal)
        if line is None:
            continue
        line_pt, line_dir = line
        span_i = _near_span(pi.vertices, line_pt, line_dir, tol)
        if span_i is None:
            continue
        span_j = _near_span(pj.vertices, line_pt, line_dir, tol)
        if span_j is None:
            continue
        lo = max(span_i[0], span_j[0])
        hi = min(span_i[1], span_j[1])
        if hi < lo - tol:
            continue  # near the same infinite line but at disjoint locations
        shared = (min(lo, hi), max(lo, hi))
        neighbors[i].append((line, shared))
        neighbors[j].append((line, shared))
        pair_info.append({
            "pair": [pi.name, pj.name],
            "gap_to_line_before": [
                round(polygon_line_distance(pi.vertices, line_pt, line_dir), 6),
                round(polygon_line_distance(pj.vertices, line_pt, line_dir), 6),
            ],
            "boundary_gap_before": round(
                boundary_boundary_distance(pi.vertices, pj.vertices), 6),
            "_idx": (i, j),
            "_line": line,
        })

    # 3) per-polygon refine (insert corners), snap, clip, tidy
    result = []
    poly_stats = []
    for i, poly in enumerate(flat):
        pts2 = poly.to_2d(poly.vertices)
        n_in = len(pts2)
        constraints = [(_line_to_2d(poly, pt, dr), span)
                       for (pt, dr), span in neighbors[i]]
        if constraints:
            corners = _corner_candidates(constraints, ext, dedupe_tol)
            refined = _insert_corner_vertices_2d(pts2, corners, tol)
            snapped = _snap_vertices_2d(refined, constraints, tol,
                                        corner_snap, corner_slack, ext)
            disp = float(np.max(np.linalg.norm(snapped - refined, axis=1)))
            if do_clip:
                snapped = _clip_overshoot_2d(snapped, constraints, clip_eps)
            snapped = _dedupe(snapped, dedupe_tol)
            snapped = _remove_collinear(snapped, collinear_tol)
            inserted = len(refined) - n_in
        else:
            snapped, disp, inserted = pts2, 0.0, 0
        poly_stats.append({
            "polygon": poly.name,
            "max_vertex_move": round(disp, 6),
            "vertices_in": n_in,
            "vertices_inserted": inserted,
            "vertices_out": len(snapped),
        })
        result.append(PlanarPolygon(poly.name, poly.to_3d(snapped)))

    # 4) post-join quality metrics
    for info in pair_info:
        i, j = info.pop("_idx")
        line_pt, line_dir = info.pop("_line")
        info["gap_to_line_after"] = [
            round(polygon_line_distance(result[i].vertices, line_pt, line_dir), 6),
            round(polygon_line_distance(result[j].vertices, line_pt, line_dir), 6),
        ]
        info["boundary_gap_after"] = round(
            boundary_boundary_distance(result[i].vertices, result[j].vertices), 6)

    report = {
        "num_polygons_in": len(polygons),
        "num_polygons_out": len(result),
        "join_tolerance": tol,
        "num_merged_groups": len(merges),
        "merged_groups": merges,
        "num_adjacent_pairs": len(pair_info),
        "pairs": pair_info,
        "polygons": poly_stats,
    }
    return result, report
