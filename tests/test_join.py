"""End-to-end checks for the join algorithm (cube and room cases).

Runs fully in memory (numpy + shapely only; no STEP I/O needed):
  python tests/test_join.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from polygon_join.snap import join_polygons  # noqa: E402
from make_example import ROOM_CONFIG, build_cube, build_room  # noqa: E402


def _has_vertex(poly, point, tol=1e-6):
    return bool(np.min(np.linalg.norm(poly.vertices - np.asarray(point), axis=1)) <= tol)


def test_cube():
    polys = build_cube(np.random.default_rng(42))
    joined, report = join_polygons(polys, {"join_tolerance": 0.2})

    assert report["num_adjacent_pairs"] == 12, report["num_adjacent_pairs"]
    assert report["num_merged_groups"] == 0
    for pair in report["pairs"]:
        assert max(pair["gap_to_line_after"]) <= 1e-6, pair
        assert pair["boundary_gap_after"] <= 1e-6, pair

    # every face collapses back to the exact unit square of the cube
    for poly in joined:
        assert len(poly.vertices) == 4, (poly.name, len(poly.vertices))
    pts = np.vstack([p.vertices for p in joined])
    assert pts.min() >= -1e-6 and pts.max() <= 3.0 + 1e-6

    # 8 corners, each shared by exactly 3 faces
    counts = {}
    for p in joined:
        for v in np.round(p.vertices, 6):
            counts[tuple(v)] = counts.get(tuple(v), 0) + 1
    assert len(counts) == 8, len(counts)
    assert all(c == 3 for c in counts.values()), counts
    print("cube case: OK (12 pairs closed, 8 watertight corners)")


def test_room():
    polys = build_room(np.random.default_rng(7))
    joined, report = join_polygons(polys, ROOM_CONFIG)

    # 9 inputs -> 7 outputs via two coplanar merges (floor pair, wall_y0 pair)
    assert report["num_merged_groups"] == 2, report["merged_groups"]
    assert len(joined) == 7, [p.name for p in joined]
    assert report["num_adjacent_pairs"] == 15, report["num_adjacent_pairs"]
    for pair in report["pairs"]:
        assert max(pair["gap_to_line_after"]) <= 1e-6, pair

    by_name = {p.name: p for p in joined}
    floor = next(p for name, p in by_name.items() if name.startswith("floor"))
    ceiling = by_name["ceiling"]

    # N:1 contact: floor/ceiling were drawn as rectangles but gain the apex
    # vertex where the two north wall planes meet -> pentagons
    apex = (3.0, 3.15)
    expected = [(0, 0), (6, 0), (6, 3), apex, (0, 3)]
    for poly, z in ((floor, 0.0), (ceiling, 3.0)):
        assert len(poly.vertices) == 5, (poly.name, len(poly.vertices))
        for x, y in expected:
            assert _has_vertex(poly, (x, y, z)), (poly.name, (x, y, z))

    # merged/plain walls collapse to clean rectangles
    for name in ("wall_x0", "wall_x6", "wall_zig_a", "wall_zig_b"):
        assert len(by_name[name].vertices) == 4, (name, len(by_name[name].vertices))
    wall_y0 = next(p for name, p in by_name.items() if name.startswith("wall_y0"))
    assert len(wall_y0.vertices) == 4, len(wall_y0.vertices)

    inserted = {s["polygon"]: s["vertices_inserted"] for s in report["polygons"]}
    assert inserted[floor.name] >= 1 and inserted["ceiling"] >= 1, inserted
    print("room case: OK (2 merges, 15 pairs closed, apex vertices inserted)")


if __name__ == "__main__":
    test_cube()
    test_room()
    print("All tests passed.")
