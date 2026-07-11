"""Example input generators.

Cases:
  cube  -- 6 faces of a 3 m cube, each boundary edge offset by +-0.15 m.
           Exercises plain 1:1 edge snapping and triple-plane corners.
  room  -- 9 polygons exercising every join combination at once:
             * floor split into two coplanar slabs      -> coplanar merge
             * south wall split into two coplanar parts -> coplanar merge
             * two slightly angled north walls meeting mid-span
               -> N:1 contact: the merged floor and the ceiling gain an
                  apex vertex where the two wall planes meet (vertex
                  count changes)
             * ordinary wall/floor/ceiling contacts     -> pairwise snap
  all   -- generate both cases.

Usage (venv_occ):
  python tools/make_example.py [cube|room|all]

The cube case writes input/. The room case writes input_room/ plus
config_room.json at the project root: its two north wall planes differ
by only ~5.7 deg, so plane_angle_min_deg is lowered to 2.0 there.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from polygon_join.model import PlanarPolygon  # noqa: E402

VERTEX_JITTER = 0.01  # in-plane per-vertex jitter


def make_rect(rng, name, c0, c1, c3, edge_noise):
    """Rectangle face from three corners (origin, u-corner, v-corner).

    Each of the four boundary edges is offset by a random amount within
    +-edge_noise to imitate RANSAC boundary uncertainty (gaps/overlaps).
    """
    c0, c1, c3 = (np.asarray(c, dtype=float) for c in (c0, c1, c3))
    u = c1 - c0
    lu = np.linalg.norm(u)
    u = u / lu
    v = c3 - c0
    lv = np.linalg.norm(v)
    v = v / lv

    umin = rng.uniform(-edge_noise, edge_noise)
    umax = lu + rng.uniform(-edge_noise, edge_noise)
    vmin = rng.uniform(-edge_noise, edge_noise)
    vmax = lv + rng.uniform(-edge_noise, edge_noise)
    pts2 = np.array([[umin, vmin], [umax, vmin], [umax, vmax], [umin, vmax]])
    pts2 += rng.uniform(-VERTEX_JITTER, VERTEX_JITTER, size=pts2.shape)

    verts3 = c0 + np.outer(pts2[:, 0], u) + np.outer(pts2[:, 1], v)
    return PlanarPolygon(name, verts3)


def build_cube(rng, size=3.0, edge_noise=0.15):
    """6 faces of a [0, size]^3 cube with noisy boundaries."""
    s = size
    faces = [
        ("floor",     (0, 0, 0), (s, 0, 0), (0, s, 0)),
        ("ceiling",   (0, 0, s), (s, 0, s), (0, s, s)),
        ("wall_xmin", (0, 0, 0), (0, s, 0), (0, 0, s)),
        ("wall_xmax", (s, 0, 0), (s, s, 0), (s, 0, s)),
        ("wall_ymin", (0, 0, 0), (s, 0, 0), (0, 0, s)),
        ("wall_ymax", (0, s, 0), (s, s, 0), (0, s, s)),
    ]
    return [make_rect(rng, name, c0, c1, c3, edge_noise)
            for name, c0, c1, c3 in faces]


def build_room(rng, edge_noise=0.08):
    """Room with coplanar splits and an N:1 zigzag wall contact.

    True footprint (pentagon): (0,0) (6,0) (6,3) (3,3.15) (0,3), height 3.
    The floor and ceiling are DRAWN as plain rectangles up to y=3.05, so
    the apex vertex (3, 3.15) can only appear through vertex insertion.
    """
    apex = (3.0, 3.15)
    top_y = 3.05  # drawn depth of floor/ceiling rectangles
    h = 3.0
    faces = [
        # floor split into two coplanar slabs (seam at x=2, away from the apex)
        ("floor_a",   (0, 0, 0),      (2, 0, 0),          (0, top_y, 0)),
        ("floor_b",   (2, 0, 0),      (6, 0, 0),          (2, top_y, 0)),
        # ceiling as a single rectangle -> gains the apex vertex too
        ("ceiling",   (0, 0, h),      (6, 0, h),          (0, top_y, h)),
        # side walls
        ("wall_x0",   (0, 0, 0),      (0, 3, 0),          (0, 0, h)),
        ("wall_x6",   (6, 0, 0),      (6, 3, 0),          (6, 0, h)),
        # south wall split into two coplanar parts (seam at x=3.5)
        ("wall_y0_a", (0, 0, 0),      (3.5, 0, 0),        (0, 0, h)),
        ("wall_y0_b", (3.5, 0, 0),    (6, 0, 0),          (3.5, 0, h)),
        # two slightly angled north walls meeting at the apex (~5.7 deg apart)
        ("wall_zig_a", (0, 3, 0),     (apex[0], apex[1], 0), (0, 3, h)),
        ("wall_zig_b", (apex[0], apex[1], 0), (6, 3, 0),   (apex[0], apex[1], h)),
    ]
    return [make_rect(rng, name, c0, c1, c3, edge_noise)
            for name, c0, c1, c3 in faces]


ROOM_CONFIG = {
    "input_dir": "input_room",
    "output_dir": "output_room",
    "output_filename": "joined.step",
    "join_tolerance": 0.2,
    "plane_angle_min_deg": 2.0,
    "merge_coplanar": True,
    "corner_snap": True,
    "corner_slack": 1.6,
    "clip_beyond_lines": True,
    "clip_epsilon": 1e-09,
    "dedupe_tolerance": 1e-07,
    "collinear_sin_tolerance": 1e-08,
    "write_report": True,
    "units": "M",
}


def _write_case(polygons, folder):
    from polygon_join.step_io import write_step_polygons  # lazy: needs OCC

    folder.mkdir(exist_ok=True)
    for poly in polygons:
        out = folder / f"{poly.name}.step"
        write_step_polygons([poly], out)
        print(f"Saved: {out}")


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "cube"
    if case not in ("cube", "room", "all"):
        raise SystemExit(f"unknown case: {case} (expected cube|room|all)")

    if case in ("cube", "all"):
        _write_case(build_cube(np.random.default_rng(42)), ROOT / "input")
        print("Cube case done (3 m cube, boundary noise +-0.15 m) -> input/")

    if case in ("room", "all"):
        _write_case(build_room(np.random.default_rng(7)), ROOT / "input_room")
        config_path = ROOT / "config_room.json"
        config_path.write_text(json.dumps(ROOM_CONFIG, indent=2) + "\n",
                               encoding="utf-8")
        print(f"Room case done (coplanar splits + zigzag N:1) -> input_room/")
        print(f"Saved: {config_path}  (run: python main.py --config {config_path.name})")


if __name__ == "__main__":
    main()
