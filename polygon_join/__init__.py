"""polygon_join: joining library for RANSAC-extracted planar polygons.

Joins the adjacent boundaries of 3D planar polygons extracted from point
clouds: cross-plane contacts are extended/trimmed onto plane-plane
intersection lines (with triple-plane corner snapping and N:1 vertex
insertion), and coplanar contacts are merged via 2D boolean union.
"""
from .model import PlanarPolygon
from .snap import join_polygons
from .merge import merge_coplanar_polygons
from .json_io import read_json_polygons, write_json_polygons

__all__ = [
    "PlanarPolygon",
    "join_polygons",
    "merge_coplanar_polygons",
    "read_json_polygons",
    "write_json_polygons",
]
