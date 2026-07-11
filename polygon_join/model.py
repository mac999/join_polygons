"""Planar polygon model and basic geometric primitives."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def fit_plane(points: np.ndarray):
    """Least-squares plane fit via SVD. Returns (origin, unit normal)."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    return centroid, normal


def plane_frame(normal: np.ndarray):
    """Build an orthonormal in-plane 2D frame (u, v) for a plane normal.

    The frame is right-handed with the normal: u x v == normal, so a
    counter-clockwise 2D ring corresponds to a +normal winding in 3D.
    """
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


@dataclass
class PlanarPolygon:
    """3D planar polygon defined by an ordered boundary vertex array (N, 3)."""

    name: str
    vertices: np.ndarray  # (N, 3), closing vertex not repeated

    origin: np.ndarray = field(default=None, repr=False)
    normal: np.ndarray = field(default=None, repr=False)
    _u: np.ndarray = field(default=None, repr=False)
    _v: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=float)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or len(self.vertices) < 3:
            raise ValueError(f"{self.name}: vertex array must have shape (N>=3, 3)")
        self.origin, self.normal = fit_plane(self.vertices)
        self._u, self._v = plane_frame(self.normal)

    def to_2d(self, pts3: np.ndarray) -> np.ndarray:
        """Project points into the plane frame (normal component dropped)."""
        rel = np.atleast_2d(pts3) - self.origin
        return np.column_stack([rel @ self._u, rel @ self._v])

    def to_3d(self, pts2: np.ndarray) -> np.ndarray:
        pts2 = np.atleast_2d(pts2)
        return self.origin + np.outer(pts2[:, 0], self._u) + np.outer(pts2[:, 1], self._v)

    def flattened(self) -> "PlanarPolygon":
        """Copy with vertices projected onto the fitted plane."""
        return PlanarPolygon(self.name, self.to_3d(self.to_2d(self.vertices)))
