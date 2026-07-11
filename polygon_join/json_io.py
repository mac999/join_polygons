"""JSON polygon I/O.

Format:
{
  "polygons": [
    {"name": "wall_0", "vertices": [[x, y, z], ...]},
    ...
  ]
}
A top-level list is also accepted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import PlanarPolygon


def read_json_polygons(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["polygons"] if isinstance(data, dict) else data
    polygons = []
    for i, item in enumerate(items):
        name = item.get("name", f"{path.stem}#{i}")
        polygons.append(PlanarPolygon(name, np.array(item["vertices"], dtype=float)))
    return polygons


def read_json_folder(folder):
    folder = Path(folder)
    polygons = []
    for path in sorted(folder.glob("*.json")):
        polygons.extend(read_json_polygons(path))
    return polygons


def write_json_polygons(polygons, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "polygons": [
            {"name": p.name, "vertices": p.vertices.tolist()} for p in polygons
        ]
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
