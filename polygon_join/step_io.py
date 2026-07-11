"""STEP file I/O based on pythonocc-core.

Converts planar polygons <-> planar STEP FACEs. Units come from the
config "units" key (default: meters).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCC.Core.BRepTools import BRepTools_WireExplorer, breptools
from OCC.Core.gp import gp_Pnt
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Compound, topods

from .model import PlanarPolygon


def _set_units(unit: str):
    """Set STEP unit parameters. Must be called BEFORE creating a Reader/Writer.

    (The Writer snapshots write.step.unit at construction time, so setting it
    afterwards is ignored.) Before the STEP module initializes, the parameters
    are not registered and SetCVal fails; in that case a throwaway Writer is
    created to trigger registration, then the call is retried.
    """
    if not Interface_Static.SetCVal("xstep.cascade.unit", unit):
        STEPControl_Writer()  # force registration of STEP static parameters
        if not Interface_Static.SetCVal("xstep.cascade.unit", unit):
            raise RuntimeError(f"failed to set internal STEP unit: {unit}")
    if not Interface_Static.SetCVal("write.step.unit", unit):
        raise RuntimeError(f"failed to set STEP write unit: {unit}")


def read_step_polygons(path, unit="M"):
    """Read the outer wire of every FACE in a STEP file as PlanarPolygon."""
    path = Path(path)
    _set_units(unit)
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise IOError(f"failed to read STEP file: {path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    polygons = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    idx = 0
    while explorer.More():
        face = topods.Face(explorer.Current())
        wire = breptools.OuterWire(face)
        wexp = BRepTools_WireExplorer(wire, face)
        pts = []
        while wexp.More():
            pnt = BRep_Tool.Pnt(wexp.CurrentVertex())
            pts.append([pnt.X(), pnt.Y(), pnt.Z()])
            wexp.Next()
        if len(pts) >= 3:
            polygons.append(PlanarPolygon(f"{path.stem}#{idx}", np.array(pts)))
        explorer.Next()
        idx += 1
    return polygons


def read_step_folder(folder, unit="M"):
    """Read polygons from every *.step / *.stp file in a folder."""
    folder = Path(folder)
    polygons = []
    for path in sorted(list(folder.glob("*.step")) + list(folder.glob("*.stp"))):
        polygons.extend(read_step_polygons(path, unit=unit))
    return polygons


def write_step_polygons(polygons, path, unit="M"):
    """Write a list of PlanarPolygon into one STEP file (compound of FACEs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for poly in polygons:
        maker = BRepBuilderAPI_MakePolygon()
        for vtx in poly.vertices:
            maker.Add(gp_Pnt(float(vtx[0]), float(vtx[1]), float(vtx[2])))
        maker.Close()
        face = BRepBuilderAPI_MakeFace(maker.Wire(), True).Face()
        builder.Add(compound, face)

    _set_units(unit)
    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_AsIs)
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise IOError(f"failed to write STEP file: {path}")
