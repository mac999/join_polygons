"""Polygon joining pipeline CLI.

Usage:
  python main.py                          # use config.json defaults
  python main.py --config config_room.json
  python main.py --input input --output output
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from polygon_join.json_io import read_json_folder
from polygon_join.snap import join_polygons
from polygon_join.step_io import read_step_folder, write_step_polygons

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(
        description="Joining tool for planar polygons")
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--input", default=None, help="input folder (overrides config)")
    parser.add_argument("--output", default=None, help="output folder (overrides config)")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    input_dir = Path(args.input or ROOT / config.get("input_dir", "input"))
    output_dir = Path(args.output or ROOT / config.get("output_dir", "output"))
    unit = config.get("units", "M")

    polygons = read_step_folder(input_dir, unit=unit)
    polygons += read_json_folder(input_dir)
    if not polygons:
        raise SystemExit(f"no input polygons found in: {input_dir}")
    print(f"Loaded {len(polygons)} polygon(s) from {input_dir}")

    joined, report = join_polygons(polygons, config)

    if report["merged_groups"]:
        print(f"Merged {report['num_merged_groups']} coplanar group(s):")
        for group in report["merged_groups"]:
            print(f"  {' + '.join(group['members'])} -> {group['result']}")

    out_step = output_dir / config.get("output_filename", "joined.step")
    write_step_polygons(joined, out_step, unit=unit)
    print(f"Saved joined result: {out_step}")

    if config.get("write_report", True):
        report_path = output_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"Saved report: {report_path}")

    print(f"\n{report['num_adjacent_pairs']} adjacent pair(s) "
          f"(tolerance {report['join_tolerance']} {unit.lower()})")
    for pair in report["pairs"]:
        before = max(pair["gap_to_line_before"])
        after = max(pair["gap_to_line_after"])
        print(f"  {pair['pair'][0]} <-> {pair['pair'][1]}: "
              f"gap to intersection line {before:.4f} -> {after:.2e}")
    for stats in report["polygons"]:
        if stats["vertices_inserted"]:
            print(f"  {stats['polygon']}: {stats['vertices_inserted']} vertex(es) "
                  f"inserted ({stats['vertices_in']} -> {stats['vertices_out']})")


if __name__ == "__main__":
    main()
