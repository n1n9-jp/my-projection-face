#!/usr/bin/env python3
"""
SVG -> GeoJSON 変換スクリプト（初期版）

対応範囲:
  - 要素: path / polyline / polygon
  - path コマンド: M / L / H / V / Z（絶対・相対）
  - スタイル属性・行列変換・曲線コマンドには未対応
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Number = float
Point = Tuple[Number, Number]

ALLOWED_PATH_COMMANDS = {"M", "L", "H", "V", "Z", "m", "l", "h", "v", "z"}
TOKEN_PATTERN = re.compile(r"[MLHVZmlhvz]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

DEFAULT_LON_MIN = -179.0
DEFAULT_LON_MAX = 179.0
DEFAULT_LAT_MIN = -85.0
DEFAULT_LAT_MAX = 85.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a limited SVG subset to GeoJSON.")
    parser.add_argument("-i", "--input", required=True, help="入力となる SVG ファイルパス")
    parser.add_argument(
        "-o",
        "--output",
        help="出力先 GeoJSON ファイルパス（省略時は標準出力）",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON のインデント幅（省略時は 2）",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="GeoJSON 出力時に座標を経緯度範囲へ正規化しない（ピクセル値のまま出力）",
    )
    parser.add_argument(
        "--normalize-range",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="正規化時に使用する経度・緯度の範囲を指定（デフォルト: -179 179 -85 85）",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="出力 GeoJSON に元座標範囲などのメタデータを含める",
    )
    return parser.parse_args()


def strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def parse_points(points_str: str) -> List[Point]:
    cleaned = points_str.replace(",", " ")
    values = [float(v) for v in cleaned.split() if v]
    if len(values) % 2 != 0:
        raise ValueError("points 属性の数値がペアになっていません")
    return [(values[i], values[i + 1]) for i in range(0, len(values), 2)]


def tokenize_path(d: str) -> List[str]:
    tokens = TOKEN_PATTERN.findall(d.replace(",", " "))
    if not tokens:
        raise ValueError("path d 属性が空です")
    return tokens


def parse_path(d: str) -> List[List[Point]]:
    tokens = tokenize_path(d)
    sequences: List[List[Point]] = []
    current: List[Point] = []
    cursor: Optional[Point] = None
    start_point: Optional[Point] = None
    command: Optional[str] = None
    idx = 0

    def flush_current():
        nonlocal current
        if current:
            sequences.append(current)
            current = []

    while idx < len(tokens):
        token = tokens[idx]
        if token in ALLOWED_PATH_COMMANDS:
            command = token
            idx += 1
            if command.upper() == "Z":
                if not current:
                    raise ValueError("Z コマンドの前に座標がありません")
                if start_point and current[-1] != start_point:
                    current.append(start_point)
                flush_current()
                cursor = start_point
                start_point = None
            continue

        if command is None:
            raise ValueError("コマンド指定のない数値が出現しました")

        cmd_upper = command.upper()
        is_relative_cmd = command.islower()

        def base_point() -> Point:
            nonlocal cursor
            if cursor is None:
                return (0.0, 0.0)
            return cursor

        if cmd_upper == "M":
            x = float(token)
            idx += 1
            if idx >= len(tokens):
                raise ValueError("M コマンドに Y 座標がありません")
            y = float(tokens[idx])
            idx += 1
            bx, by = base_point()
            if is_relative_cmd:
                point = (bx + x, by + y)
            else:
                point = (x, y)
            flush_current()
            current.append(point)
            cursor = point
            start_point = cursor
            command = "l" if is_relative_cmd else "L"
            continue

        if cmd_upper == "L":
            x = float(token)
            idx += 1
            if idx >= len(tokens):
                raise ValueError("L コマンドに Y 座標がありません")
            y = float(tokens[idx])
            idx += 1
            bx, by = base_point()
            if is_relative_cmd:
                point = (bx + x, by + y)
            else:
                point = (x, y)
            current.append(point)
            cursor = point
            continue

        if cmd_upper == "H":
            x = float(token)
            idx += 1
            bx, by = base_point()
            point = (bx + x, by) if is_relative_cmd else (x, by)
            current.append(point)
            cursor = point
            continue

        if cmd_upper == "V":
            y = float(token)
            idx += 1
            bx, by = base_point()
            point = (bx, by + y) if is_relative_cmd else (bx, y)
            current.append(point)
            cursor = point
            continue

        raise ValueError(f"未対応のコマンドまたは形式です: {command}")

    flush_current()

    if sequences and sequences[0]:
        first_start = sequences[0][0]
        if sequences[-1] and sequences[-1][-1] == first_start:
            sequences[-1] = sequences[-1][:-1]

    return sequences


def compute_bounds(features: Sequence[Dict]) -> Optional[Tuple[Number, Number, Number, Number]]:
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    found = False

    def update(point: Sequence[Number]) -> None:
        nonlocal min_x, min_y, max_x, max_y, found
        x, y = point
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        found = True

    for feature in features:
        geometry = feature["geometry"]
        gtype = geometry["type"]
        if gtype == "LineString":
            for point in geometry["coordinates"]:
                update(point)
        elif gtype == "Polygon":
            for ring in geometry["coordinates"]:
                for point in ring:
                    update(point)

    if not found:
        return None
    return (min_x, min_y, max_x, max_y)


def normalize_features(
    features: Sequence[Dict],
    lon_min: Number,
    lon_max: Number,
    lat_min: Number,
    lat_max: Number,
) -> Optional[Tuple[Number, Number, Number, Number]]:
    bounds = compute_bounds(features)
    if bounds is None:
        return None

    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min

    def transform(x: Number, y: Number) -> List[Number]:
        if width == 0:
            lon = (lon_min + lon_max) / 2.0
        else:
            ratio_x = (x - min_x) / width
            lon = lon_min + ratio_x * lon_span
        if height == 0:
            lat = (lat_min + lat_max) / 2.0
        else:
            ratio_y = (y - min_y) / height
            lat = lat_max - ratio_y * lat_span
        return [lon, lat]

    for feature in features:
        geometry = feature["geometry"]
        gtype = geometry["type"]
        if gtype == "LineString":
            geometry["coordinates"] = [transform(x, y) for x, y in geometry["coordinates"]]
        elif gtype == "Polygon":
            geometry["coordinates"] = [
                [transform(x, y) for x, y in ring]
                for ring in geometry["coordinates"]
            ]
        else:
            raise ValueError(f"未対応のジオメトリタイプ: {gtype}")

    return bounds


def convert_svg(path: str) -> Dict:
    tree = ET.parse(path)
    root = tree.getroot()
    features: List[Dict] = []

    for elem in root.iter():
        tag = strip_namespace(elem.tag)
        if tag not in {"path", "polyline", "polygon"}:
            continue

        properties = {"type": tag}
        if "id" in elem.attrib:
            properties["id"] = elem.attrib["id"]

        if tag == "path":
            data = elem.attrib.get("d")
            if not data:
                raise ValueError("path 要素に d 属性がありません")
            sequences = parse_path(data)
            for seq in sequences:
                if len(seq) < 2:
                    continue
                geometry_type = "Polygon" if seq[0] == seq[-1] and len(seq) >= 4 else "LineString"
                if geometry_type == "Polygon":
                    polygon = [seq]
                    features.append(
                        {
                            "type": "Feature",
                            "geometry": {"type": "Polygon", "coordinates": polygon},
                            "properties": properties,
                        }
                    )
                else:
                    features.append(
                        {
                            "type": "Feature",
                            "geometry": {"type": "LineString", "coordinates": seq},
                            "properties": properties,
                        }
                    )
        elif tag in {"polyline", "polygon"}:
            points_attr = elem.attrib.get("points")
            if not points_attr:
                raise ValueError(f"{tag} 要素に points 属性がありません")
            points = parse_points(points_attr)
            if tag == "polygon" and not is_closed(points):
                points = points + points[:1]
            features.append(feature_from_points(points, properties))

    return {"type": "FeatureCollection", "features": features}


def feature_from_points(points: Sequence[Point], properties: Dict) -> Dict:
    geometry_type = "Polygon" if is_closed(points) and len(points) >= 4 else "LineString"
    coords = list(points)
    if geometry_type == "Polygon":
        return {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": properties,
        }
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": properties,
    }


def is_closed(points: Sequence[Point]) -> bool:
    return len(points) >= 2 and points[0] == points[-1]


def main() -> int:
    args = parse_args()
    try:
        geojson = convert_svg(args.input)
    except Exception as exc:  # noqa: BLE001
        print(f"変換エラー: {exc}", file=sys.stderr)
        return 1

    metadata: Dict[str, object] = {}
    if not args.no_normalize:
        lon_min, lon_max, lat_min, lat_max = (
            args.normalize_range
            if args.normalize_range is not None
            else (DEFAULT_LON_MIN, DEFAULT_LON_MAX, DEFAULT_LAT_MIN, DEFAULT_LAT_MAX)
        )
        if lon_min >= lon_max or lat_min >= lat_max:
            print("正規化範囲が不正です（MIN < MAX を満たす必要があります）", file=sys.stderr)
            return 1
        try:
            bounds = normalize_features(geojson["features"], lon_min, lon_max, lat_min, lat_max)
        except Exception as exc:  # noqa: BLE001
            print(f"正規化エラー: {exc}", file=sys.stderr)
            return 1
        if bounds is not None and args.include_metadata:
            metadata["source_bbox"] = {
                "min_x": bounds[0],
                "min_y": bounds[1],
                "max_x": bounds[2],
                "max_y": bounds[3],
            }
            metadata["normalize_range"] = {
                "lon_min": lon_min,
                "lon_max": lon_max,
                "lat_min": lat_min,
                "lat_max": lat_max,
            }

    if metadata and args.include_metadata:
        geojson["metadata"] = metadata

    output = json.dumps(geojson, indent=args.indent)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
            f.write("\n")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
