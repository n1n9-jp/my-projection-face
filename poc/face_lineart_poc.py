#!/usr/bin/env python3
"""
Proof-of-concept pipeline:
  1. 顔検出（Haar cascade）
  2. 前処理（ヒストグラム平坦化 + 平滑化）
  3. エッジ抽出（Canny）
  4. 輪郭抽出して簡易ベクタ化（SVG polyline）

サンプル画像を指定して実行する。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="顔線画化PoC")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="入力画像（顔写真）へのパス",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="出力先ディレクトリ（デフォルト: outputs）",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.2,
        help="顔領域を切り出す際の余白率（デフォルト: 0.2 = 20%）",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="処理途中の画像をOpenCVウィンドウで表示（手動クローズが必要）",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_face(gray: np.ndarray) -> Tuple[int, int, int, int]:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"カスケードファイルの読み込みに失敗しました: {cascade_path}")

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )
    if len(faces) == 0:
        raise RuntimeError("顔が検出できませんでした。別の画像で試すかパラメータを調整してください。")
    # 面積の大きい順にソートして最大のものを選ぶ
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    return int(x), int(y), int(w), int(h)


def expand_roi(x: int, y: int, w: int, h: int, padding: float, width: int, height: int) -> Tuple[int, int, int, int]:
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + w + pad_x)
    y1 = min(height, y + h + pad_y)
    return x0, y0, x1 - x0, y1 - y0


def preprocess(gray_face: np.ndarray) -> np.ndarray:
    # ヒストグラム平坦化で明暗を均一化
    equalized = cv2.equalizeHist(gray_face)
    # ノイズ抑制しつつエッジ保持するためにbilateral filter
    smoothed = cv2.bilateralFilter(equalized, d=7, sigmaColor=75, sigmaSpace=75)
    return smoothed


def extract_edges(image: np.ndarray) -> np.ndarray:
    # Cannyエッジ検出。閾値は経験的に調整。
    edges = cv2.Canny(image, threshold1=40, threshold2=120)
    return edges


def filter_contours(contours: Iterable[np.ndarray], min_length: float = 40.0) -> List[np.ndarray]:
    filtered: List[np.ndarray] = []
    for contour in contours:
        length = cv2.arcLength(contour, closed=False)
        if length >= min_length:
            filtered.append(contour)
    return filtered


def contour_to_path(contour: np.ndarray) -> str:
    points = contour.reshape(-1, 2)
    if len(points) == 0:
        return ""
    commands = [f"M {points[0, 0]:.2f} {points[0, 1]:.2f}"]
    for x, y in points[1:]:
        commands.append(f"L {x:.2f} {y:.2f}")
    return " ".join(commands)


def contours_to_svg(
    contours: Sequence[np.ndarray],
    width: int,
    height: int,
    output_path: Path,
) -> None:
    header = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <g fill="none" stroke="#000" stroke-width="1">
"""
    body_lines = []
    for contour in contours:
        path_data = contour_to_path(contour)
        if not path_data:
            continue
        body_lines.append(f'    <path d="{path_data}" />')
    footer = "\n  </g>\n</svg>\n"
    svg_content = header + "\n".join(body_lines) + footer
    output_path.write_text(svg_content, encoding="utf-8")


def show_debug(title: str, image: np.ndarray) -> None:
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


def run_pipeline(image_path: Path, output_dir: Path, padding: float, show: bool) -> None:
    color = cv2.imread(str(image_path))
    if color is None:
        raise FileNotFoundError(f"画像を読み込めませんでした: {image_path}")
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    x, y, w, h = detect_face(gray)
    x0, y0, w_pad, h_pad = expand_roi(x, y, w, h, padding, gray.shape[1], gray.shape[0])

    face_color = color[y0 : y0 + h_pad, x0 : x0 + w_pad]
    face_gray = gray[y0 : y0 + h_pad, x0 : x0 + w_pad]

    processed = preprocess(face_gray)
    edges = extract_edges(processed)

    # 輪郭抽出（細いノイズを減らすためにモルフォロジー処理で線を太らせる）
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filtered = filter_contours(contours, min_length=60.0)

    # 近似して点数削減
    simplified: List[np.ndarray] = []
    for contour in filtered:
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon=epsilon, closed=False)
        simplified.append(approx)

    ensure_dir(output_dir)
    crop_path = output_dir / f"{image_path.stem}_crop.png"
    edges_path = output_dir / f"{image_path.stem}_edges.png"
    svg_path = output_dir / f"{image_path.stem}_contours.svg"

    cv2.imwrite(str(crop_path), face_color)
    cv2.imwrite(str(edges_path), edges)
    contours_to_svg(simplified, w_pad, h_pad, svg_path)

    if show:
        show_debug("face crop", face_color)
        show_debug("processed gray", processed)
        show_debug("edges", edges)

    print(f"[INFO] 顔検出領域: x={x0}, y={y0}, w={w_pad}, h={h_pad}")
    print(f"[INFO] 出力: {crop_path}, {edges_path}, {svg_path}")
    print(f"[INFO] 輪郭数: {len(simplified)}")


def main() -> None:
    args = parse_args()
    run_pipeline(args.image, args.output_dir, args.padding, args.show)


if __name__ == "__main__":
    main()
