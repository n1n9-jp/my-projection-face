#!/usr/bin/env python3
"""
ControlNet → SVG → GeoJSON を一括実行するパイプライン本体。

`scripts/run_pipeline.py` から呼び出されることを想定しているが、
このモジュール単体でもエントリーポイントとして利用できる。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

from controlnet_aux import LineartDetector

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.controlnet_lineart.lineart_poc import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_BIN_THRESHOLD,
    DEFAULT_COARSE,
    DEFAULT_CONTROLNET_MODEL,
    DEFAULT_CONTROL_WEIGHT,
    DEFAULT_DETECTOR_REPO,
    DEFAULT_DETECT_RESOLUTION,
    create_pipeline,
    generate_lineart,
)

PNG_TO_SVG_SCRIPT = PIPELINE_ROOT / "png_to_svg.sh"
SVG_TO_GEOJSON_SCRIPT = PIPELINE_ROOT / "svg_to_geojson.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full ControlNet → SVG → GeoJSON pipeline.")
    parser.add_argument("--input", type=Path, required=True, help="入力画像 (JPG/PNG)")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "controlnet",
        help="出力のルートディレクトリ (デフォルト: outputs/controlnet)",
    )
    parser.add_argument(
        "--session",
        help="出力をサブディレクトリにまとめるためのセッション名 (例: session1)",
    )
    parser.add_argument(
        "--basename",
        help="生成するファイルのベース名 (省略時は入力ファイル名から決定)",
    )

    # ControlNet parameters
    parser.add_argument("--width", type=int, default=384, help="生成画像の幅 (default: 384)")
    parser.add_argument("--height", type=int, default=384, help="生成画像の高さ (default: 384)")
    parser.add_argument("--steps", type=int, default=8, help="推論ステップ数")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="ガイダンススケール")
    parser.add_argument(
        "--control-weight",
        type=float,
        default=DEFAULT_CONTROL_WEIGHT,
        help="ControlNet の寄与 (0-1)",
    )
    parser.add_argument(
        "--detect-resolution",
        type=int,
        default=DEFAULT_DETECT_RESOLUTION,
        help="LineartDetector の detect_resolution",
    )
    parser.add_argument(
        "--bin-threshold",
        type=int,
        default=DEFAULT_BIN_THRESHOLD,
        help="最終線画の二値化閾値 (0-255)",
    )
    parser.add_argument("--coarse", action="store_true", dest="coarse", help="coarse モードを有効化")
    parser.add_argument("--no-coarse", action="store_false", dest="coarse", help="coarse モードを無効化")
    parser.set_defaults(coarse=DEFAULT_COARSE)
    parser.add_argument("--seed", type=int, default=42, help="再現性確保用の乱数シード")
    parser.add_argument("--prompt", default="line art portrait, clean smooth lines", help="生成プロンプト")
    parser.add_argument("--negative-prompt", default="blurry, messy, noisy", help="ネガティブプロンプト")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Stable Diffusion ベースモデル ID")
    parser.add_argument(
        "--controlnet-model",
        default=DEFAULT_CONTROLNET_MODEL,
        help="ControlNet モデル ID",
    )

    # Potrace parameters
    parser.add_argument(
        "--turdsize",
        type=int,
        default=2,
        help="potrace の --turdsize (小領域除去の閾値)",
    )
    parser.add_argument(
        "--alphamax",
        type=float,
        default=0.8,
        help="potrace の --alphamax (曲線化のしきい値)",
    )

    # svg_to_geojson parameters
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="GeoJSON へ変換する際に座標の正規化をスキップ",
    )
    parser.add_argument(
        "--normalize-range",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="正規化に使用する経緯度範囲",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="GeoJSON にメタデータを含める",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="GeoJSON のインデント幅 (default: 2)",
    )

    return parser.parse_args()


def run_subprocess(cmd: Sequence[str]) -> None:
    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_pipeline(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {args.input}")

    output_dir = Path(args.output_root)
    if args.session:
        output_dir = output_dir / args.session
    output_dir.mkdir(parents=True, exist_ok=True)

    basename = args.basename or args.input.stem

    print("[INFO] Loading ControlNet pipeline...")
    pipe = create_pipeline(args.base_model, args.controlnet_model)
    detector = LineartDetector.from_pretrained(DEFAULT_DETECTOR_REPO)

    print("[INFO] Generating line art via ControlNet...")
    control_img, binary_img = generate_lineart(
        pipe=pipe,
        input_path=args.input,
        width=args.width,
        height=args.height,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        detector=detector,
        control_weight=args.control_weight,
        coarse=args.coarse,
        detect_resolution=args.detect_resolution,
        bin_threshold=args.bin_threshold,
    )

    control_path = output_dir / f"{basename}_control_lineart.png"
    binary_path = output_dir / f"{basename}_generated.png"
    control_img.save(control_path)
    binary_img.save(binary_path)
    print(f"[INFO] Saved ControlNet preview to {control_path}")
    print(f"[INFO] Saved binary line art to {binary_path}")

    svg_path = output_dir / f"{basename}.svg"
    potrace_args: List[str] = ["--turdsize", str(args.turdsize), "--alphamax", str(args.alphamax)]

    run_subprocess(
        [
            "bash",
            str(PNG_TO_SVG_SCRIPT),
            str(binary_path),
            str(svg_path),
            *potrace_args,
        ]
    )
    print(f"[INFO] Saved SVG to {svg_path}")

    geojson_path = output_dir / f"{basename}.geojson"
    geojson_cmd: List[str] = [
        sys.executable,
        str(SVG_TO_GEOJSON_SCRIPT),
        "--input",
        str(svg_path),
        "--output",
        str(geojson_path),
        "--indent",
        str(args.indent),
    ]
    if args.no_normalize:
        geojson_cmd.append("--no-normalize")
    if args.normalize_range:
        geojson_cmd.extend(["--normalize-range", *map(str, args.normalize_range)])
    if args.include_metadata:
        geojson_cmd.append("--include-metadata")

    run_subprocess(geojson_cmd)
    print(f"[INFO] Saved GeoJSON to {geojson_path}")
    print("[DONE] Pipeline completed successfully.")


def main() -> int:
    args = parse_args()
    try:
        run_pipeline(args)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] コマンド実行中に失敗しました: {exc}", file=sys.stderr)
        return exc.returncode or 1
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] パイプライン実行中にエラーが発生しました: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
