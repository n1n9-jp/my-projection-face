#!/usr/bin/env python3
"""
Stable Diffusion + ControlNet(LineArt) を使って顔写真から線画を生成する PoC スクリプト。

前提:
  - 仮想環境 (.sd-venv) にて PyTorch(MPS), diffusers, controlnet-aux をインストール済み
  - Apple Silicon (MPS) / CPU どちらでも動作は可能だが、MPS 推奨

使い方 (例):
  source .sd-venv/bin/activate
  python experiments/controlnet_lineart/lineart_poc.py \
      --image samples/self.jpg \
      --output-dir outputs/controlnet \
      --prompt "line art portrait, clean smooth lines" \
      --negative-prompt "blurry, messy, noisy"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from controlnet_aux import LineartDetector
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
from PIL import Image


DEFAULT_BASE_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_lineart"
DEFAULT_DETECTOR_REPO = "lllyasviel/Annotators"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable Diffusion + ControlNet(LineArt) PoC")
    parser.add_argument("--image", type=Path, required=True, help="入力画像パス (RGB)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/controlnet"),
        help="出力先ディレクトリ",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="line art portrait, sketch, clean smooth lines, high quality",
        help="生成プロンプト",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, messy, noisy, extra limbs",
        help="ネガティブプロンプト",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Stable Diffusion ベースモデル (HuggingFace ID)",
    )
    parser.add_argument(
        "--controlnet-model",
        type=str,
        default=DEFAULT_CONTROLNET_MODEL,
        help="ControlNet モデル (HuggingFace ID)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="生成画像の幅 (ControlNet用にリサイズされます)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="生成画像の高さ (ControlNet用にリサイズされます)",
    )
    parser.add_argument("--steps", type=int, default=20, help="推論ステップ数")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="ガイダンススケール")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード (再現性確保用)")
    return parser.parse_args()


def create_pipeline(base_model: str, controlnet_model: str) -> StableDiffusionControlNetPipeline:
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    if torch.backends.mps.is_available():
        pipe.to("mps")
    else:
        pipe.to("cpu")

    return pipe


def generate_lineart(
    pipe: StableDiffusionControlNetPipeline,
    input_path: Union[Path, Image.Image],
    width: int,
    height: int,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int,
    detector: Optional[LineartDetector] = None,
) -> Tuple[Image.Image, Image.Image]:
    if isinstance(input_path, Image.Image):
        image = input_path.convert("RGB").resize((width, height))
    else:
        image = load_image(str(input_path)).convert("RGB").resize((width, height))

    if detector is None:
        detector = LineartDetector.from_pretrained(DEFAULT_DETECTOR_REPO)

    control_result = detector(image)
    if isinstance(control_result, Image.Image):
        control_image = control_result
    else:
        control_image = Image.fromarray(control_result)

    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    output_image = result.images[0]

    return control_image, output_image


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipe = create_pipeline(args.base_model, args.controlnet_model)
    detector = LineartDetector.from_pretrained(DEFAULT_DETECTOR_REPO)

    control_image, output_image = generate_lineart(
        pipe=pipe,
        input_path=args.image,
        width=args.width,
        height=args.height,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        detector=detector,
    )

    control_path = args.output_dir / f"{args.image.stem}_control_lineart.png"
    output_path = args.output_dir / f"{args.image.stem}_generated.png"
    control_image.save(control_path)
    output_image.save(output_path)

    print(f"[INFO] ControlNet line art saved to {control_path}")
    print(f"[INFO] Generated line art saved to {output_path}")


if __name__ == "__main__":
    main()
