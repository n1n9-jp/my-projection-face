#!/usr/bin/env python3
"""
Gradio UI for tuning Stable Diffusion + ControlNet (LineArt) parameters.

This UI is intended for local parameter exploration only.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import gradio as gr
import torch
from controlnet_aux import LineartDetector
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TMP_ROOT = ROOT / ".tmp"
TMP_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("TMPDIR", str(TMP_ROOT))

from experiments.controlnet_lineart.lineart_poc import (
    DEFAULT_BASE_MODEL,
    DEFAULT_CONTROLNET_MODEL,
    DEFAULT_DETECTOR_REPO,
    create_pipeline,
    generate_lineart,
)


pipe: torch.nn.Module | None = None
detector: LineartDetector | None = None
CURRENT_BASE_MODEL: str | None = None
CURRENT_CONTROLNET_MODEL: str | None = None


def ensure_pipeline(
    base_model: str,
    controlnet_model: str,
) -> None:
    global pipe, detector, CURRENT_BASE_MODEL, CURRENT_CONTROLNET_MODEL
    if pipe is None or CURRENT_BASE_MODEL != base_model or CURRENT_CONTROLNET_MODEL != controlnet_model:
        pipe = create_pipeline(base_model, controlnet_model)
        CURRENT_BASE_MODEL = base_model
        CURRENT_CONTROLNET_MODEL = controlnet_model
    if detector is None:
        detector = LineartDetector.from_pretrained(DEFAULT_DETECTOR_REPO)


def infer(
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int,
    width: int,
    height: int,
    control_weight: float,
    use_coarse: bool,
    detect_resolution: int,
    bin_threshold: int,
    base_model: str,
    controlnet_model: str,
):
    import numpy as np
    import traceback

    try:
        ensure_pipeline(base_model, controlnet_model)
        if pipe is None or detector is None:
            raise RuntimeError("Pipeline or detector not initialized")

        width = int(width)
        height = int(height)
        seed = int(seed)
        print("[INFO] Starting generation with params:", {
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance_scale,
            "seed": seed,
        })
        control_img, generated_img = generate_lineart(
            pipe=pipe,
            input_path=image,
            width=width,
            height=height,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            detector=detector,
            control_weight=control_weight,
            coarse=use_coarse,
            detect_resolution=int(detect_resolution),
            bin_threshold=int(bin_threshold),
        )
        print(
            "[INFO] Generation succeeded:",
            {
                "control_size": control_img.size,
                "control_mode": control_img.mode,
                "generated_size": generated_img.size,
                "generated_mode": generated_img.mode,
                "generated_bbox": generated_img.getbbox(),
            },
        )

        if control_img.mode != "RGB":
            control_img = control_img.convert("RGB")
        if generated_img.mode != "RGB":
            generated_img = generated_img.convert("RGB")

        debug_dir = TMP_ROOT / "gradio_debug"
        debug_dir.mkdir(exist_ok=True)
        control_path = debug_dir / "last_control.png"
        generated_path = debug_dir / "last_generated.png"
        try:
            control_img.save(control_path)
            generated_img.save(generated_path)
            print(f"[DEBUG] Saved control image to {control_path}")
            print(f"[DEBUG] Saved generated image to {generated_path}")
            print("[DEBUG] Generated extrema:", generated_img.getextrema())
        except Exception as exc:  # noqa: BLE001
            print("[WARN] Failed saving debug images:", exc)

        return np.array(control_img), np.array(generated_img)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        print("[ERROR] Generation failed:", exc)
        print(tb)
        raise gr.Error(f"Generation failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ControlNet LineArt Gradio UI")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=0, help="Server port (0=auto)")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    args = parser.parse_args()

    with gr.Blocks() as demo:
        gr.Markdown("# ControlNet LineArt Parameter Tuner")
        gr.Markdown(
            "This UI is for local experimentation only. "
            "Upload a face photo and tweak parameters to inspect the resulting line art."
        )

        with gr.Row():
            image_input = gr.Image(type="pil", label="Input Image")
            with gr.Column():
                prompt = gr.Textbox(
                    value="minimalist black and white line art portrait, clean outlines, white background",
                    label="Prompt",
                )
                negative_prompt = gr.Textbox(
                    value="color, shading, gradient, messy, noisy, abstract shapes",
                    label="Negative Prompt",
                )
                steps = gr.Slider(1, 30, value=6, step=1, label="Steps")
                guidance_scale = gr.Slider(1.0, 15.0, value=3.5, step=0.1, label="Guidance Scale")
                seed = gr.Slider(0, 2**31 - 1, value=42, step=1, label="Seed")
                width = gr.Slider(224, 512, value=320, step=32, label="Width")
                height = gr.Slider(224, 512, value=320, step=32, label="Height")
                control_weight = gr.Slider(0.3, 1.2, value=0.6, step=0.05, label="Control Weight")
                use_coarse = gr.Checkbox(value=False, label="Use Coarse LineArt", info="太め輪郭 (OFFで細線)")
                detect_resolution = gr.Slider(128, 512, value=224, step=16, label="Detect Resolution")
                bin_threshold = gr.Slider(50, 220, value=150, step=5, label="Binary Threshold")
                base_model = gr.Textbox(value=DEFAULT_BASE_MODEL, label="Base Model ID")
                control_model = gr.Textbox(value=DEFAULT_CONTROLNET_MODEL, label="ControlNet Model ID")
                run_btn = gr.Button("Generate")

        with gr.Row():
            control_output = gr.Image(type="pil", label="ControlNet Line Art")
            generated_output = gr.Image(type="pil", label="Generated Line Art")

        run_btn.click(
            fn=infer,
            inputs=[
                image_input,
                prompt,
                negative_prompt,
                steps,
                guidance_scale,
                seed,
                width,
                height,
                control_weight,
                use_coarse,
                detect_resolution,
                bin_threshold,
                base_model,
                control_model,
            ],
            outputs=[control_output, generated_output],
        )

    demo.queue(concurrency_count=1, max_size=4)

    launch_kwargs = {
        "server_name": args.host,
        "share": args.share,
        "show_api": False,
    }
    if args.port > 0:
        launch_kwargs["server_port"] = args.port

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
