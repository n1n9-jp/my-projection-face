#!/usr/bin/env python3
"""
Gradio UI for tuning Stable Diffusion + ControlNet (LineArt) parameters.

This UI is intended for local parameter exploration only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import torch
from controlnet_aux import LineartDetector
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    base_model: str,
    controlnet_model: str,
):
    ensure_pipeline(base_model, controlnet_model)
    if pipe is None or detector is None:
        raise RuntimeError("Pipeline or detector not initialized")

    width = int(width)
    height = int(height)
    seed = int(seed)
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
    )

    return control_img, generated_img


def main() -> None:
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
                    value="line art portrait, sketch, clean smooth lines, high quality",
                    label="Prompt",
                )
                negative_prompt = gr.Textbox(
                    value="blurry, messy, noisy, extra limbs",
                    label="Negative Prompt",
                )
                steps = gr.Slider(1, 30, value=15, step=1, label="Steps")
                guidance_scale = gr.Slider(1.0, 15.0, value=7.0, step=0.1, label="Guidance Scale")
                seed = gr.Slider(0, 2**31 - 1, value=42, step=1, label="Seed")
                width = gr.Slider(256, 768, value=512, step=64, label="Width")
                height = gr.Slider(256, 768, value=512, step=64, label="Height")
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
                base_model,
                control_model,
            ],
            outputs=[control_output, generated_output],
        )

    demo.launch()


if __name__ == "__main__":
    main()
