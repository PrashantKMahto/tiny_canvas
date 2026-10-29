"""Tiny Canvas — a local text-to-image studio powered by a lightweight model."""

from __future__ import annotations

import random
import threading
from pathlib import Path

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline


MODEL_ID = "segmind/tiny-sd"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

_pipeline: StableDiffusionPipeline | None = None
_load_lock = threading.Lock()


def get_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def get_pipeline() -> StableDiffusionPipeline:
    """Load only once, so the model download happens on the first generation."""
    global _pipeline
    with _load_lock:
        if _pipeline is None:
            device, dtype = get_device()
            _pipeline = StableDiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                safety_checker=None,
            )
            _pipeline = _pipeline.to(device)
            _pipeline.enable_attention_slicing()
    return _pipeline


def generate(prompt: str, negative_prompt: str, steps: int, guidance: float, seed: int, progress=gr.Progress()):
    if not prompt or not prompt.strip():
        raise gr.Error("Describe the image you want to create first.")

    chosen_seed = random.randint(0, 2_147_483_647) if seed < 0 else int(seed)
    progress(0.05, desc="Preparing Tiny Canvas…")
    pipe = get_pipeline()
    device, _ = get_device()
    generator = torch.Generator(device=device).manual_seed(chosen_seed)

    def callback(step: int, _timestep: int, _latents: torch.Tensor):
        progress(0.15 + (0.8 * (step + 1) / steps), desc=f"Painting frame {step + 1} of {steps}")

    image = pipe(
        prompt=prompt.strip(),
        negative_prompt=negative_prompt.strip() or None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator,
        height=512,
        width=512,
        callback=callback,
        callback_steps=1,
    ).images[0]

    image_path = OUTPUT_DIR / f"tiny-canvas-{chosen_seed}.png"
    image.save(image_path)
    return image, chosen_seed, str(image_path)


CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@500;600;700;800&display=swap');
:root { --ink: #15251e; --leaf: #1d6b4c; --mint: #dff6df; --cream: #faf9f3; --line: #c8d7c8; }
body, .gradio-container { background: var(--cream) !important; color: var(--ink) !important; font-family: 'DM Mono', monospace !important; }
.gradio-container { max-width: 1180px !important; margin: auto !important; padding: 28px 22px 40px !important; }
.hero { border: 1px solid var(--ink); padding: 30px 34px; background: var(--mint); border-radius: 20px; margin-bottom: 20px; position: relative; overflow: hidden; }
.hero:after { content: '✦'; position: absolute; right: 28px; top: 15px; font: 100px 'Syne'; color: #91c99b; line-height: 1; }
.hero h1 { font: 800 52px/1 'Syne', sans-serif; letter-spacing: -3px; margin: 0 0 10px; position: relative; z-index: 1; }
.hero p { max-width: 650px; margin: 0; font-size: 14px; line-height: 1.6; position: relative; z-index: 1; }
.eyebrow { color: var(--leaf); font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 14px !important; }
.panel { border: 1px solid var(--line); border-radius: 18px; background: #fffefa; padding: 20px !important; }
.panel h2 { font: 700 18px 'Syne', sans-serif; margin: 0 0 16px; }
textarea, input { border-radius: 10px !important; border-color: var(--line) !important; box-shadow: none !important; }
.generate { background: var(--ink) !important; color: #fff !important; border: 0 !important; border-radius: 10px !important; font: 700 15px 'Syne', sans-serif !important; height: 48px !important; }
.generate:hover { background: var(--leaf) !important; }
.quick button { border-radius: 99px !important; border: 1px solid var(--line) !important; background: transparent !important; color: var(--ink) !important; }
.quick button:hover { background: var(--mint) !important; }
.status { color: #48705b; font-size: 11px; margin-top: 8px; }
.output img { border-radius: 14px !important; border: 1px solid var(--ink); }
footer { display: none !important; }
@media (max-width: 700px) { .hero h1 { font-size: 39px; } .hero { padding: 25px; } }
"""


with gr.Blocks(title="Tiny Canvas", css=CSS, theme=gr.themes.Base()) as demo:
    gr.HTML("""
    <section class='hero'>
      <p class='eyebrow'>Local AI image studio · open source</p>
      <h1>Tiny Canvas</h1>
      <p>Turn a thought into a small work of art. Everything runs on your machine using Segmind's Tiny Stable Diffusion model.</p>
    </section>
    """)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes="panel"):
            gr.HTML("<h2>Describe your scene</h2>")
            prompt = gr.Textbox(label="Prompt", placeholder="A quiet glasshouse in a moonlit forest, botanical illustration", lines=4, autofocus=True)
            with gr.Row(elem_classes="quick"):
                b1 = gr.Button("Moonlit garden", size="sm")
                b2 = gr.Button("Retro robot", size="sm")
                b3 = gr.Button("Ocean dream", size="sm")
            negative = gr.Textbox(label="Avoid (optional)", placeholder="blurry, low quality, text", lines=2)
            with gr.Accordion("Tune the image", open=False):
                steps = gr.Slider(1, 30, value=12, step=1, label="Steps")
                guidance = gr.Slider(1, 12, value=7, step=0.5, label="Prompt strength")
                seed = gr.Number(value=-1, precision=0, label="Seed (−1 = surprise me)")
            create = gr.Button("Create image  →", elem_classes="generate")
            gr.HTML("<p class='status'>First run downloads the model once. A GPU speeds things up, but CPU works too.</p>")
        with gr.Column(scale=6, elem_classes="panel"):
            gr.HTML("<h2>Your canvas</h2>")
            output = gr.Image(label="", elem_classes="output", height=480)
            with gr.Row():
                used_seed = gr.Number(label="Seed used", interactive=False)
                download = gr.File(label="Download PNG", interactive=False)

    create.click(generate, [prompt, negative, steps, guidance, seed], [output, used_seed, download])
    b1.click(lambda: "A moonlit garden inside an elegant glasshouse, bioluminescent flowers, dreamy botanical illustration", outputs=prompt)
    b2.click(lambda: "A friendly retro robot tending tiny plants on a space station, warm cinematic light", outputs=prompt)
    b3.click(lambda: "An underwater library with floating books and shafts of sunlight, surreal watercolor", outputs=prompt)

if __name__ == "__main__":
    demo.launch(inbrowser=True)