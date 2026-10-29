# Tiny Canvas

A beautiful local text-to-image app using [Segmind Tiny Stable Diffusion](https://huggingface.co/segmind/tiny-sd), a compact open-source diffusion model.

## Run it

```bash
cd tiny_canvas
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

The first image generation downloads the model and caches it locally. Images are saved into `outputs/`.

## Notes

- Works on NVIDIA CUDA, Apple Silicon (MPS), or CPU.
- On CPU, generation can take a while. Keep **Steps** around 8–15 for a quick result.
- This compact model is ideal for experimentation; use a larger Stable Diffusion model if you need higher fidelity.