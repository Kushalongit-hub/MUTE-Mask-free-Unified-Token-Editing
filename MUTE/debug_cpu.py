import torch
from diffusers import StableDiffusionPipeline

print("Loading SD 1.5 on CPU...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)

print("Running inference on CPU...")
generator = torch.Generator("cpu").manual_seed(42)
image = pipe(
    "a red apple on a wooden table",
    num_inference_steps=3,
    generator=generator
).images[0]

output_path = "/mnt/c/Users/kusha/test_cpu.png"
image.save(output_path)
print(f"Saved {output_path}")
