import torch
from diffusers import StableDiffusion3Pipeline

print("Loading SD3 Medium...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

print("Running inference without custom processor...")
generator = torch.Generator("cuda").manual_seed(42)
image = pipe(
    "a red apple on a wooden table",
    num_inference_steps=5,
    generator=generator
).images[0]

output_path = "/mnt/c/Users/kusha/test_sd3_output.png"
image.save(output_path)
print(f"Saved {output_path}")
