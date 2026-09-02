import torch
from diffusers import StableDiffusionPipeline

print("Loading SD 1.5...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

print("Running inference...")
generator = torch.Generator("cuda").manual_seed(42)
image = pipe(
    "a red apple on a wooden table",
    num_inference_steps=5,
    generator=generator
).images[0]

output_path = "/mnt/c/Users/kusha/test_sd15.png"
image.save(output_path)
print(f"Saved {output_path}")
