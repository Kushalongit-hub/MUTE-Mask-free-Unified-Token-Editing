import torch
from diffusers import PixArtAlphaPipeline

print("Loading PixArt with attention slicing...")
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-512x512",
    torch_dtype=torch.float16
)
pipe.to("cuda")
pipe.enable_attention_slicing()

print("Running inference...")
generator = torch.Generator("cuda").manual_seed(42)
image = pipe(
    "a red apple on a wooden table",
    num_inference_steps=5,
    generator=generator
).images[0]

output_path = "/mnt/c/Users/kusha/test_slicing.png"
image.save(output_path)
print(f"Saved {output_path}")
