import torch
from diffusers import PixArtAlphaPipeline

print("Loading PixArt without CPU offload...")
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-512x512",
    torch_dtype=torch.float16
)
pipe.to("cuda")

print("Running inference without CPU offload...")
generator = torch.Generator("cuda").manual_seed(42)
try:
    image = pipe(
        "a red apple on a wooden table",
        num_inference_steps=5,
        generator=generator
    ).images[0]
    output_path = "/mnt/c/Users/kusha/test_no_offload.png"
    image.save(output_path)
    print(f"Saved {output_path}")
except Exception as e:
    print(f"Error: {e}")
