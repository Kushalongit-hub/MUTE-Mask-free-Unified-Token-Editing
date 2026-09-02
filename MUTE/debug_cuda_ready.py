import torch
import time

print("Initializing CUDA...")
torch.cuda.init()
time.sleep(2)

print("Checking CUDA status...")
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name())
print("Current device:", torch.cuda.current_device())

# Try a simple operation
x = torch.randn(1, device="cuda")
print("Simple CUDA op OK")

# Try importing diffusers
from diffusers import PixArtAlphaPipeline
print("Diffusers imported")

# Load model
print("Loading PixArt...")
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-512x512",
    torch_dtype=torch.float16
)

print("Moving to CUDA...")
try:
    pipe.to("cuda")
    print("pipe.to('cuda') successful")
except Exception as e:
    print(f"pipe.to('cuda') failed: {e}")
