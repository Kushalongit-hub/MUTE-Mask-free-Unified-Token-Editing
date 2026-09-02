import torch
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
)
unet = pipe.unet
unet = unet.to("cuda")

noisy_target = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16)
masked_context = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16)
binary_mask = torch.rand(1, 1, 64, 64, device="cuda", dtype=torch.float16)

stacked_input = torch.cat([noisy_target, masked_context, binary_mask], dim=1)

print(f"Expected U-Net Input Channels: {unet.config.in_channels}")
print(f"Noisy Target Shape:   {list(noisy_target.shape)}")
print(f"Masked Context Shape: {list(masked_context.shape)}")
print(f"Binary Mask Shape:    {list(binary_mask.shape)}")
print(f"Final Stacked Input:  {list(stacked_input.shape)}")
