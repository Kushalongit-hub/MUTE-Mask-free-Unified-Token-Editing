import torch
from diffusers import PixArtAlphaPipeline

print("Loading PixArt...")
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-512x512",
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

transformer = pipe.transformer
print("Transformer type:", type(transformer))

# Get first block
block = transformer.transformer_blocks[0]
print("Block type:", type(block))

attn = block.attn1
print("Attn type:", type(attn))
print("Attn attributes:", [a for a in dir(attn) if not a.startswith('_')])

# Check to_out
print("to_out type:", type(attn.to_out))
if hasattr(attn.to_out, '__len__'):
    print("to_out length:", len(attn.to_out))
    for i, layer in enumerate(attn.to_out):
        print(f"  to_out[{i}]:", type(layer))

# Check head dim
print("attn.head_dim:", attn.head_dim)
print("attn.num_heads:", attn.num_heads)

# Check sequence length from config
config = transformer.config
print("Config sample_size:", getattr(config, 'sample_size', None))
print("Config patch_size:", getattr(config, 'patch_size', None))
seq_len = (getattr(config, 'sample_size', 64) // getattr(config, 'patch_size', 2)) ** 2
print("Computed seq_len:", seq_len)
