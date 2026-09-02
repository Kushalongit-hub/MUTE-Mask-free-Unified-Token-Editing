import torch
from diffusers import PixArtAlphaPipeline

print("Loading PixArt...")
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-512x512",
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

transformer = pipe.transformer
block = transformer.transformer_blocks[0]
attn = block.attn1

print("Attn inner_dim:", attn.inner_dim)
print("Attn heads:", attn.heads)
print("Attn head_dim (computed):", attn.inner_dim // attn.heads)

config = transformer.config
print("Config sample_size:", getattr(config, 'sample_size', None))
print("Config patch_size:", getattr(config, 'patch_size', None))
seq_len = (getattr(config, 'sample_size', 64) // getattr(config, 'patch_size', 2)) ** 2
print("Computed seq_len:", seq_len)

# Now let's test a forward pass with the custom processor
from diffusers.models.attention_processor import Attention

class SoundproofAttnProcessor:
    def __init__(self, is_active_mask):
        self.is_active = is_active_mask

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        seq_len = sequence_length
        custom_mask = torch.zeros((seq_len, seq_len), dtype=query.dtype, device=query.device)
        
        anchor_rows = (~self.is_active).unsqueeze(1)
        active_cols = self.is_active.unsqueeze(0)
        custom_mask[anchor_rows & active_cols] = float('-inf')
        
        custom_mask = custom_mask.unsqueeze(0)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=custom_mask
        )

        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

active_mask = torch.zeros(seq_len, dtype=torch.bool, device="cuda")
active_mask[400:450] = True

processor = SoundproofAttnProcessor(is_active_mask=active_mask)

# Monkey-patch one attention layer
attn.set_processor(processor)

# Create dummy input
x = torch.randn(1, seq_len, attn.inner_dim, dtype=torch.float16, device="cuda")
print("Input shape:", x.shape)

# Forward pass
with torch.no_grad():
    out = block(x)
print("Output shape:", out.shape)
print("Forward pass successful!")
