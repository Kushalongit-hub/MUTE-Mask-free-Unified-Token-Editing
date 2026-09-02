import torch
import time

torch.cuda.init()
time.sleep(2)

from diffusers import PixArtAlphaPipeline
import torch.nn.functional as F
import torch.nn as nn

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

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=custom_mask
        )

        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

print("Loading PixArt...")
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-512x512",
    torch_dtype=torch.float16
)
pipe.to("cuda")

active_mask = torch.zeros(1024, dtype=torch.bool, device="cuda")
active_mask[400:450] = True

soundproof_processor = SoundproofAttnProcessor(is_active_mask=active_mask)
pipe.transformer.set_attn_processor(soundproof_processor)

print("Running 1-step inference...")
generator = torch.Generator("cuda").manual_seed(42)
image = pipe(
    "a red apple on a wooden table",
    num_inference_steps=1,
    generator=generator
).images[0]

output_path = "/mnt/c/Users/kusha/soundproof_output.png"
image.save(output_path)
print(f"Saved {output_path}")
