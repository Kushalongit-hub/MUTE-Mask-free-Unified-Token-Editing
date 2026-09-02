import torch
import sys
from diffusers import PixArtAlphaPipeline

class Logger:
    def __init__(self):
        self.f = open("/mnt/c/Users/kusha/pixart_debug.log", "w")
    def write(self, msg):
        self.f.write(msg + "\n")
        self.f.flush()
    def close(self):
        self.f.close()

logger = Logger()

try:
    logger.write("Loading PixArt...")
    pipe = PixArtAlphaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-512x512",
        torch_dtype=torch.float16
    )
    logger.write("Loaded PixArt")
    pipe.enable_model_cpu_offload()
    logger.write("Enabled CPU offload")

    active_mask = torch.zeros(1024, dtype=torch.bool, device="cuda")
    active_mask[400:450] = True
    logger.write("Created active mask")

    from diffusers.models.attention_processor import Attention
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

    soundproof_processor = SoundproofAttnProcessor(is_active_mask=active_mask)
    logger.write("Created processor")

    pipe.transformer.set_attn_processor(soundproof_processor)
    logger.write("Set processor")

    generator = torch.Generator("cuda").manual_seed(42)
    logger.write("Starting inference...")
    image = pipe(
        "a red apple on a wooden table",
        num_inference_steps=5,
        generator=generator
    ).images[0]
    
    output_path = "/mnt/c/Users/kusha/soundproof_output.png"
    image.save(output_path)
    logger.write(f"Saved {output_path}")
except Exception as e:
    logger.write(f"Error: {e}")
    import traceback
    logger.write(traceback.format_exc())
finally:
    logger.close()
